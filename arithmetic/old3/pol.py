import numpy as np
import quimb.tensor as qtn
from .utils import ADD,CP2,scale,parallelized_looped_function
from .gaussian_mps import compress_row_1step 
import time
def get_sum_scalar(ys,tag):
    nf,ng = len(ys),len(ys[1])
    tn = qtn.TensorNetwork([])
    for j in range(1,nf+1):
        data = np.ones((2,ng))
        data[1,:] = ys[j].copy()
        inds = f'{tag}{j},{j+1}',f'{tag}{j}'
        if j>1:
            data = np.einsum('ix,ijk->jkx',data,ADD)
            inds = (f'{tag}{j-1},{j}',) + inds
        if j==nf:
            data = np.einsum('jkx,k->jx',data,np.array([0.,1.]))
            inds = inds[0],inds[-1]
        tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags=f'{tag}{j}'))
    return tn
def get_sum(ys,tag,iprint=0,cutoff=1e-10):
    if iprint>0:
        print('getting field...')
    nf,ng = len(ys),len(ys[1])
    CP = np.zeros((nf,)*3)
    for i in range(nf):
        CP[i,i,i] = 1.

    tn = qtn.TensorNetwork([])
    yi = np.ones((ng,2))
    yi[:,1] = ys[nf].copy()
    ei = np.zeros((nf,2))
    ei[:,0] = np.ones(nf)
    ei[nf-1,1] = 1.
    data = np.einsum('yi,pj,ijk->ypk',yi,ei,CP2)
    tn.add_tensor(qtn.Tensor(data=data,inds=(f'{tag}{nf}','p','k'),tags=f'{tag}{nf}'))
    for j in range(nf-1,0,-1):
        tn[f'{tag}{j+1}'].reindex_({'p':'p_','k':'k_'})
        yi = np.ones((ng,2))
        yi[:,1] = ys[j].copy()
        ei = np.zeros((nf,2))
        ei[:,0] = np.ones(nf)
        ei[j-1,1] = 1.
        data = np.einsum('yi,ri,rqp,ijk->ypkqj',yi,ei,CP,ADD)
        tn.add_tensor(qtn.Tensor(data=data,inds=(f'{tag}{j}','p','k','p_','k_'),
                                 tags=f'{tag}{j}'))
        T = qtn.tensor_contract(tn[f'{tag}{j}'],tn[f'{tag}{j+1}'])
        tl,tr = T.split((f'{tag}{j}','p','k'),bond_ind=f'{tag}{j},{j+1}',
                        absorb='left',cutoff=cutoff)
        tn[f'{tag}{j}'].modify(data=tl.data,inds=tl.inds)
        tn[f'{tag}{j+1}'].modify(data=tr.data,inds=tr.inds)
        if iprint>1:
            print(f'bdim({tag}{j+1},{tag}{j})={tr.shape[0]}')
    tn.add_tensor(qtn.Tensor(data=np.ones(nf),inds=('p',),tags=f'{tag}1'))
    tn.add_tensor(qtn.Tensor(data=np.array([0.,1.]),inds=('k',),tags=f'{tag}1'))
    tn.contract_tags(f'{tag}1',which='any',inplace=True)
    for j in range(1,nf):
        tn.compress_between(f'{tag}{j}',f'{tag}{j+1}',absorb='right',
                            cutoff=cutoff)

    if iprint>0:
        print(tn)
    return tn
def trace_field(tn,tag,nf):
    for j in range(1,nf-1):
        tn.contract_tags((f'{tag}{j}',f'{tag}{j+1}'),which='any',inplace=True)
        fac = np.amax(np.absolute(tn[f'{tag}{j}'].data))
        tn[f'{tag}{j}'].modify(data=tn[f'{tag}{j}'].data/fac)
        tn.exponent += np.log10(fac)
    out = tn.contract()
    return out/abs(out), np.log10(abs(out)) + tn.exponent
def trace_pol_compress_row(tnx,tag,tr,iprint=0,cutoff=1e-10,max_bond=None):
    n,nf = len(tnx),tnx[1].num_tensors
    n1 = n//2
    tnx1,tnx2 = dict(),dict()
    for i in range(1,n1+1):
        tnx1[i] = tnx[i]
    for i in range(n1+1,n+1):
        tnx2[i-n1] = tnx[i]
    if n1>=n-n1:
        iprint1,iprint2 = iprint,0
    else:
        iprint1,iprin2 = 0,iprint
    iterate_over = [(tnx1,iprint1),(tnx2,iprint2)]
    fxn = compress_row_wrapper
    args = [tag,tr]
    kwargs = {'cutoff':cutoff,'max_bond':max_bond}
    tn1,tn2 = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    tn1.add_tensor_network(tn2,check_collisions=True)
    return trace_field(tn1,tag,nf)
    
def compress_row_wrapper(info,tag,tr,cutoff=1e-10,max_bond=None):
    tnx,iprint = info
    n,nf,ng = len(tnx),tnx[1].num_tensors,len(tr[1])
    
    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1.

    tn = tnx[1].copy()
    for j in range(1,nf+1):
        tn[f'{tag}{j}'].reindex_({f'{tag}{j}':'a'})
        T = qtn.tensor_contract(tn[f'{tag}{j}'],
                                qtn.Tensor(data=CP,inds=('a','b',f'{tag}{j}')),
                                qtn.Tensor(data=np.sqrt(tr[j]),inds=('b',)))
        tn[f'{tag}{j}'].modify(data=T.data,inds=T.inds)
    try:
        for j in range(1,nf):
            tn.compress_between(f'{tag}{j}',f'{tag}{j+1}',absorb='right',
                                 cutoff=cutoff)
        for j in range(nf-1,0,-1):
            tn.compress_between(f'{tag}{j}',f'{tag}{j+1}',absorb='left',
                                 cutoff=cutoff)
    except ValueError:
        tn = tn

    start_time = time.time()
    for i in range(2,n+1): 
        tn = compress_row_1step(tn,tnx[i].copy(),tag,cutoff=cutoff,max_bond=max_bond)
        if iprint>0:
            print(f'i={i},exponent={tn.exponent},time={time.time()-start_time}')
            print(tn)
    return tn
def trace_pol_compress_col(tnx,tag,new_tag,tr,iprint=0,cutoff=1e-10,max_bond=None):
    n = len(tnx)
    iterate_over = ['left','right']
    fxn = compress_col_wrapper
    args = [n,tnx,tag,new_tag,tr]
    kwargs = {'iprint':iprint,'cutoff':cutoff,'max_bond':max_bond}
    tn1,tn2 = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    tn1.add_tensor_network(tn2,check_collisions=True)
    return trace_field(tn1,new_tag,n)
def compress_col_wrapper(side,n,tnx,tag,new_tag,tr,iprint=0,
                         cutoff=1e-10,max_bond=None):
    if side=='right':
        tn = compress_pol_from_right(n,tnx,tag,new_tag,tr,iprint=iprint,
                                       cutoff=cutoff,max_bond=max_bond)
    else:
        tn = compress_pol_from_left(n,tnx,tag,new_tag,tr,iprint=0,
                                      cutoff=cutoff,max_bond=max_bond)
    return tn
def compress_pol_from_right(n,tnx,tag,new_tag,tr,iprint=0,cutoff=1e-10,max_bond=None):
    n,nf,ng = len(tnx),tnx[1].num_tensors,len(tr[1])
    stop = nf // 2

    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1.
 
    def get_col(j):
        col = qtn.TensorNetwork([])
        for i in range(1,n+1):
            T = tnx[i][f'{tag}{j}'].reindex({f'{tag}{j-1},{j}':f'{new_tag}{i}',
                                          f'{tag}{j},{j+1}':f'{i}_'})
            if i==1:
                T.reindex_({f'{tag}{j}':f'{new_tag}1,2'})
            else:
                T = qtn.tensor_contract(T,qtn.Tensor(data=CP,
                    inds=(f'{tag}{j}',f'{new_tag}{i-1},{i}',f'{new_tag}{i},{i+1}')))
            if i==n:
                T = qtn.tensor_contract(T,
                    qtn.Tensor(data=tr[j],inds=(f'{new_tag}{i},{i+1}',)))
            T.modify(tags=f'{new_tag}{i}')
            col.add_tensor(T)
        return col

    tn = get_col(nf)
    start_time = time.time()
    for j in range(nf-1,stop,-1):
        col = get_col(j)
        for i in range(1,n+1):
            tn[f'{new_tag}{i}'].reindex_({f'{new_tag}{i}':f'{i}_'})
            col[f'{new_tag}{i}'].reindex_({
                 f'{new_tag}{i-1},{i}':f'{new_tag}{i-1},{i}_',
                 f'{new_tag}{i},{i+1}':f'{new_tag}{i},{i+1}_'})
        tn.add_tensor_network(col,check_collisions=False)
        for i in range(1,n+1):
            tn.contract_tags(f'{new_tag}{i}',which='any',inplace=True)
        tn.fuse_multibonds_()
        tn = scale(tn)
        try:
            for i in range(1,n):
                #tn.compress_between(f'{new_tag}{i}',f'{new_tag}{i+1}',
                #                    absorb='right',cutoff=cutoff)
                tn.canonize_between(f'{new_tag}{i}',f'{new_tag}{i+1}',
                                    absorb='right')
            for i in range(n-1,0,-1):
                tn.compress_between(f'{new_tag}{i}',f'{new_tag}{i+1}',
                                    absorb='left',cutoff=cutoff,max_bond=max_bond)
        except ValueError:
            tn = tn
        for i in range(1,n+1):
            tn[f'{new_tag}{i}'].reindex_({
                f'{new_tag}{i-1},{i}_':f'{new_tag}{i-1},{i}',
                f'{new_tag}{i},{i+1}_':f'{new_tag}{i},{i+1}'})
        if iprint>0:
            print(f'j={j},exponent={tn.exponent},time={time.time()-start_time}')
            print(tn)
    return tn
def compress_pol_from_left(n,tnx,tag,new_tag,tr,iprint=0,cutoff=1e-10,max_bond=None):
    n,nf,ng = len(tnx),tnx[1].num_tensors,len(tr[1])
    stop = nf // 2

    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1. 

    def get_col(j):
        col = qtn.TensorNetwork([])
        for i in range(1,n+1):
            T = tnx[i][f'{tag}{j}'].reindex({f'{tag}{j-1},{j}':f'{i}_',
                                          f'{tag}{j},{j+1}':f'{new_tag}{i}'})
            if i==1:
                T.reindex_({f'{tag}{j}':f'{new_tag}1,2'})
            else:
                T = qtn.tensor_contract(T,qtn.Tensor(data=CP,
                    inds=(f'{tag}{j}',f'{new_tag}{i-1},{i}',f'{new_tag}{i},{i+1}')))
            if i==n:
                T = qtn.tensor_contract(T,
                    qtn.Tensor(data=tr[j],inds=(f'{new_tag}{i},{i+1}',)))
            T.modify(tags=f'{new_tag}{i}')
            col.add_tensor(T)
        return col

    tn = get_col(1) 
    start_time = time.time()
    for j in range(2,stop+1):
        col = get_col(j)
        for i in range(1,n+1):
            tn[f'{new_tag}{i}'].reindex_({f'{new_tag}{i}':f'{i}_'})
            col[f'{new_tag}{i}'].reindex_({
                 f'{new_tag}{i-1},{i}':f'{new_tag}{i-1},{i}_',
                 f'{new_tag}{i},{i+1}':f'{new_tag}{i},{i+1}_'})
        tn.add_tensor_network(col,check_collisions=False)
        for i in range(1,n+1):
            tn.contract_tags(f'{new_tag}{i}',which='any',inplace=True)
        tn.fuse_multibonds_()
        tn = scale(tn)
        try:
            for i in range(1,n):
                #tn.compress_between(f'{new_tag}{i}',f'{new_tag}{i+1}',
                #                    absorb='right',cutoff=cutoff)
                tn.canonize_between(f'{new_tag}{i}',f'{new_tag}{i+1}',
                                    absorb='right')
            for i in range(n-1,0,-1):
                tn.compress_between(f'{new_tag}{i}',f'{new_tag}{i+1}',
                                    absorb='left',cutoff=cutoff,max_bond=max_bond)
        except ValueError:
            tn = tn
        for i in range(1,n+1):
            tn[f'{new_tag}{i}'].reindex_({
                f'{new_tag}{i-1},{i}_':f'{new_tag}{i-1},{i}',
                f'{new_tag}{i},{i+1}_':f'{new_tag}{i},{i+1}'})
        if iprint>0:
            print(f'j={j},exponent={tn.exponent},time={time.time()-start_time}')
            print(tn)
    return tn

