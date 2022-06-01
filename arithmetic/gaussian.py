import numpy as np
import quimb.tensor as qtn
from .utils import ADD,CP2,scale,parallelized_looped_function
import time
def get_field(ys,tag,nf,iprint=0,cutoff=1e-10):
    if iprint>0:
        print('getting field...')
    ng = len(ys)
    CP = np.zeros((nf,)*3)
    for i in range(nf):
        CP[i,i,i] = 1.

    tn = qtn.TensorNetwork([])
    yi = np.ones((ng,2))
    yi[:,1] = ys.copy()
    ei = np.zeros((nf,2))
    ei[:,0] = np.ones(nf)
    ei[nf-1,1] = 1.
    data = np.einsum('yi,pj,ijk->ypk',yi,ei,CP2)
    tn.add_tensor(qtn.Tensor(data=data,inds=(f'{tag}{nf}','p','k'),tags=f'{tag}{nf}'))
    for j in range(nf-1,0,-1):
        tn[f'{tag}{j+1}'].reindex_({'p':'p_','k':'k_'})
        yi = np.ones((ng,2))
        yi[:,1] = ys.copy()
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
    tl,tr = tn[f'{tag}1'].split(('p','k'),bond_ind=f'{tag}0,1',absorb='left',
                                 cutoff=cutoff)
    tl.modify(tags=f'{tag}0')
    tn.add_tensor(tl)
    tn[f'{tag}1'].modify(data=tr.data,inds=tr.inds)

    if iprint>0:
        print(tn)
    return tn
def get_quadratic(tny,A,tag,iprint=0,cutoff=1e-10):
    if iprint>0:
        print('getting exponent...')
    nf,n = A.shape[0],len(A.shape)
    ng = tny[f'{tag}1'].shape[tny[f'{tag}1'].inds.index(f'{tag}1')]
    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1.
    sCP2 = np.zeros((2,)*3)
    sCP2[0,0,0] = 1./nf
    sCP2[1,1,1] = 1.

    data = np.ones(A.shape+(2,))
    data[...,1] = A
    T1 = qtn.Tensor(data=data,inds=[f'p{i}' for i in range(n)]+['k'])

    tn = qtn.TensorNetwork([])
    for i in range(n):
        T1.reindex_({'k':'i1'})
        T2 = tny[f'{tag}0'].reindex({'p':f'p{i}','k':'i2',
                                     f'{tag}0,1':f'{tag}0,1_{i}'})
        T1 = qtn.tensor_contract(T1,T2,
                                 qtn.Tensor(data=sCP2,inds=('i1','i2','k')))
    tn.add_tensor(T1)
    for j in range(1,nf+1):
        T1 = tny[f'{tag}{j}'].reindex({f'{tag}{j}':'a',
                                       f'{tag}{j-1},{j}':f'{tag}{j-1},{j}_0',
                                       f'{tag}{j},{j+1}':f'{tag}{j},{j+1}_0'})
        T2 = tny[f'{tag}{j}'].reindex({f'{tag}{j}':'b',
                                  f'{tag}{j-1},{j}':f'{tag}{j-1},{j}_1',
                                  f'{tag}{j},{j+1}':f'{tag}{j},{j+1}_1'})
        T = qtn.tensor_contract(tn[f'{tag}{j-1}'],T1,T2,
                                qtn.Tensor(data=CP,inds=('a','b',f'{tag}{j}')))
        lix = ('k',) if j==1 else (f'{tag}{j-1}',f'{tag}{j-2},{j-1}')
        tl,tr = T.split(lix,bond_ind=f'{tag}{j-1},{j}',absorb='right',
                        cutoff=cutoff)
        tn[f'{tag}{j-1}'].modify(data=tl.data,inds=tl.inds)
        tr.modify(tags=f'{tag}{j}')
        tn.add_tensor(tr)
        if iprint>1:
            print(f'bdim({tag}{j-1},{tag}{j})={tr.shape[0]}')

    for j in range(nf-1,-1,-1):
        tn.compress_between(f'{tag}{j}',f'{tag}{j+1}',absorb='left',
                            cutoff=cutoff)

    if iprint>0:
        print(tn)
    return tn
def get_quartic(tny,A,tag,iprint=0,cutoff=1e-10):
    if iprint>0:
        print('getting exponent...')
    nf,n = A.shape[0],len(A.shape)
    ng = tny[f'{tag}1'].shape[tny[f'{tag}1'].inds.index(f'{tag}1')]
    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1.
    sCP2 = np.zeros((2,)*3)
    sCP2[0,0,0] = 1./nf
    sCP2[1,1,1] = 1.

    data = np.ones(A.shape+(2,))
    data[...,1] = A
    T = qtn.Tensor(data=data,inds=[f'p{i}' for i in range(n)]+['i2'])

    tn = tny.copy()
    tn[f'{tag}0'].reindex_({'p':'p0','k':'i1'})
    T = qtn.tensor_contract(tn[f'{tag}0'],T,
                            qtn.Tensor(data=sCP2,inds=('i1','i2','k')))
    tn[f'{tag}0'].modify(data=T.data,inds=T.inds)
    for i in range(1,n):
        if iprint>0:
            print(f'compressing row {i-1},{i}...')
        tn[f'{tag}0'].reindex_({'k':'i1'})
        T = tny[f'{tag}0'].reindex({'p':f'p{i}','k':'i2',f'{tag}0,1':f'{tag}0,1_'})
        T = qtn.tensor_contract(tn[f'{tag}0'],T,
                                qtn.Tensor(data=sCP2,inds=('i1','i2','k')))
        tn[f'{tag}0'].modify(data=T.data,inds=T.inds)
        for j in range(1,nf+1):
            tn[f'{tag}{j}'].reindex_({f'{tag}{j}':'a'})
            T = tny[f'{tag}{j}'].reindex({f'{tag}{j}':'b',
                                          f'{tag}{j-1},{j}':f'{tag}{j-1},{j}_',
                                          f'{tag}{j},{j+1}':f'{tag}{j},{j+1}_'})
            T = qtn.tensor_contract(tn[f'{tag}{j-1}'],tn[f'{tag}{j}'],T,
                                    qtn.Tensor(data=CP,inds=('a','b',f'{tag}{j}')))
            rix = (f'{tag}{j}',) if j==nf else \
                  (f'{tag}{j}',f'{tag}{j},{j+1}',f'{tag}{j},{j+1}_')
            tl,tr = T.split(left_inds=None,right_inds=rix,bond_ind=f'{tag}{j-1},{j}',
                            absorb='right',cutoff=cutoff)
            tn[f'{tag}{j-1}'].modify(data=tl.data,inds=tl.inds)
            tn[f'{tag}{j}'].modify(data=tr.data,inds=tr.inds)
            if iprint>1:
                print(f'bdim({tag}{j-1},{tag}{j})={tr.shape[0]}')

        for j in range(nf-1,-1,-1):
            tn.compress_between(f'{tag}{j}',f'{tag}{j+1}',absorb='left',
                                cutoff=cutoff)

        if iprint>0:
            print(tn)
    return tn
def add_exponent(tnA,tnB,tag,iprint=0,cutoff=1e-10):
    if iprint>0:
        print('adding exponent...')
    nf = tnA.num_tensors-1
    ng = tnA[f'{tag}1'].shape[tnA[f'{tag}1'].inds.index(f'{tag}1')]
    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1.

    tn = qtn.TensorNetwork([]) 
    T1 = tnA[f'{tag}0'].reindex({'k':'i1'})
    T2 = tnB[f'{tag}0'].reindex({'k':'i2',f'{tag}0,1':f'{tag}0,1_'})
    T = qtn.tensor_contract(T1,T2,qtn.Tensor(data=ADD,inds=('i1','i2','k')))
    tn.add_tensor(T)
    for j in range(1,nf+1):
        T1 = tnA[f'{tag}{j}'].reindex({f'{tag}{j}':'a'})
        T2 = tnB[f'{tag}{j}'].reindex({f'{tag}{j}':'b',
                                       f'{tag}{j-1},{j}':f'{tag}{j-1},{j}_',
                                       f'{tag}{j},{j+1}':f'{tag}{j},{j+1}_'})
        T = qtn.tensor_contract(tn[f'{tag}{j-1}'],T1,T2,
                                qtn.Tensor(data=CP,inds=('a','b',f'{tag}{j}')))
        lix = ('k',) if j==1 else (f'{tag}{j-1}',f'{tag}{j-2},{j-1}')
        tl,tr = T.split(lix,bond_ind=f'{tag}{j-1},{j}',absorb='right',
                        cutoff=cutoff)
        tn[f'{tag}{j-1}'].modify(data=tl.data,inds=tl.inds)
        tr.modify(tags=f'{tag}{j}')
        tn.add_tensor(tr)
        if iprint>1:
            print(f'bdim({tag}{j-1},{tag}{j})={tr.shape[0]}')

    for j in range(nf-1,-1,-1):
        tn.compress_between(f'{tag}{j}',f'{tag}{j+1}',absorb='left',
                            cutoff=cutoff)

    if iprint>0:
        print(tn)
    return tn

def parse(data,sign,log10_a0=0.):
    data,sign = np.array([log10_a0]+list(data)),np.array([1]+list(sign))
    idxs = np.argsort(data)
    data,sign = data[idxs],sign[idxs]
    if data[-1]<1.:
        return np.log10(sum([s*10.**e for s,e in zip(sign,data)]))
    while len(data)>2:
        data,sign = list(data),list(sign)
        e0 = data.pop()
        e1 = data.pop()
        s0 = sign.pop()
        s1 = sign.pop()
        print(f'len={len(data)+2},(s0,s1)={(s0,s1)},(e0,e1)={(e0,e1)}')
        fac = s0+s1*10.**(e1-e0)
        ei = e0 + np.log10(abs(fac))
        si = 1 if fac>0 else -1
        if data[-1]<1.:
            sum_ = sum([s*10.**e for s,e in zip(sign,data)])
            sj,ej = sum_/abs(sum_), np.log10(abs(sum_))
            return ei+np.log10(si+sj*10.**(ej-ei))        
        data.append(ei)
        sign.append(si)

        data,sign = np.array(data),np.array(sign)
        idxs = np.argsort(data)
        data,sign = data[idxs],sign[idxs]

def trace_field(tn,tag,nf):
    for j in range(1,nf-1):
        tn.contract_tags((f'{tag}{j}',f'{tag}{j+1}'),which='any',inplace=True)
        fac = np.amax(np.absolute(tn[f'{tag}{j}'].data))
        tn[f'{tag}{j}'].modify(data=tn[f'{tag}{j}'].data/fac)
        tn.exponent += np.log10(fac)
    out = tn.contract()
    exp = np.log10(abs(out)) + tn.exponent
    if out<0.:
        print(f'contract={out},exponent={tn.exponent}')
    return exp
def trace_pol_compress_row(tnx,tag,tr,n,trace_first=True,iprint=0,
                           exp_cut=-6.,cutoff=1e-10,max_bond=None):
    # taylor series of exponential
    tnx = tnx.copy()
    tnx.add_tensor(qtn.Tensor(data=np.array([0.,1.]),inds=('k',),tags=f'{tag}0'))
    tnx.contract_tags((f'{tag}0',f'{tag}1'),which='any',inplace=True)
    tnx[f'{tag}1'].modify(tags=f'{tag}1')
    nf = tnx.num_tensors
    for j in range(1,nf):
        tnx.compress_between(f'{tag}{j}',f'{tag}{j+1}',absorb='right',cutoff=cutoff)
    for j in range(nf-1,0,-1):
        tnx.compress_between(f'{tag}{j}',f'{tag}{j+1}',absorb='left',cutoff=cutoff)
    if iprint>1:
        print(tnx)
 
    ng = tnx[f'{tag}1'].shape[tnx[f'{tag}1'].inds.index(f'{tag}1')]
    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1.

    tn = tnx.copy()
    cap = tnx.copy()
    start_time = time.time()

    tni = tn.copy()
    for j in range(1,nf+1):
        tni.add_tensor(qtn.Tensor(data=tr[j],inds=(f'{tag}{j}',),tags=f'{tag}{j}'))
    data = [trace_field(tni,tag,nf)]
    if iprint>0:
        print(f'i=1,data={data[-1]},time={time.time()-start_time}')

    tn_ = tn if trace_first else cap
    for j in range(1,nf+1):
        tn_[f'{tag}{j}'].reindex_({f'{tag}{j}':'a'})
        T = qtn.tensor_contract(tn_[f'{tag}{j}'],
                                qtn.Tensor(data=CP,inds=('a','b',f'{tag}{j}')),
                                qtn.Tensor(data=tr[j],inds=('b',)))
        tn_[f'{tag}{j}'].modify(data=T.data,inds=T.inds)
    try:
        for j in range(1,nf):
            tn_.compress_between(f'{tag}{j}',f'{tag}{j+1}',absorb='right',
                                 cutoff=cutoff)
        for j in range(nf-1,0,-1):
            tn_.compress_between(f'{tag}{j}',f'{tag}{j+1}',absorb='left',
                                 cutoff=cutoff)
    except ValueError:
        tn_ = tn_
    if iprint>1:
        print(tn_)

    for i in range(2,n+1): 
        tn.exponent -= np.log10(i)
        if iprint>1:
            print(tn)
        tni = tn.copy()
        tni.add_tensor_network(cap,check_collisions=True)
        data.append(trace_field(tni,tag,nf))
        if iprint>0:
            print(f'i={i},data={data[-1]},time={time.time()-start_time}')
        if data[-1]<exp_cut:
            break
        if i<n: 
            tn = compress_row_1step(tn,tnx,tag,cutoff=cutoff,max_bond=max_bond)
    return data
def compress_row_1step(tn,tnx,tag,cutoff=1e-10,max_bond=None):
    nf = tnx.num_tensors
    ng = tnx[f'{tag}1'].shape[tnx[f'{tag}1'].inds.index(f'{tag}1')]

    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1.

    for j in range(1,nf+1):
        tn[f'{tag}{j}'].reindex_({f'{tag}{j}':'a'})
        T = tnx[f'{tag}{j}'].reindex({f'{tag}{j}':'b',
                                      f'{tag}{j-1},{j}':f'{tag}{j-1},{j}_',
                                      f'{tag}{j},{j+1}':f'{tag}{j},{j+1}_'})
        T = qtn.tensor_contract(tn[f'{tag}{j}'],T,
                                qtn.Tensor(data=CP,inds=('a','b',f'{tag}{j}')))
        tn[f'{tag}{j}'].modify(data=T.data,inds=T.inds)
    tn.fuse_multibonds_()
    tn = scale(tn)
    try:
        for j in range(1,nf):
            #tn.compress_between(f'{tag}{j}',f'{tag}{j+1}',absorb='right',
            #                    cutoff=cutoff)
            tn.canonize_between(f'{tag}{j}',f'{tag}{j+1}',absorb='right')
        for j in range(nf-1,0,-1):
            tn.compress_between(f'{tag}{j}',f'{tag}{j+1}',absorb='left',
                                cutoff=cutoff,max_bond=max_bond)
    except ValueError:
        tn = tn
    for j in range(1,nf+1):
        tn[f'{tag}{j}'].reindex_({f'{tag}{j-1},{j}_':f'{tag}{j-1},{j}',
                                  f'{tag}{j},{j+1}_':f'{tag}{j},{j+1}'})
    return tn

def trace_pol_compress_col(tnx,tag,new_tag,tr,n,iprint=0,cutoff=1e-10,max_bond=None):
    # taylor series of exponential
    tnx = tnx.copy()
    tnx.add_tensor(qtn.Tensor(data=np.array([0.,1.]),inds=('k',),tags=f'{tag}0'))
    tnx.contract_tags((f'{tag}0',f'{tag}1'),which='any',inplace=True)
    tnx[f'{tag}1'].modify(tags=f'{tag}1')
    nf = tnx.num_tensors
    for j in range(1,nf):
        tnx.compress_between(f'{tag}{j}',f'{tag}{j+1}',absorb='right',cutoff=cutoff)
    for j in range(nf-1,0,-1):
        tnx.compress_between(f'{tag}{j}',f'{tag}{j+1}',absorb='left',cutoff=cutoff)
    if iprint>1:
        print(tnx)

    ng = tnx[f'{tag}1'].shape[tnx[f'{tag}1'].inds.index(f'{tag}1')]
    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1.

    tni = tnx.copy()
    for j in range(1,nf+1):
        tni.add_tensor(qtn.Tensor(data=tr[j],inds=(f'{tag}{j}',),tags=f'{tag}{j}'))
    data = [trace_field(tni,tag,nf)]
    if iprint>0:
        print(f'i=1,data={data[-1]}')

    iterate_over = []
    for i in range(2,n+1):
        iprint_ = iprint if i==n else 0
        iterate_over += [('left',i,0),('right',i,iprint_)]
    fxn = compress_col_wrapper
    args = [tnx,tag,new_tag,tr]
    kwargs = {'cutoff':cutoff,'max_bond':max_bond}
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    tn_map = dict()
    for side,i,tn in ls:
        tn_map[side,i] = tn
    for i in range(2,n+1):
        tn1,tn2 = tn_map['left',i],tn_map['right',i]
        tn1.add_tensor_network(tn2,check_collisions=True)
        tn1.exponent -= sum([np.log10(k) for k in range(2,i+1)])
        data.append(trace_field(tn1,new_tag,i))
        if iprint>0:
            print(f'i={i},data={data[-1]}')
    return data 
def compress_col_wrapper(info,tnx,tag,new_tag,tr,cutoff=1e-10,max_bond=None):
    side,n,iprint = info
    tnx = tnx.copy()
    nf = tnx.num_tensors
    #fac = 10.**(-sum([np.log10(i) for i in range(2,n+1)])/(n*nf))
    #for j in range(1,nf+1):
    #    tnx[f'{tag}{j}'].modify(data=tnx[f'{tag}{j}'].data*fac)
    if side=='right':
        tn = compress_pol_from_right(n,tnx,tag,new_tag,tr,iprint=iprint,
                                       cutoff=cutoff,max_bond=max_bond)
    else:
        tn = compress_pol_from_left(n,tnx,tag,new_tag,tr,iprint=iprint,
                                      cutoff=cutoff,max_bond=max_bond)
    return side,n,tn
def compress_pol_from_right(n,tnx,tag,new_tag,tr,iprint=0,cutoff=1e-10,max_bond=None):
    nf = tnx.num_tensors
    stop = nf // 2
    ng = tnx[f'{tag}1'].shape[tnx[f'{tag}1'].inds.index(f'{tag}1')]

    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1.
 
    def get_col(j):
        col = qtn.TensorNetwork([])
        for i in range(1,n+1):
            T = tnx[f'{tag}{j}'].reindex({f'{tag}{j-1},{j}':f'{new_tag}{i}',
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

    start_time = time.time()
    tn = get_col(nf)
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
    nf = tnx.num_tensors
    stop = nf // 2
    ng = tnx[f'{tag}1'].shape[tnx[f'{tag}1'].inds.index(f'{tag}1')]

    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1. 

    def get_col(j):
        col = qtn.TensorNetwork([])
        for i in range(1,n+1):
            T = tnx[f'{tag}{j}'].reindex({f'{tag}{j-1},{j}':f'{i}_',
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

    start_time = time.time()
    tn = get_col(1)
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

