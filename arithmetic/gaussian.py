import numpy as np
import quimb.tensor as qtn
from .utils import ADD,CP2,scale
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

def trace_pol_compress_col(tnx,tag,new_tag,tr,coeff,iprint=0,**compress_opts):
    n = len(coeff)-1
    tnx = tnx.copy()
    tnx.add_tensor(qtn.Tensor(data=np.array([0.,1.]),inds=('k',),tags=f'{tag}0'))
    tnx.contract_tags((f'{tag}0',f'{tag}1'),which='any',inplace=True)
    tnx[f'{tag}1'].modify(tags=f'{tag}1')
    tnx.equalize_norms_()
    tnx.balance_bonds_()
    nf = tnx.num_tensors

    out = 0.
    for (i,ai) in coeff.items():
        if i==0:
            outi = ai if isinstance(ai,float) else ai[0]*10.**ai[1]
            sign = 1.
        else:
            tni = tnx.copy()
            if isinstance(ai,float):
                ai_abs = abs(ai)
                sign = ai/ai_abs
                exp = np.log10(ai_abs)
            else:
                sign,exp = ai 
            fac = 10.**(exp/(i*nf))
            for j in range(1,nf+1):
                tni[f'{tag}{j}'].modify(data=tni[f'{tag}{j}'].data*fac)
            if i==1:
                for j in range(1,nf+1):
                    tni.add_tensor(qtn.Tensor(data=tr[j],inds=(f'{tag}{j}',),
                                              tags=f'{tag}{j}'))
                outi = tni.contract()
            else: 
                tn1 = compress_pol_from_right(tni,i,tag,new_tag,tr,iprint=iprint,
                                              **compress_opts)
                tn2 = compress_pol_from_left(tni,i,tag,new_tag,tr,iprint=iprint,
                                             **compress_opts)
                tn1.add_tensor_network(tn2,check_collisions=True)
                outi = tn1.contract()*10.**tn1.exponent
        out += sign*outi
        if iprint>0:
            print(f'order={i},outi={outi},out={out}')
    return out
def compress_pol_from_right(tni,n,tag,new_tag,tr,iprint=0,**compress_opts):
    nf = tni.num_tensors
    stop = nf // 2
    ng = tni[f'{tag}1'].shape[tni[f'{tag}1'].inds.index(f'{tag}1')]
    if iprint>1:
        print(f'{nf-stop} right cols')

    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1.
 
    def get_col(j):
        col = qtn.TensorNetwork([])
        for i in range(n):
            T = tni[f'{tag}{j}'].reindex({f'{tag}{j-1},{j}':f'{new_tag}{i}',
                                          f'{tag}{j},{j+1}':f'{i}_'})
            if i==0:
                T.reindex_({f'{tag}{j}':f'{new_tag}0,1'})
            else:
                T = qtn.tensor_contract(T,qtn.Tensor(data=CP,
                    inds=(f'{tag}{j}',f'{new_tag}{i-1},{i}',f'{new_tag}{i},{i+1}')))
            if i==n-1:
                T = qtn.tensor_contract(T,
                    qtn.Tensor(data=tr[j],inds=(f'{new_tag}{i},{i+1}',)))
            T.modify(tags=f'{new_tag}{i}')
            col.add_tensor(T)
        return col

    tn = get_col(nf)
    for j in range(nf-1,stop,-1):
        if iprint>1:
            print(f'compressing col {j+1},{j}...')
        col = get_col(j)
        for i in range(n):
            T1 = tn[f'{new_tag}{i}'].reindex_({f'{new_tag}{i}':f'{i}_'})
            T2 = col[f'{new_tag}{i}'].reindex_({
                 f'{new_tag}{i-1},{i}':f'{new_tag}{i-1},{i}_',
                 f'{new_tag}{i},{i+1}':f'{new_tag}{i},{i+1}_'})
        tn.add_tensor_network(col,check_collisions=False)
        for i in range(n):
            tn.contract_tags(f'{new_tag}{i}',which='any',inplace=True)
        tn.fuse_multibonds_()

        try:
            for i in range(n-1):
                tn.compress_between(f'{new_tag}{i}',f'{new_tag}{i+1}',
                                    absorb='right',**compress_opts)
            for i in range(n-2,-1,-1):
                tn.compress_between(f'{new_tag}{i}',f'{new_tag}{i+1}',
                                    absorb='left',**compress_opts)
        except ValueError:
            tn = tn
        tn = scale(tn)

        if iprint>1:
            print(f'exponent={tn.exponent}')
            print(tn)
    return tn
def compress_pol_from_left(tni,n,tag,new_tag,tr,iprint=0,**compress_opts):
    nf = tni.num_tensors
    stop = nf // 2
    ng = tni[f'{tag}1'].shape[tni[f'{tag}1'].inds.index(f'{tag}1')]
    if iprint>1:
        print(f'{stop+1} left cols')

    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1. 

    def get_col(j):
        col = qtn.TensorNetwork([])
        for i in range(n):
            T = tni[f'{tag}{j}'].reindex({f'{tag}{j-1},{j}':f'{i}_',
                                          f'{tag}{j},{j+1}':f'{new_tag}{i}'})
            if i==0:
                T.reindex_({f'{tag}{j}':f'{new_tag}0,1'})
            else:
                T = qtn.tensor_contract(T,qtn.Tensor(data=CP,
                    inds=(f'{tag}{j}',f'{new_tag}{i-1},{i}',f'{new_tag}{i},{i+1}')))
            if i==n-1:
                T = qtn.tensor_contract(T,
                    qtn.Tensor(data=tr[j],inds=(f'{new_tag}{i},{i+1}',)))
            T.modify(tags=f'{new_tag}{i}')
            col.add_tensor(T)
        return col

    tn = get_col(1) 
    for j in range(2,stop+1):
        if iprint>1:
            print(f'compressing col {j-1},{j}...')
        col = get_col(j)
        for i in range(n):
            T1 = tn[f'{new_tag}{i}'].reindex_({f'{new_tag}{i}':f'{i}_'})
            T2 = col[f'{new_tag}{i}'].reindex_({
                 f'{new_tag}{i-1},{i}':f'{new_tag}{i-1},{i}_',
                 f'{new_tag}{i},{i+1}':f'{new_tag}{i},{i+1}_'})
        tn.add_tensor_network(col,check_collisions=False)
        for i in range(n):
            tn.contract_tags(f'{new_tag}{i}',which='any',inplace=True)
        tn.fuse_multibonds_()

        try:
            for i in range(n-1):
                tn.compress_between(f'{new_tag}{i}',f'{new_tag}{i+1}',
                                    absorb='right',**compress_opts)
            for i in range(n-2,-1,-1):
                tn.compress_between(f'{new_tag}{i}',f'{new_tag}{i+1}',
                                    absorb='left',**compress_opts)
        except ValueError:
            tn = tn
        tn = scale(tn)

        if iprint>1:
            print(f'exponent={tn.exponent}')
            print(tn)
    return tn

def compress_pol_from_top(tni,n,tag,tr,iprint=0,**compress_opts):
    nf = tni.num_tensors
    stop = n // 2
    ng = tni[f'{tag}1'].shape[tni[f'{tag}1'].inds.index(f'{tag}1')]
    if iprint>0:
        print(f'{n-stop} top rows')

    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1.

    tn = tni.copy()
    for i in range(n-1,stop,-1):
        if iprint>0:
            print(f'compressing row {i+1},{i}...')
        for j in range(1,nf+1):
            T1 = tn[f'{tag}{j}'].reindex_({f'{tag}{j}':'a'})
            T2 = tni[f'{tag}{j}'].reindex({f'{tag}{j}':'b',
                                            f'{tag}{j-1},{j}':f'{tag}{j-1},{j}_',
                                            f'{tag}{j},{j+1}':f'{tag}{j},{j+1}_'})
            T = qtn.tensor_contract(T1,T2,qtn.Tensor(data=CP,inds=('a','b',f'{tag}{j}')))
            T1.modify(data=T.data,inds=T.inds)
        tn.fuse_multibonds_()
        try:
            for j in range(1,nf):
                tn.compress_between(f'{tag}{j}',f'{tag}{j+1}',absorb='right',**compress_opts)
            for j in range(nf-1,0,-1):
                tn.compress_between(f'{tag}{j}',f'{tag}{j+1}',absorb='left',**compress_opts)
        except ValueError:
            tn = tn
        tn = scale(tn)
        if iprint>0:
            print(f'exponent={tn.exponent}')
            print(tn)
    return tn
def compress_pol_from_bottom(tni,n,tag,tr,iprint=0,**compress_opts):
    nf = tni.num_tensors
    stop = n // 2
    ng = tni[f'{tag}1'].shape[tni[f'{tag}1'].inds.index(f'{tag}1')]
    if iprint>0:
        print(f'{stop} bottom rows')

    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1.

    tn = tni.copy()
    for j in range(1,nf+1):
        T1 = tn[f'{tag}{j}'].reindex_({f'{tag}{j}':'a'})
        T = qtn.tensor_contract(T1,qtn.Tensor(data=CP,inds=('a','b',f'{tag}{j}')),
                                   qtn.Tensor(data=tr[j],inds=('b',)))
        T1.modify(data=T.data,inds=T.inds)

    for i in range(2,stop+1):
        if iprint>0:
            print(f'compressing row {i-1},{i}...')
        for j in range(1,nf+1):
            T1 = tn[f'{tag}{j}'].reindex_({f'{tag}{j}':'a'})
            T2 = tni[f'{tag}{j}'].reindex({f'{tag}{j}':'b',
                                            f'{tag}{j-1},{j}':f'{tag}{j-1},{j}_',
                                            f'{tag}{j},{j+1}':f'{tag}{j},{j+1}_'})
            T = qtn.tensor_contract(T1,T2,qtn.Tensor(data=CP,inds=('a','b',f'{tag}{j}')))
            T1.modify(data=T.data,inds=T.inds)
        tn.fuse_multibonds_()
        try:
            for j in range(1,nf):
                tn.compress_between(f'{tag}{j}',f'{tag}{j+1}',absorb='right',**compress_opts)
            for j in range(nf-1,0,-1):
                tn.compress_between(f'{tag}{j}',f'{tag}{j+1}',absorb='left',**compress_opts)
        except ValueError:
            tn = tn
        tn = scale(tn)
        if iprint>0:
            print(f'exponent={tn.exponent}')
            print(tn)
    return tn
def compress_row(tni,n,tag,tr,iprint=0,**compress_opts):
    nf = tni.num_tensors
    tn1 = compress_pol_from_top(tni,n,tag,tr,iprint=iprint,**compress_opts)
    tn2 = compress_pol_from_bottom(tni,n,tag,tr,iprint=iprint,**compress_opts)
    for j in range(1,nf+1):
        tn2[f'{tag}{j}'].reindex_({f'{tag}{j-1},{j}':f'{tag}{j-1},{j}_',
                                  f'{tag}{j},{j+1}':f'{tag}{j},{j+1}_'})
    tn1.add_tensor_network(tn2,check_collisions=False)
    for j in range(2,nf+1):
        out = tn1.contract_tags((f'{tag}{j-1}',f'{tag}{j}'),which='any',inplace=True)
    return out*10.**tn1.exponent
