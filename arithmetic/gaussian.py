import numpy as np
import quimb.tensor as qtn
from .utils import (
    ADD,CP2,
    scale,compress1D,
)
def get_field(ys,tag,nf,iprint=0,**compress_opts):
    if iprint>0:
        print('getting field...')
    ng = len(ys)
    CP = np.zeros((nf,)*3)
    for i in range(nf):
        CP[i,i,i] = 1.

    tn = qtn.TensorNetwork([])   
    for i in range(1,nf+1):
        yi = np.ones((ng,2))
        yi[:,1] = ys.copy()
        ei = np.zeros((nf,2))
        ei[:,0] = np.ones(nf)
        ei[i-1,1] = 1.
        data = np.einsum('yi,pj,ijk->ypk',yi,ei,CP2)
        inds = f'{tag}{i}',f'{tag}{i-1},{i}',f'{tag}{i-1},{i}_'
        if i<nf:
            data = np.einsum('ypk,pRL,krl->yLlRr',data,CP,ADD)
            inds += f'{tag}{i},{i+1}',f'{tag}{i},{i+1}_'
        tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags=f'{tag}{i}'))
    tn[f'{tag}1'].reindex_({f'{tag}0,1':'p',f'{tag}0,1_':'k'})
    t0,t1 = tn[f'{tag}1'].split(('p','k'),bond_ind=f'{tag}0,1',absorb='both',**compress_opts)
    t0.modify(tags=f'{tag}0')
    tn.add_tensor(t0)
    tn[f'{tag}1'].modify(data=t1.data,inds=t1.inds)
    tn.fuse_multibonds_()
    tn = compress1D(tn,tag,final='right',iprint=iprint,**compress_opts)
    if iprint>0:
        print(tn)
    return tn
def get_exponent(tny,A,tag,iprint=0,**compress_opts):
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

    tn = tny.copy()
    data = np.ones(A.shape+(2,))
    data[...,1] = A
    T1 = tn[f'{tag}0'].reindex_({'p':'p0','k':'i1'})
    T2 = qtn.Tensor(data=data,inds=[f'p{i}' for i in range(n)]+['i2'])
    T = qtn.tensor_contract(T1,T2,qtn.Tensor(data=sCP2,inds=('i1','i2','k')))
    T1.modify(data=T.data,inds=T.inds)
    for i in range(1,n):
        if iprint>0:
            print(f'compressing row {i-1},{i}...')
        T1 = tn[f'{tag}0'].reindex_({'k':'i1'})
        T2 = tny[f'{tag}0'].reindex({'p':f'p{i}','k':'i2',f'{tag}0,1':f'{tag}0,1_'})
        T = qtn.tensor_contract(T1,T2,qtn.Tensor(data=sCP2,inds=('i1','i2','k')))
        T1.modify(data=T.data,inds=T.inds)
        for j in range(1,nf+1):
            T1 = tn[f'{tag}{j}'].reindex_({f'{tag}{j}':'a'})
            T2 = tny[f'{tag}{j}'].reindex({f'{tag}{j}':'b',
                                           f'{tag}{j-1},{j}':f'{tag}{j-1},{j}_',
                                           f'{tag}{j},{j+1}':f'{tag}{j},{j+1}_'})
            T = qtn.tensor_contract(T1,T2,qtn.Tensor(data=CP,inds=('a','b',f'{tag}{j}')))
            T1.modify(data=T.data,inds=T.inds)
        tn.fuse_multibonds_()
        tn = compress1D(tn,tag,final='right',iprint=iprint,**compress_opts)
        if iprint>0:
            print(tn)
    return tn
def add_exponent(tnA,tnB,tag,iprint=0,**compress_opts):
    if iprint>0:
        print('adding exponent...')
    nf = tnA.num_tensors-1
    ng = tnA[f'{tag}1'].shape[tnA[f'{tag}1'].inds.index(f'{tag}1')]
    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1.

    tn = tnA.copy()
    T1 = tn[f'{tag}0'].reindex_({'k':'i1'})
    T2 = tnB[f'{tag}0'].reindex({'k':'i2',f'{tag}0,1':f'{tag}0,1_'})
    T = qtn.tensor_contract(T1,T2,qtn.Tensor(data=ADD,inds=('i1','i2','k')))
    T1.modify(data=T.data,inds=T.inds)
    for i in range(1,nf+1):
        T1 = tn[f'{tag}{i}'].reindex_({f'{tag}{i}':'a'})
        T2 = tnB[f'{tag}{i}'].reindex({f'{tag}{i}':'b',
                                       f'{tag}{i-1},{i}':f'{tag}{i-1},{i}_',
                                       f'{tag}{i},{i+1}':f'{tag}{i},{i+1}_'})
        T = qtn.tensor_contract(T1,T2,qtn.Tensor(data=CP,inds=('a','b',f'{tag}{i}')))
        T1.modify(data=T.data,inds=T.inds)
    tn.fuse_multibonds_()
    tn = compress1D(tn,tag,final='right',iprint=iprint,**compress_opts)
    if iprint>0:
        print(tn)
    return tn
def trace_pol_compress_row(tni,tag,tr,coeff,iprint=0,**compress_opts):
    nf = tni.num_tensors-1
    n = len(coeff) - 1
    n1 = n // 2 # nrows from top
    n2 = n - n1
    coeff2,coeff1 = coeff[:n2],coeff[n2:]
    tn1 = compress_pol_from_top(tni,tag,coeff1,tr,iprint=iprint,**compress_opts)
    tn2 = compress_pol_from_bottom(tni,tag,coeff2,tr,iprint=iprint,**compress_opts)
    for j in range(nf+1):
        tn2[f'{tag}{j}'].reindex_({f'{tag}{j-1},{j}':f'{tag}{j-1},{j}_',
                                  f'{tag}{j},{j+1}':f'{tag}{j},{j+1}_'})
    tn1.add_tensor_network(tn2,check_collisions=False)
    for j in range(1,nf+1):
        out = tn1.contract_tags((f'{tag}{j-1}',f'{tag}{j}'),which='any',inplace=True)
    return out*10.**tn1.exponent
def compress_pol_from_top(tni,tag,coeff,tr,iprint=0,**compress_opts):
    nf = tni.num_tensors-1
    n = len(coeff)-1
    ng = tni[f'{tag}1'].shape[tni[f'{tag}1'].inds.index(f'{tag}1')]
    if iprint>0:
        print(f'{n} top rows')

    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1.

    tn = tni.copy()
    T1 = tn[f'{tag}0'].reindex_({'k':'_'})
    data = np.einsum('i,j,ijk->ik',np.array([1.,coeff[n]]),np.array([1.,coeff[n-1]]),ADD)
    T = qtn.tensor_contract(T1,qtn.Tensor(data=data,inds=('_','k')))
    T1.modify(data=T.data,inds=T.inds)
    for i in range(n-2,-1,-1):
        if iprint>0:
            print(f'compressing row {i+1},{i}...')
        T1 = tn[f'{tag}0'].reindex_({'k':'i1'})
        T2 = tni[f'{tag}0'].reindex({'k':'i2',f'{tag}0,1':f'{tag}0,1_'})
        data = np.einsum('ijk,klm,l->ijm',CP2,ADD,np.array([1.,coeff[i]]))
        T = qtn.tensor_contract(T1,T2,qtn.Tensor(data=data,inds=('i1','i2','k')))
        T1.modify(data=T.data,inds=T.inds)
        for j in range(1,nf+1):
            T1 = tn[f'{tag}{j}'].reindex_({f'{tag}{j}':'a'})
            T2 = tni[f'{tag}{j}'].reindex({f'{tag}{j}':'b',
                                            f'{tag}{j-1},{j}':f'{tag}{j-1},{j}_',
                                            f'{tag}{j},{j+1}':f'{tag}{j},{j+1}_'})
            T = qtn.tensor_contract(T1,T2,qtn.Tensor(data=CP,inds=('a','b',f'{tag}{j}')))
            T1.modify(data=T.data,inds=T.inds)
        tn.fuse_multibonds_()
        try:
            tn = compress1D(tn,tag,final='right',iprint=iprint,**compress_opts)
        except ValueError:
            tn = tn
        tn = scale(tn)
        if iprint>0:
            print(f'exponent={tn.exponent}')
            print(tn)
    return tn
def compress_pol_from_bottom(tni,tag,coeff,tr,iprint=0,**compress_opts):
    nf = tni.num_tensors-1
    n = len(coeff)
    ng = tni[f'{tag}1'].shape[tni[f'{tag}1'].inds.index(f'{tag}1')]
    if iprint>0:
        print(f'{n} bottom rows')

    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1.

    tn = tni.copy()
    T1 = tn[f'{tag}0'].reindex_({'k':'_'})
    data = np.einsum('j,k,ijk,ilm->lm',np.array([1.,coeff[0]]),np.array([0.,1.]),ADD,CP2)
    T = qtn.tensor_contract(T1,qtn.Tensor(data=data,inds=('_','k')))
    T1.modify(data=T.data,inds=T.inds)

    for j in range(1,nf+1):
        T1 = tn[f'{tag}{j}'].reindex_({f'{tag}{j}':'a'})
        T = qtn.tensor_contract(T1,qtn.Tensor(data=CP,inds=('a','b',f'{tag}{j}')),
                                   qtn.Tensor(data=tr[j],inds=('b',)))
        T1.modify(data=T.data,inds=T.inds)

    for i in range(1,n):
        if iprint>0:
            print(f'compressing row {i-1},{i}...')
        T1 = tn[f'{tag}0'].reindex_({'k':'_'})
        data = np.einsum('ijk,j->ik',ADD,np.array([1.,coeff[i]]))
        T = qtn.tensor_contract(T1,qtn.Tensor(data=data,inds=('i1','_')))
        T1.modify(data=T.data,inds=T.inds)

        T2 = tni[f'{tag}0'].reindex({'k':'i2',f'{tag}0,1':f'{tag}0,1_'})
        T = qtn.tensor_contract(T1,T2,qtn.Tensor(data=CP2,inds=('i1','i2','k')))
        T1.modify(data=T.data,inds=T.inds)
        for j in range(1,nf+1):
            T1 = tn[f'{tag}{j}'].reindex_({f'{tag}{j}':'a'})
            T2 = tni[f'{tag}{j}'].reindex({f'{tag}{j}':'b',
                                            f'{tag}{j-1},{j}':f'{tag}{j-1},{j}_',
                                            f'{tag}{j},{j+1}':f'{tag}{j},{j+1}_'})
            T = qtn.tensor_contract(T1,T2,qtn.Tensor(data=CP,inds=('a','b',f'{tag}{j}')))
            T1.modify(data=T.data,inds=T.inds)
        tn.fuse_multibonds_()
        try:
            tn = compress1D(tn,tag,final='right',iprint=iprint,**compress_opts)
        except ValueError:
            tn = tn
        tn = scale(tn)
        if iprint>0:
            print(f'exponent={tn.exponent}')
            print(tn)
    return tn
def trace_pol_compress_col(tni,tag,new_tag,tr,coeff,iprint=0,**compress_opts):
    tn1 = compress_pol_from_right(tni,tag,new_tag,tr,coeff,iprint=iprint,**compress_opts)
    tn2 = compress_pol_from_left(tni,tag,new_tag,tr,coeff,iprint=iprint,**compress_opts)
    n = len(coeff)-1
    for i in range(n):
        tn2[f'{new_tag}{i}'].reindex_({f'{new_tag}{i-1},{i}':f'{new_tag}{i-1},{i}_',
                                       f'{new_tag}{i},{i+1}':f'{new_tag}{i},{i+1}_'})
    tn1.add_tensor_network(tn2,check_collisions=False)
    for i in range(1,n):
        out = tn1.contract_tags((f'{new_tag}{i-1}',f'{new_tag}{i}'),which='any',inplace=True)
    return out*10.**tn1.exponent
def compress_pol_from_right(tni,tag,new_tag,tr,coeff,iprint=0,**compress_opts):
    nf = tni.num_tensors-1
    n = len(coeff)-1
    stop = nf // 2
    ng = tni[f'{tag}1'].shape[tni[f'{tag}1'].inds.index(f'{tag}1')]
    if iprint>0:
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
                T = qtn.tensor_contract(T,qtn.Tensor(data=CP,inds=(f'{tag}{j}',f'{new_tag}{i-1},{i}',f'{new_tag}{i},{i+1}')))
            if i==n-1:
                T = qtn.tensor_contract(T,qtn.Tensor(data=tr[j],inds=(f'{new_tag}{i},{i+1}',)))
            T.modify(tags=f'{new_tag}{i}')
            col.add_tensor(T)
        return col

    tn = get_col(nf)
    for j in range(nf-1,stop,-1):
        if iprint>0:
            print(f'compressing col {j+1},{j}...')
        col = get_col(j)
        for i in range(n):
            T1 = tn[f'{new_tag}{i}'].reindex_({f'{new_tag}{i}':f'{i}_'})
            T2 = col[f'{new_tag}{i}'].reindex_({f'{new_tag}{i-1},{i}':f'{new_tag}{i-1},{i}_',f'{new_tag}{i},{i+1}':f'{new_tag}{i},{i+1}_'})
        tn.add_tensor_network(col,check_collisions=False)
        for i in range(n):
            tn.contract_tags(f'{new_tag}{i}',which='any',inplace=True)
        tn.fuse_multibonds_()
        try:
            tn = compress1D(tn,new_tag,iprint=iprint,**compress_opts)
        except ValueError:
            tn = tn
        tn = scale(tn)
        if iprint>0:
            print(f'exponent={tn.exponent}')
            print(tn)
    return tn
def compress_pol_from_left(tni,tag,new_tag,tr,coeff,iprint=0,**compress_opts):
    nf = tni.num_tensors-1
    n = len(coeff)-1
    stop = nf // 2
    ng = tni[f'{tag}1'].shape[tni[f'{tag}1'].inds.index(f'{tag}1')]
    if iprint>0:
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
                T = qtn.tensor_contract(T,qtn.Tensor(data=CP,inds=(f'{tag}{j}',f'{new_tag}{i-1},{i}',f'{new_tag}{i},{i+1}')))
            if i==n-1:
                T = qtn.tensor_contract(T,qtn.Tensor(data=tr[j],inds=(f'{new_tag}{i},{i+1}',)))
            T.modify(tags=f'{new_tag}{i}')
            col.add_tensor(T)
        return col
    tn = qtn.TensorNetwork([])
    for i in range(n):
        T = tni[f'{tag}0'].reindex({f'{tag}0,1':f'{new_tag}{i}'})
        data = np.einsum('ijk,j,ilm->klm',ADD,np.array([1.,coeff[i]]),CP2)
        T = qtn.tensor_contract(T,qtn.Tensor(data=data,inds=(f'{new_tag}{i-1},{i}','k',f'{new_tag}{i},{i+1}')))
        if i==0:
            T = qtn.tensor_contract(T,qtn.Tensor(data=np.array([0.,1.]),inds=(f'{new_tag}{i-1},{i}',))) 
        if i==n-1:
            T = qtn.tensor_contract(T,qtn.Tensor(data=np.array([1.,coeff[n]]),inds=(f'{new_tag}{i},{i+1}',)))
        T.modify(tags=f'{new_tag}{i}')
        tn.add_tensor(T)
    for j in range(1,stop+1):
        if iprint>0:
            print(f'compressing col {j-1},{j}...')
        col = get_col(j)
        for i in range(n):
            T1 = tn[f'{new_tag}{i}'].reindex_({f'{new_tag}{i}':f'{i}_'})
            T2 = col[f'{new_tag}{i}'].reindex_({f'{new_tag}{i-1},{i}':f'{new_tag}{i-1},{i}_',f'{new_tag}{i},{i+1}':f'{new_tag}{i},{i+1}_'})
        tn.add_tensor_network(col,check_collisions=False)
        for i in range(n):
            tn.contract_tags(f'{new_tag}{i}',which='any',inplace=True)
        tn.fuse_multibonds_()
        try:
            tn = compress1D(tn,new_tag,iprint=iprint,**compress_opts)
        except ValueError:
            tn = tn
        tn = scale(tn)
        if iprint>0:
            print(f'exponent={tn.exponent}')
            print(tn)
    return tn
       
