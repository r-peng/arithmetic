import numpy as np
import quimb.tensor as qtn
CP2 = np.zeros((2,)*3)
CP2[0,0,0] = CP2[1,1,1] = 1.0
ADD = np.zeros((2,)*3)
ADD[0,0,0] = ADD[0,1,1] = ADD[1,0,1] = 1.0
def get_CP(d):
    out = np.zeros((d,)*3)
    for i in range(d):
        out[i,i,i] = 1.0
    return out
def get_weighted_sum1(X,ws,b):
    # weighted sum
    arrays = []
    for i in range(len(ws)):
        tmp = np.einsum('ipq,i->ipq',X,np.array([1.0,ws[i]]))
        if i>0:
            tmp = np.einsum('ipq,ijk->jkpq',tmp,ADD)
        arrays.append(tmp)
    tmp = np.einsum('ijk,i->jk',ADD,np.array([1.0,b]))
    arrays.append(tmp)
    return arrays
def get_weighted_sum(hs,ws,b,**compress_opts):
    arrays = [t.copy() for t in hs[0]]
    tmp = np.einsum('ijk,j->ik',CP2,np.array([1.0,ws[0]])).reshape(2,1,2)
    arrays[-1] = np.einsum('...i,ijk->...jk',arrays[-1],tmp)
    arrays = [np.ascontiguousarray(array) for array in arrays]
    mpo = qtn.MatrixProductOperator(arrays,shape='lrud')
    for i in range(1,len(ws)):
        arrays = [t.copy() for t in hs[i]]
        tmp = np.einsum('ijk,klm,j->ilm',CP2,ADD,np.array([1.0,ws[0]]))
        arrays[-1] = np.einsum('...i,ijk->...jk',arrays[-1],tmp)
        arrays = [np.ascontiguousarray(array) for array in arrays]
        other = qtn.MatrixProductOperator(arrays,shape='lrud')
        mpo = mpo._apply_mpo(other,compress=True,**compress_opts)
    arrays = list(mpo.tensor_map.values())
    arrays = [array.data for array in arrays]
    tmp = np.einsum('ijk,j->ik',ADD,np.array([1.0,b]))
    arrays[-1] = np.einsum('li,ik->lk',arrays[-1][:,0,:],tmp)
    return arrays
def get_pol(h,coeff,**compress_opts):
    for i in range(len(coeff)-2,-1,-1):
        arrays = [t.copy() for t in h]
        tmp = np.einsum('ijk,klm,l->ijm',CP2,ADD,np.array([1.0,coeff[i]])) 
        arrays[-1] = np.einsum('...i,ijk->...jk',arrays[-1],tmp)
        arrays = [np.ascontiguousarray(array) for array in arrays]
        uind,lind = 'u{},'.format(i),'l{},'.format(i)
        if i==len(coeff)-2:
            mpo = qtn.MatrixProductOperator(arrays,shape='lrud',
                  upper_ind_id=uind+'{}',lower_ind_id=lind+'{}')
            print(mpo)
        else:
            other = qtn.MatrixProductOperator(arrays,shape='lrud',
                  upper_ind_id=uind+'{}',lower_ind_id=lind+'{}')
            print(other)
            mpo = mpo._apply_mpo(other,compress=,**compress_opts)
            print(mpo)
    arrays = list(mpo.tensor_map.values())
    arrays = [array.data for array in arrays]
    arrays[-1] = np.einsum('lud,u->ld',arrays[-1],np.array([1.0,coeff[-1]]))
    return arrays
def get_layer1(xs,W,B,coeff,**compress_opts):
    # hi = Wijxj+Bi
    d = len(xs)
    X = np.zeros((d,2))
    X[:,0] = np.ones(d)
    X[:,1] = xs.copy()
    X = np.einsum('ri,pqr->ipq',X,get_CP(d))
    ls = []
    for i in range(len(B)):
        h = get_weighted_sum1(X,W[i,:],B[i])
        ls.append(get_pol(h,coeff,**compress_opts))
    return ls
def get_layer(hs,W,B,coeff,**compress_opts):
    ls = []
    for i in range(len(B)):
        h = get_weighted_sum(hs,W[i,:],B[i],**compress_opts)
        ls.append(get_pol(h,coeff,**compress_opts))
    return ls
def integrate(h,tr):
    out = np.einsum('rud,u,d->r',h[0],tr,np.ones_like(tr))
    for i in range(1,len(h)-1):
        tmp = np.einsum('lrud,u,d->lr',h[i],tr,np.ones_like(tr))
        out = np.einsum('l,lr->r',out,tmp)
    return np.dot(out,h[-1][:,1])
    
