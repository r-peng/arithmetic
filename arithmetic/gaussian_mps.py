import numpy as np
import quimb.tensor as qtn
import math
ADD = np.zeros((2,)*3)
ADD[0,0,0] = ADD[1,0,1] = ADD[0,1,1] = 1.0
CP2 = np.zeros((2,)*3)
CP2[0,0,0] = CP2[1,1,1] = 1.0
def get_energy(A,xs,contract=False,**compress_opts):
    N,d = xs.shape
    arrays = []
    # 1-variable terms
    for i in range(N):
        fac = A[i,i]
        data = np.ones((2,d))
        data[1,:] = np.square(xs[i,:])*fac
        if i>0:
            data = np.einsum('ip,ijk->jkp',data,ADD)
        arrays.append(data)
    arrays.append(np.eye(2))
    mps = qtn.MatrixProductState(arrays,shape='lrp')
    # 2-variable terms
    dummy = np.einsum('ij,pq->ijpq',np.eye(2),np.eye(d))
    CP = np.zeros((d,)*3)
    for i in range(d):
        CP[i,i,i] = 1.0
    for i in range(N):
        for j in range(i+1,N):
            arrays = []
            fac = A[i,j]+A[j,i]
            for k in range(i):
                newshape = (1,d,d) if k==0 else (1,1,d,d)
                arrays.append(np.reshape(np.eye(d),newshape))
            # xi
            data = np.ones((2,d))
            data[1,:] = xs[i,:]*fac
            data = np.einsum('ip,pqr->iqr',data,CP)
            if i>0:
                data = np.reshape(data,(1,2,d,d))
            arrays.append(data)
            for k in range(i+1,j):
                arrays.append(dummy)
            # xj
            data = np.ones((2,d))
            data[1,:] = xs[j,:].copy()
            arrays.append(np.einsum('ip,ijk,pqr->jkqr',data,CP2,CP))
            for k in range(j+1,N):
                arrays.append(dummy)
            arrays.append(ADD.transpose(0,2,1))
            mpo = qtn.MatrixProductOperator(arrays,shape='lrud')
            mps = mpo.apply(mps,compress=True,**compress_opts)
    if contract:
        tr = np.ones(d)/d
        for i in range(N+1):
            data = np.array([0.0,1.0]) if i==N else tr
            mps.add_tensor(qtn.Tensor(data=data,inds=[mps.site_ind(i)]))
        return qtn.tensor_contract(*mps.tensors)
    return mps
def get_pol(A,xs,coeff,**compress_opts):
    tmp = np.einsum('ijk,klm->ijlm',CP2,ADD)
    N,d = xs.shape
    E = get_energy(A,xs,**compress_opts)
    # a_n*x+a_{n-1}
    mps = E.copy()
    T = mps[mps.site_tag(N)]
    inds = qtn.rand_uuid(),mps.site_ind(N)
    T.reindex_({inds[1]:inds[0]})
    data = np.einsum('ijlm,j,l->im',tmp,np.array([1.0,coeff[-1]]),
                                        np.array([1.0,coeff[-2]]))
    t = qtn.tensor_contract(T,qtn.Tensor(data=data,inds=inds))
    T.modify(data=t.data,inds=t.inds)
    mps.view_as_(qtn.MatrixProductState)
    print(mps)

    CP = np.zeros((d,)*3)
    for i in range(d):
        CP[i,i,i] = 1.0
    for ai in coeff[:-2][::-1]:
        mpo = E.copy()
        lid = mpo._site_ind_id
        uid = 'b{}'
        for i in range(N+1):
            T = mpo[mpo.site_tag(i)]
            inds = qtn.rand_uuid(),lid.format(i),uid.format(i)
            T.reindex_({inds[1]:inds[0]})
            data = np.einsum('ijlm,l->ijm',tmp,np.array([1.0,ai])) if i==N else CP
            t = qtn.tensor_contract(T,qtn.Tensor(data=data,inds=inds))
            T.modify(data=t.data,inds=t.inds)
        mpo.view_as_(qtn.MatrixProductOperator,upper_ind_id=uid,lower_ind_id=lid)
        mps = mpo.apply(mps,compress=True,**compress_opts)
        print(mps)

    tr = np.ones(d)/d
    for i in range(N+1):
        data = np.array([0.0,1.0]) if i==N else tr
        mps.add_tensor(qtn.Tensor(data=data,inds=[mps.site_ind(i)]))
    return qtn.tensor_contract(*mps.tensors)
def get_coeffs(a,M):
    # M degree taylor approximation of exp(-x) from [0,2*a] centered at a
    coeff = []
    fac = [math.factorial(k) for k in range(M+1)]
    for k in range(M+1):
        out = 0.0
        for l in range(M-k+1):
            out += (-a)**l/fac[l]
        coeff.append(np.exp(a)*out/fac[k])
    return coeff
