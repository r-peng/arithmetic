import numpy as np
import quimb.tensor as qtn
import math
def triangular(N,V2,split_opts={},contract_opts={}):
    # E = sum_{i<j}V(xi,xj)
    # N: number of particle
    # Vij = exp(-V(xi,xj))
    d = V2.shape[0]
    CP = np.zeros((d,)*3)
    for i in range(d):
        CP[i,i,i] = 1.0
    V3 = np.einsum('ij,ipr->prj',V2,CP)
    V4 = np.einsum('prj,jqs->prqs',V3,CP)
    
    i = 0
    vj = np.einsum('prqs,p->rqs',V4,np.ones(d)/d)
    v0 = np.einsum('rqs,q->rs',vj,np.ones(d)/d)
    inds = 'i{},{},'.format(N-1,i),'i{},{},'.format(i,N-1)
    b = qtn.Tensor(data=v0,inds=inds)
    ps = []
    for j in range(N-2,i+1,-1):
        inds = 'i{},{},'.format(j,i),'i{},{},'.format(i,j+1),'i{},{},'.format(i,j)
        v = qtn.Tensor(data=vj,inds=inds)
        b = qtn.tensor_contract(b,v)
        r1 = 'i{},{},'.format(j+1,i) if j==N-2 else 's{},{},'.format(j+1,i)
        right_inds = r1,'i{},{},'.format(j,i)
        bond_ind = 's{},{},'.format(j,i)
        b,p = qtn.tensor_split(b,left_inds=None,right_inds=right_inds,
                               bond_ind=bond_ind,get='tensors',**split_opts)
        ps.append(p)
    vN = np.einsum('prj,p->rj',V3,np.ones(d)/d)
    inds = 'i{},{},'.format(i+1,i),'i{},{},'.format(i,i+2)
    v = qtn.Tensor(data=vN,inds=inds)
    bs = qtn.TensorNetwork([qtn.tensor_contract(b,v)])

    for i in range(1,N-2):
        inds = 'i{},{},'.format(N-1,i-1),'i{},{},'.format(N-1,i),\
               'i{},{},'.format(i,i-1),'i{},{},'.format(i,N-1)
        b = qtn.Tensor(data=V4,inds=inds)
        for j in range(N-2,i+1,-1):
            inds = 'i{},{},'.format(j,i-1),'i{},{},'.format(j,i),\
                   'i{},{},'.format(i,j+1),'i{},{},'.format(i,j)
            v = qtn.Tensor(data=V4,inds=inds)
            b = qtn.tensor_contract(b,v,ps.pop(0))
            r1 = 'i{},{},'.format(j+1,i) if j==N-2 else 's{},{},'.format(j+1,i)
            right_inds = r1,'i{},{},'.format(j,i)
            bond_ind = 's{},{},'.format(j,i)
            b,p = qtn.tensor_split(b,left_inds=None,right_inds=right_inds,
                                   bond_ind=bond_ind,get='tensors',**split_opts)
            ps.append(p)
        inds = 'i{},{},'.format(i+1,i-1),'i{},{},'.format(i+1,i),\
               'i{},{},'.format(i,i+2)
        v = qtn.Tensor(data=V3,inds=inds)
        bs.add_tensor(qtn.tensor_contract(b,v,ps.pop(0)))

    i = N-2
    inds = 'i{},{},'.format(N-1,i-1),'i{},{},'.format(i,i-1)
    bs.add_tensor(qtn.Tensor(data=V2,inds=inds))
    return bs.contract(**contract_opts)
def triangular_to_peps(N,V2,tr):
    d = V2.shape[0]

    L1,s,R1 = np.linalg.svd(V2)
    s = np.sqrt(s)
    L1 = np.einsum('is,s->is',L1,s)
    R1 = np.einsum('si,s->si',R1,s)

    CP3 = np.zeros((d,)*3)
    for i in range(d):
        CP3[i,i,i] = 1.0
    L2 = np.einsum('is,ijk->jks',L1,CP3)
    R2 = np.einsum('si,ijk->sjk',R1,CP3)
    
    arrays = []
    # b=0
    tmp = np.einsum('is,skl,k->il',L1,R2,tr)
    row = [tmp.reshape(tmp.shape+(1,))]
    tmp = np.einsum('ijs,skl,k->jil',L2,R2,tr)
    row += [tmp.reshape(tmp.shape+(1,)) for j in range(N-3)]
    tmp = np.einsum('ijs,skl,i,k->jl',L2,R2,tr,tr)
    row.append(tmp.reshape(tmp.shape+(1,)))
    arrays.append(row)
    CP4 = np.einsum('ij,kl->ijkl',np.eye(d),np.eye(d))
    for b in range(1,N-2):
        for i in range(N-1,b,-1):
            row = []
            for j in range(1,b):
                if j==1:
                    row.append(np.array([1]).reshape((1,)*4))
                else:
                    row.append(np.array([1]).reshape((1,)*5))
            # j=b
            if i==b+1:
                tmp = L1.T
                tmp = tmp.reshape(tmp.shape+(1,))
            else:
                tmp = L2.transpose(2,0,1)
            if b>1:
                tmp = tmp.reshape((1,)+tmp.shape)
            row.append(tmp.reshape(tmp.shape+(1,)))
            for j in range(b+1,i):
                row.append(CP4.reshape(CP4.shape+(1,)))
            # j=i
            tmp = R2.copy()
            if i<N-1:
                tmp = tmp.reshape(tmp.shape[:1]+(1,)+tmp.shape[1:])
            row.append(tmp.reshape(tmp.shape+(1,)))
            for j in range(i+1,N):
                if j==N-1:
                    row.append(np.eye(d).reshape((1,d,d,1)))
                else:
                    row.append(np.eye(d).reshape((1,1,d,d,1)))
            arrays.append(row)
    row = [np.array([1]).reshape((1,)*3)]
    for j in range(2,N-2):
        row.append(np.array([1]).reshape((1,)*4))
    tmp = L1.T
    row.append(tmp.reshape((1,)+tmp.shape+(1,)))
    row.append(R1.reshape(R1.shape+(1,)))
    arrays.append(row)

    peps = qtn.PEPS(arrays,shape='lrdup')
    return peps
