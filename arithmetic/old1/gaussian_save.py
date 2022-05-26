import numpy as np
import quimb.tensor as qtn
import math
def diag(N,tensors,A):
    X,XSQ,CPd,L,R,ADD,CP2,CP2d = tensors

    row = []
    for i in range(N):
        tmp = np.einsum('ri,i,pqr->ipq',XSQ,np.array([1.0,0.5*A[0,0]]),CPd)
        if i>0:
            tmp = np.einsum('i...,ijk->jk...',tmp,ADD)
        row.append(tmp.reshape(tmp.shape+(1,)))
    row.append(np.eye(2).reshape(2,1,1,2,1))
    return row
def off_diag(N,tensors,A,b,i):
    X,XSQ,CPd,L,R,ADD,CP2,CP2d = tensors
    d = X.shape[0]

    row = []
    for j in range(b):
        if j==0:
            row.append(np.eye(d).reshape(1,d,d,1))
        else:
            row.append(np.eye(d).reshape(1,1,d,d,1))
    tmp = L.copy()
    if b>0:
        tmp = tmp.reshape((1,)+tmp.shape)
    row.append(tmp.reshape(tmp.shape+(1,)))
    for j in range(b+1,i):
        row.append(CP2d.reshape(CP2d.shape+(1,)))
    row.append(R.reshape(R.shape+(1,)))
    for j in range(i+1,N):
        row.append(CP2d.reshape(CP2d.shape+(1,)))
    # last col
    tmp = np.einsum('ijk,klm,j->ilm',CP2,ADD,np.array([1.0,A[b,i]]))
    if b==N-2 and i==b+1:
        tmp = tmp.transpose(0,2,1)
        row.append(tmp.reshape(2,2,2,1,1))
    else:
        row.append(tmp.reshape(2,1,2,2,1))
    return row
def off_diag_bottom(N,tensors,A,b,i):
    X,XSQ,CPd,L,R,ADD,CP2,CP2d = tensors
    d = X.shape[0]
    row = []
    for j in range(b):
        if j==0 and b==N-2:
            row.append(np.array([1.0]).reshape((1,1,1)))
        elif j>0 and b<N-2:
            row.append(np.array([1.0]).reshape((1,1,1,1,1)))
        else:
            row.append(np.array([1.0]).reshape((1,1,1,1)))
    # j==b
    if i==b+1:
        tmp = X.T
        if b<N-2:
            tmp = tmp.reshape(tmp.shape+(1,))
    else: 
        tmp = L.copy()
    if b>0:
        tmp = tmp.reshape((1,)+tmp.shape)
    row.append(tmp.reshape(tmp.shape+(1,)))
    for j in range(b+1,i):
        row.append(CP2d.reshape(CP2d.shape+(1,)))
    # j==i
    tmp = np.einsum('qi,ijk->jkq',X,CP2) if (b==N-2 and i==N-1) else R.copy()
    row.append(tmp.reshape(tmp.shape+(1,)))
    for j in range(i+1,N):
        row.append(CP2d.reshape(CP2d.shape+(1,)))
    # last
    tmp = np.einsum('ijk,klm,j->ilm',CP2,ADD,np.array([1.0,A[b,i]]))
    if b==N-2 and i==N-1:
        tmp = tmp.transpose(0,2,1)
        row.append(tmp.reshape(2,2,2,1))
    else:
        tmp = tmp.reshape(2,1,2,2)
        row.append(tmp.reshape(2,1,2,2,1))
    return row
def energy(N,tensors,A,bottom=False):
    arrays = []
    _off_diag = off_diag_bottom if bottom else off_diag
    arrays.append(diag(N,tensors,A))
    for b in range(N-1):
        for i in range(N-1,b,-1):
            arrays.append(_off_diag(N,tensors,A,b,i))
    return arrays
def horner(N,tensors,A,ai,an=None,bottom=False):
    X,XSQ,CPd,L,R,ADD,CP2,CP2d = tensors
    arrays = energy(N,tensors,A,bottom=bottom)
    nrows = len(arrays)
    for i in range(nrows-1):
        row = arrays[i]
        if an is None:
            row.append(np.eye(2).reshape(1,1,2,2,1))
        else:
            row.append(np.array([1.0]).reshape(1,1,1,1,1))
        arrays[i] = row
    row = arrays[-1]
    tmp = np.einsum('ijk,klm,l->ijm',CP2,ADD,np.array([1.0,ai]))
    if an is None:
        if bottom:
            tmp = tmp.transpose(0,2,1)
            row.append(tmp.reshape(2,2,2,1))
        else:
            row.append(tmp.reshape(2,1,2,2,1))
    else:
        tmp = np.einsum('ijm,j->im',tmp,np.array([1.0,an]))
        row.append(tmp.reshape(2,1,1,2,1))
    arrays[-1] = row
    return arrays
def get_tensors(N,xs):
    d = len(xs)
    X = np.zeros((d,2))
    X[:,0] = np.ones(d)
    X[:,1] = xs.copy()
    
    XSQ = np.zeros((d,2))
    XSQ[:,0] = np.ones(d)
    XSQ[:,1] = np.square(xs)

    ADD = np.zeros((2,)*3)
    ADD[0,0,0] = ADD[1,0,1] = ADD[0,1,1] = 1.0

    CPd = np.zeros((d,)*3)
    for i in range(d):
        CPd[i,i,i] = 1.0

    CP2 = np.zeros((2,)*3)
    CP2[0,0,0] = CP2[1,1,1] = 1.0

    L = np.einsum('pqr,qi->ipr',CPd,X)
    R = np.einsum('ipr,ijk->jkpr',L,CP2)
    CP2d = np.einsum('ij,pq->ijpq',np.eye(2),np.eye(d))
    return X,XSQ,CPd,L,R,ADD,CP2,CP2d
def poly(N,xs,A,coeff,bottom=True,tr=None,cap_right=True):
    # coeff = a0,...an
    tensors = get_tensors(N,xs)
    arrays = horner(N,tensors,A,ai=coeff[-2],an=coeff[-1])
    for i in range(len(coeff)-3,0,-1):
        arrays += horner(N,tensors,A,ai=coeff[i])
    arrays += horner(N,tensors,A,ai=coeff[0],bottom=bottom)
    if tr is not None:
        row = arrays[0]
        for i in range(len(row)):
            v = tr if i<N else np.array([1.0])
            row[i] = np.einsum('...pqk,p->...qk',row[i],v)
        arrays[0] = row
    if cap_right:
        for i in range(len(arrays)-1):
            row = arrays[i]
            row[-1] = np.einsum('ij...,j->i...',row[-1],np.array([1.0]))
            arrays[i] = row
        row = arrays[-1]
        if bottom:
            row[-1] = np.einsum('ij...,j->i...',row[-1],np.array([0.0,1.0]))
        else:
            row[-1] = np.einsum('ij...,j->i...',row[-1],np.array([1.0]))
        arrays[-1] = row
    return qtn.PEPS(arrays,shape='lrdup')
