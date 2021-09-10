import numpy as np
import quimb.tensor as qtn
import math
def get_diag_row_top(N,tensors,A,an,tr):
    X,XSQ,CPd,L,R,ADD,CP2,CP2d = tensors
    d = X.shape[0]
    row = []
    tmp = XSQ.copy()
    tmp[:,1] *= 0.5*A[0,0]
    tmp = np.einsum('pqr,qi,p->ir',CPd,tmp,tr)
    row.append(tmp.reshape(tmp.shape+(1,)))
    for i in range(1,N):
        tmp = XSQ.copy()
        tmp[:,1] *= 0.5*A[i,i]
        tmp = np.einsum('pqr,qi,p->ir',CPd,tmp,tr)
        tmp = np.einsum('ir,ijk->jkr',tmp,ADD)
        row.append(tmp.reshape(tmp.shape+(1,)))
    row.append(np.eye(2).reshape(2,1,2,1))
    row.append(np.array([1.0,an]).reshape(1,2,1))
    return row
def get_diag_row(N,tensors,A):
    X,XSQ,CPd,L,R,ADD,CP2,CP2d = tensors
    d = X.shape[0]
    row = []
    tmp = XSQ.copy()
    tmp[:,1] *= 0.5*A[0,0]
    tmp = np.einsum('pqr,qi->ipr',CPd,tmp)
    row.append(tmp.reshape(tmp.shape+(1,)))
    for i in range(1,N):
        tmp = XSQ.copy()
        tmp[:,1] *= 0.5*A[i,i]
        tmp = np.einsum('pqr,qi->ipr',CPd,tmp)
        tmp = np.einsum('ipr,ijk->jkpr',tmp,ADD)
        row.append(tmp.reshape(tmp.shape+(1,)))
    row.append(np.eye(2).reshape(2,1,1,2,1))
    row.append(np.eye(2).reshape(1,2,2,1))
    return row
def get_off_diag_row(N,tensors,A,a,b,i):
    X,XSQ,CPd,L,R,ADD,CP2,CP2d = tensors
    d =X.shape[0]
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
    # 2nd last col
    tmp = np.einsum('ijk,klm,j->ilm',CP2,ADD,np.array([1.0,A[b,i]]))
    if b==N-2 and i==b+1:
        tmp = tmp.transpose(0,2,1)
        row.append(tmp.reshape(2,2,2,1,1))
        tmp = np.einsum('ijk,klm,l->ijm',CP2,ADD,np.array([1.0,a]))
        row.append(tmp.reshape(tmp.shape+(1,)))
    else:
        row.append(tmp.reshape(2,1,2,2,1))
        row.append(np.eye(2).reshape(1,2,2,1))
    return row
def get_off_diag_row_bottom(N,tensors,A,b,i):
    X,XSQ,CPd,L,R,ADD,CP2,CP2d = tensors
    d = X.shape[0]
    row = []
    for j in range(b):
        if j==0:
            row.append(np.array([1.0]).reshape(1,1,1,1))
        else:
            row.append(np.array([1.0]).reshape(1,1,1,1,1))
    if i==b+1:
        tmp = X.T
        tmp = tmp.reshape(tmp.shape+(1,))
    else: 
        tmp = L.copy()
    if b>0:
        tmp = tmp.reshape((1,)+tmp.shape)
    row.append(tmp.reshape(tmp.shape+(1,)))
    for j in range(b+1,i):
        row.append(CP2d.reshape(CP2d.shape+(1,)))
    row.append(R.reshape(R.shape+(1,)))
    for j in range(i+1,N):
        row.append(CP2d.reshape(CP2d.shape+(1,)))
    # 2nd last col
    tmp = np.einsum('ijk,klm,j->ilm',CP2,ADD,np.array([1.0,A[b,i]]))
    row.append(tmp.reshape(2,1,2,2,1))
    row.append(np.eye(2).reshape(1,2,2,1))
    return row
def get_last_row(N,tensors,A,a0):
    X,XSQ,CPd,L,R,ADD,CP2,CP2d = tensors
    d = X.shape[0]
    row = []
    b = N-2
    for j in range(b):
        if j==0:
            row.append(np.array([1.0]).reshape(1,1,1))
        else:
            row.append(np.array([1.0]).reshape(1,1,1,1))
    tmp = X.T.copy()
    row.append(tmp.reshape((1,)+tmp.shape+(1,)))
    tmp = np.einsum('qi,ijk->jkq',X,CP2)
    row.append(tmp.reshape(tmp.shape+(1,)))
    # 2nd last col
    tmp = np.einsum('ijk,klm,j->iml',CP2,ADD,np.array([1.0,A[b,b+1]]))
    row.append(tmp.reshape(tmp.shape+(1,)))
    tmp = np.einsum('ijk,kl,l->ij',CP2,ADD[:,:,1],np.array([1.0,a0]))
    row.append(tmp.reshape(tmp.shape+(1,)))
    return row
def energy_top(N,tensors,A,a,an,tr):
    arrays = []
    arrays.append(get_diag_row_top(N,tensors,A,an,tr))

    for b in range(N-1):
        for i in range(N-1,b,-1):
            arrays.append(get_off_diag_row(N,tensors,A,a,b,i))
#    print('top',len(arrays))
    return arrays
def energy_intermediate(N,tensors,A,a):
    arrays = []
    arrays.append(get_diag_row(N,tensors,A))

    for b in range(N-1):
        for i in range(N-1,b,-1):
            arrays.append(get_off_diag_row(N,tensors,A,a,b,i))
#    print('intermediate',len(arrays))
    return arrays
def energy_bottom(N,tensors,A,a):
    arrays = []
    arrays.append(get_diag_row(N,tensors,A))

    for b in range(N-2):
        for i in range(N-1,b,-1):
            arrays.append(get_off_diag_row_bottom(N,tensors,A,b,i))
    arrays.append(get_last_row(N,tensors,A,a))
#    print('bottom',len(arrays))
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
def energy(N,xs,A,tr):
    tensors = get_tensors(N,xs)
    arrays = []
    arrays.append(get_diag_row_top(N,tensors,A,1.0,tr))
    for b in range(N-2):
        for i in range(N-1,b,-1):
            arrays.append(get_off_diag_row_bottom(N,tensors,A,b,i))
    arrays.append(get_last_row(N,tensors,A,0.0))
    return qtn.PEPS(arrays,shape='lrdup')
def pol(N,xs,A,coeff,tr):
    tensors = get_tensors(N,xs)
    arrays = energy_top(N,tensors,A,coeff[-2],coeff[-1],tr)
    for i in range(len(coeff)-3,0,-1):
        arrays += energy_intermediate(N,tensors,A,coeff[i])
    arrays += energy_bottom(N,tensors,A,coeff[0])
    for i in range(len(arrays)):
        row = arrays[i]
        for j in range(len(row)):
            row[j] = np.ascontiguousarray(row[j])
        arrays[i] = row
    return qtn.PEPS(arrays,shape='lrdup')
def get_coeffs(a,M):
    # M degree taylor approximation of exp(-x) from [0,2*a] centered at a
    coeff = []
    fac = [math.factorial(k) for k in range(M+1)]
    for k in range(M+1):
        out = 0.0
        for n in range(k,M+1):
            out += a**(n-k)/(fac[k]*fac[n-k])
        coeff.append((-1)**k*np.exp(-a)*out)
    return coeff
     
