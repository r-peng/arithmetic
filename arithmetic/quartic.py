import numpy as np
import quimb.tensor as qtn
# done 1. get_off_diagonal/cap_left: dummy before first x
# done 2. get_diagonal_row
# 3. only 4-body term
CP2 = np.zeros((2,)*3)
CP2[0,0,0] = CP2[1,1,1] = 1.0 
ADD = np.zeros((2,)*3)
ADD[0,0,0] = ADD[0,1,1]= ADD[1,0,1] = 1.0
def get_diag(Qs):
    row = []
    for i in range(len(Qs)):
        tmp = Qs[i].copy()
        if i==0:
            tmp = tmp[0,...].reshape((1,)+tmp.shape[1:])
        row.append(tmp)
    row.append(np.eye(2).reshape(2,1,1,2))
    row.append(np.eye(2).reshape(1,2,2))
    return row
def get_off_diag(N,xmap,cexp,cpol=None):
    inds = list(xmap.keys())
    inds.sort()
    d = xmap[inds[0]].shape[-1]
    row = []
    # dummies before the first x
    for j in range(inds[0]):
        row.append(np.eye(d).reshape(1,1,d,d))
    tmp = np.einsum('i...,i->...',xmap[inds[0]],np.ones(2))
    row.append(tmp.reshape((1,)+tmp.shape))
    # dummies between 2 consecutive x's
    for i in range(1,len(inds)):
        start,stop = inds[i-1]+1,inds[i]
        for j in range(start,stop):
            row.append(np.einsum('ij,pq->ijpq',np.eye(2),np.eye(d)))
        row.append(xmap[stop])
    for j in range(inds[-1]+1,N):
        row.append(np.einsum('ij,pq->ijpq',np.eye(2),np.eye(d)))
    coeff = np.array([1.0,cexp])
    if cpol is None:
        row.append(np.einsum('ijk,klm,j->ilm',CP2,ADD,coeff).reshape(2,1,2,2))
        row.append(np.eye(2).reshape(1,2,2))
    else:
        row.append(np.einsum('ijk,klm,j->iml',CP2,ADD,coeff).reshape(2,2,2,1))
        coeff = np.array([1.0,cpol])
        row.append(np.einsum('ijk,klm,l->ijm',CP2,ADD,coeff))
    return row
def cap_top(arrays,tr,an):
    row = arrays[0]
    N = len(row)-2
    for i in range(N):
        row[i] = np.einsum('...pq,p->...q',row[i],tr)
#    row[N] = np.einsum('...pq,p->...q',row[N],np.array([1.0]))
    row[N] = row[N][...,0,:]
    row[N+1] = np.einsum('...pq,p->...q',row[N+1],np.array([1.0,an]))
    arrays[0] = row
    return arrays
def cap_bottom(arrays):
    row = arrays[-1]
    N = len(row)-2
    d = row[0].shape[-1]
    for i in range(N):
        row[i] = np.einsum('...q,q->...',row[i],np.ones(d))
#    row[N] = np.einsum('...q,q->...',row[N],np.array([1.0]))
    row[N] = row[N][...,0]
#    row[N+1] = np.einsum('...q,q->...',row[N+1],np.array([0.0,1.0]))
    row[N+1] = row[N+1][...,1]
    arrays[-1] = row
    return arrays
def cap_left(arrays):
    nrows = len(arrays)
    for i in range(nrows):
        d = arrays[i][0].shape[0]
        arrays[i][0] = np.einsum('i...,i->...',arrays[i][0],np.ones(d))
    return arrays
def to_peps(arrays):
    nrows,ncols = len(arrays),len(arrays[0])
    for i in range(nrows):
        for j in range(ncols):
            array = arrays[i][j].reshape(arrays[i][j].shape+(1,))
            arrays[i][j] = np.ascontiguousarray(array)
    return qtn.PEPS(arrays,shape='lrdup')
def get_xmap(inds,Xs):
    xmap = {}
    inds.sort()
    old = inds.pop(0)
    count = 1
    while len(inds)>0:
        new = inds.pop(0)
        if new==old:
            count += 1
        else:
            xmap.update({old:Xs[count-1]})
            old = new
            count = 1
    xmap.update({old:Xs[count-1]})
    return xmap
def get_block(N,Qs,Xs,ai,A,B=None):
    arrays = [get_diag(Qs)]
    for i in range(N):
        for j in range(i):
            xmap = get_xmap([i,j],Xs)
            cpol = ai if i==N-1 and j==i-1 and B is None else None 
            arrays.append(get_off_diag(N,xmap,A[i,j],cpol=cpol))
    if B is not None:
        for i in range(N):
            for j in range(i):
                for k in range(j):
                    for l in range(k):
                        xmap = get_xmap([i,j,k,l],Xs)
                        cpol = ai if i==N-1 and j==i-1 and k==j-1 and l==k-1 else None 
                        arrays.append(get_off_diag(N,xmap,B[i,j,k,l],cpol=cpol))
#    row = arrays[0]
#    row[-2] = row[-2][:,:,:1,:]
    return arrays
def get_diag_tensors(xs,A):
    d = len(xs)
    CPd = np.zeros((d,)*3)
    for i in range(d):
        CPd[i,i,i] = 1.0
    X = np.zeros((d,2))
    X[:,1] = np.square(xs)
    X[:,0] = np.ones(d)
    return [np.einsum('pqr,ri,i,ijk->jkpq',CPd,X,np.array([1.0,a]),ADD) for a in A]
def get_off_diag_tensors(xs):
    # common tensors for off diagonal row
    d = len(xs)
    CPd = np.zeros((d,)*3)
    for i in range(d):
        CPd[i,i,i] = 1.0
    X1 = np.zeros((d,2))
    X1[:,1] = xs.copy()
    X1[:,0] = np.ones(d)
    X2 = np.multiply(X1,X1)
    X3 = np.multiply(X2,X1)
    X4 = np.multiply(X3,X1)
    return [np.einsum('pqr,ri,ijk->jkpq',CPd,X,CP2) for X in [X1,X2,X3,X4]]
def energy(N,xs,tr,A,B=None):
    Qs = get_diag_tensors(xs,list(0.5*np.diag(A)))
    Xs = get_off_diag_tensors(xs)
    arrays = get_block(N,Qs,Xs,0.0,A,B)
    arrays = cap_top(arrays,tr,1.0)
    arrays = cap_bottom(arrays)
    arrays = cap_left(arrays)
    return to_peps(arrays)
def pol(N,xs,tr,coeff,A,B=None):
    Qs = get_diag_tensors(xs,list(0.5*np.diag(A)))
    Xs = get_off_diag_tensors(xs)
    arrays = []
    for i in range(len(coeff)-2,-1,-1):
        arrays += get_block(N,Qs,Xs,coeff[i],A,B)
    arrays = cap_top(arrays,tr,coeff[-1])
    arrays = cap_bottom(arrays)
    arrays = cap_left(arrays)
    return to_peps(arrays)
def get_coeffs(a,M):
    import arithmetic.gaussian as gs
    return gs.get_coeffs(a,M)
