import numpy as np
import quimb.tensor as qtn


def get_row(N,xmap,cexp,cpol=None):
    inds = list(xmap.keys())
    inds.sort()
    d = xmap[inds[0]].shape[-1]
    row = []
    for i in range(len(inds)):
        start = 0 if i==0 else inds[i-1]+1
        stop = inds[i]
        for j in range(start,stop):
            row.append(np.einsum('ij,pq->ijpq',np.eye(2),np.eye(d)))
        row.append(xmap[stop])
    for j in range(inds[-1]+1,N):
        row.append(np.einsum('ij,pq->ijpq',np.eye(2),np.eye(d)))
    CP = np.zeros((2,)*3)
    CP[0,0,0] = CP[1,1,1] = 1.0 
    ADD = np.zeros((2,)*3)
    ADD[0,0,0] = ADD[0,1,1]= ADD[1,0,1] = 1.0
    coeff = np.array([1.0,cexp])
    if cpol is None:
        row.append(np.einsum('ijk,klm,j->ilm',CP,ADD,coeff).reshape(2,1,2,2))
        row.append(np.eye(2).reshape(1,2,2))
    else:
        row.append(np.einsum('ijk,klm,j->iml',CP,ADD,coeff).reshape(2,2,2,1))
        coeff = np.array([1.0,cpol])
        row.append(np.einsum('ijk,klm,l->ijm',CP,ADD,coeff))
    return row
def cap_top(arrays,tr,an):
    row = arrays[0]
    N = len(row)-2
    for i in range(N):
        row[i] = np.einsum('...pq,p->...q',row[i],tr)
    row[N] = np.einsum('...pq,p->...q',row[N],np.array([1.0]))
    row[N+1] = np.einsum('...pq,p->...q',row[N+1],np.array([1.0,an]))
    arrays[0] = row
    return arrays
def cap_bottom(arrays):
    row = arrays[-1]
    N = len(row)-2
    d = row[0].shape[-1]
    for i in range(N):
        row[i] = np.einsum('...q,q->...',row[i],np.ones(d))
    row[N] = np.einsum('...q,q->...',row[N],np.array([1.0]))
    row[N+1] = np.einsum('...q,q->...',row[N+1],np.array([0.0,1.0]))
    arrays[-1] = row
    return arrays
def cap_left(arrays):
    nrows = len(arrays)
    for i in range(nrows):
        arrays[i][0] = np.einsum('i...,i->...',arrays[i][0],np.ones(2))
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
def get_block(N,Xs,ai,A,B=None):
    arrays = []
    for i in range(N):
        for j in range(N):
            xmap = get_xmap([i,j],Xs)
            cpol = ai if i==j==N-1 and B is None else None 
            arrays.append(get_row(N,xmap,A[i,j],cpol=cpol))
    if B is not None:
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        xmap = get_xmap([i,j,k,l],Xs)
                        cpol = ai if i==j==k==l==N-1 else None 
                        arrays.append(get_row(N,xmap,B[i,j,k,l],cpol=cpol))
    row = arrays[0]
    row[-2] = row[-2][:,:,:1,:]
    return arrays
def get_Xs(xs):
    d = len(xs)
    CPd = np.zeros((d,)*3)
    for i in range(d):
        CPd[i,i,i] = 1.0
    CP2 = np.zeros((2,)*3)
    CP2[0,0,0] = CP2[1,1,1] = 1.0 

    X1 = np.zeros((d,2))
    X1[:,1] = xs.copy()
    X1[:,0] = np.ones(d)
    X2 = np.multiply(X1,X1)
    X3 = np.multiply(X2,X1)
    X4 = np.multiply(X3,X1)
    return [np.einsum('pqr,ri,ijk->jkpq',CPd,X,CP2) for X in [X1,X2,X3,X4]]
def energy(N,xs,tr,A,B=None):
    Xs = get_Xs(xs)
    arrays = get_block(N,Xs,0.0,A,B)
    arrays = cap_top(arrays,tr,1.0)
    arrays = cap_bottom(arrays)
    arrays = cap_left(arrays)
    return to_peps(arrays)
def pol(N,xs,tr,coeff,A,B=None):
    Xs = get_Xs(xs)
    arrays = []
    for i in range(len(coeff)-2,-1,-1):
        arrays += get_block(N,Xs,coeff[i],A,B)
    arrays = cap_top(arrays,tr,coeff[-1])
    arrays = cap_bottom(arrays)
    arrays = cap_left(arrays)
    return to_peps(arrays)
def get_coeffs(a,M):
    import arithmetic.gaussian as gs
    return gs.get_coeffs(a,M)
