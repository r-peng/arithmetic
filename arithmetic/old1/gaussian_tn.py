import numpy as np
import quimb.tensor as qtn
import math
import arithmetic.gaussian as gs
def get_diag_row_top(N,tensors,A,tr,o):
    X,XSQ,CPd,L,R,ADD,CP2,CP2d = tensors
    d = X.shape[0]
    row = []
    tmp = XSQ.copy()
    tmp[:,1] *= 0.5*A[0,0]
    tmp = np.einsum('pqr,qi,p->ir',CPd,tmp,tr)
    inds = 'i0,1,','x0,{},'.format(N-1)
    inds = ['o{}'.format(o)+ind for ind in inds]
    row.append(qtn.Tensor(data=tmp,inds=inds))
    for i in range(1,N):
        tmp = XSQ.copy()
        tmp[:,1] *= 0.5*A[i,i]
        tmp = np.einsum('pqr,qi,p->ir',CPd,tmp,tr)
        tmp = np.einsum('ir,ijk->jkr',tmp,ADD)
        inds = 'i{},{},'.format(i-1,i),'i{},{},'.format(i,i+1),'x{},0,'.format(i)
        inds = ['o{}'.format(o)+ind for ind in inds]
        row.append(qtn.Tensor(data=tmp,inds=inds))
    return row
def get_diag_row(N,tensors,A,o):
    X,XSQ,CPd,L,R,ADD,CP2,CP2d = tensors
    d = X.shape[0]
    row = []
    tmp = XSQ.copy()
    tmp[:,1] *= 0.5*A[0,0]
    tmp = np.einsum('pqr,qi->ipr',CPd,tmp)
    inds = 'o{}i0,1,'.format(o),'o{}x0,'.format(o+1),'o{}x0,{},'.format(o,N-1)
    row.append(qtn.Tensor(data=tmp,inds=inds))
    for i in range(1,N):
        tmp = XSQ.copy()
        tmp[:,1] *= 0.5*A[i,i]
        tmp = np.einsum('pqr,qi->ipr',CPd,tmp)
        tmp = np.einsum('ipr,ijk->jkpr',tmp,ADD)
        inds = 'o{}i{},{},'.format(o,i-1,i),'o{}i{},{},'.format(o,i,i+1),\
               'o{}x{},'.format(o+1,i),'o{}x{},0,'.format(o,i)
        row.append(qtn.Tensor(data=tmp,inds=inds))
    return row
def get_off_diag_row(N,tensors,A,a,b,i,o,an):
    X,XSQ,CPd,L,R,ADD,CP2,CP2d = tensors
    d = X.shape[0]
    row = []
    # left
    r = 'x{},'.format(b) if i==b+1 else 'x{},{},'.format(b,i-1)
    inds = 'j{},{},'.format(b,i),'x{},{},'.format(b,i),r
    inds = ['o{}'.format(o)+ind for ind in inds]
    row.append(qtn.Tensor(data=L,inds=inds))
    # right
    if b==N-2 and i==N-1:
        r = 'x{},'.format(i)
    elif i==b+1:
        r = 'x{},{},'.format(i,N-1)
    else:
        r = 'x{},{},'.format(i,b+1)
    inds = 'j{},{},'.format(b,i),'j{},{},'.format(i,b),'x{},{},'.format(i,b),r
    inds = ['o{}'.format(o)+ind for ind in inds]
    row.append(qtn.Tensor(data=R,inds=inds))
    # 
    tmp = np.einsum('ijk,klm,j->ilm',CP2,ADD,np.array([1.0,A[b,i]]))
    if i==N-1:
        i2 = 'i{},{},'.format(N-1,N) if b==0 else 'k{},{},'.format(b,b-1)
    else:
        i2 = 'k{},{},'.format(i+1,b)
    inds = 'j{},{},'.format(i,b),i2,'k{},{},'.format(i,b)
    inds = ['o{}'.format(o)+ind for ind in inds]
    row.append(qtn.Tensor(data=tmp,inds=inds))
    #
    if b==N-2 and i==N-1:
        tmp = np.einsum('ijk,klm,l->ijm',CP2,ADD,np.array([1.0,a]))
        if an is None:
            inds = 'o{}k{},{},'.format(o,i,b),'o{}'.format(o+1),'o{}'.format(o)
        else:
            tmp = np.einsum('ijm,j->im',tmp,np.array([1.0,an]))
            inds = 'o{}k{},{},'.format(o,i,b),'o{}'.format(o)
        row.append(qtn.Tensor(data=tmp,inds=inds))
    return row      
def get_off_diag_row_bottom(N,tensors,A,a,b,i,o):
    X,XSQ,CPd,L,R,ADD,CP2,CP2d = tensors
    d = X.shape[0]
    row = []
    # left
    if i==b+1:
        inds = 'j{},{},'.format(b,i),'x{},{},'.format(b,i)
        inds = ['o{}'.format(o)+ind for ind in inds]
        row.append(qtn.Tensor(data=X.T,inds=inds))
    else:
        inds = 'j{},{},'.format(b,i),'x{},{},'.format(b,i),'x{},{},'.format(b,i-1)
        inds = ['o{}'.format(o)+ind for ind in inds]
        row.append(qtn.Tensor(data=L,inds=inds))
    # right
    inds = 'j{},{},'.format(b,i),'j{},{},'.format(i,b),'x{},{},'.format(i,b)
    tmp = R.copy()
    if b==N-2 and i==N-1:
        tmp = np.einsum('ri,ijk->jkr',X,CP2)
    else:
        r = 'x{},{},'.format(i,N-1) if i==b+1 else 'x{},{},'.format(i,b+1)
        inds = inds+(r,)
    inds = ['o{}'.format(o)+ind for ind in inds]
    row.append(qtn.Tensor(data=tmp,inds=inds))
    #
    tmp = np.einsum('ijk,klm,j->ilm',CP2,ADD,np.array([1.0,A[b,i]]))
    if i==N-1:
        i2 = 'i{},{},'.format(N-1,N) if b==0 else 'k{},{},'.format(b,b-1)
    else:
        i2 = 'k{},{},'.format(i+1,b)
    inds = 'j{},{},'.format(i,b),i2,'k{},{},'.format(i,b)
    inds = ['o{}'.format(o)+ind for ind in inds]
    row.append(qtn.Tensor(data=tmp,inds=inds))
    #
    if b==N-2 and i==N-1:
        tmp = np.einsum('ijk,kl,l->ij',CP2,ADD[:,:,1],np.array([1.0,a]))
        inds = 'o{}k{},{},'.format(o,i,b),'o{}'.format(o+1)
        row.append(qtn.Tensor(data=tmp,inds=inds))
    return row      
def energy_top(N,tensors,A,a,an,tr,o):
    arrays = get_diag_row_top(N,tensors,A,tr,o)
    for b in range(N-1):
        for i in range(N-1,b,-1):
            arrays += get_off_diag_row(N,tensors,A,a,b,i,o,an)
    return arrays
def energy_intermediate(N,tensors,A,a,o):
    arrays = get_diag_row(N,tensors,A,o)
    for b in range(N-1):
        for i in range(N-1,b,-1):
            arrays += get_off_diag_row(N,tensors,A,a,b,i,o,None)
    return arrays
def energy_bottom(N,tensors,A,a,o):
    arrays = get_diag_row(N,tensors,A,o)
    for b in range(N-1):
        for i in range(N-1,b,-1):
            arrays += get_off_diag_row_bottom(N,tensors,A,a,b,i,o)
    return arrays
def energy(N,xs,A,tr):
    tensors = gs.get_tensors(N,xs)
    arrays = get_diag_row_top(N,tensors,A,tr,1)
    for b in range(N-1):
        for i in range(N-1,b,-1):
            arrays += get_off_diag_row_bottom(N,tensors,A,0.0,b,i,1)
    arrays.append(qtn.Tensor(data=np.array([1.0,1.0]),inds='o0,'))
    return qtn.TensorNetwork(arrays)
def pol(N,xs,A,coeff,tr):
    tensors = gs.get_tensors(N,xs)
    arrays = energy_top(N,tensors,A,coeff[-2],coeff[-1],tr,len(coeff)-1)
    for i in range(len(coeff)-3,0,-1):
        arrays += energy_intermediate(N,tensors,A,coeff[i],i+1)
    arrays += energy_bottom(N,tensors,A,coeff[0],1)
    return qtn.TensorNetwork(arrays)
get_coeffs = gs.get_coeffs
