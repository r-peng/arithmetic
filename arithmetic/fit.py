import numpy as np
import quimb.tensor as qtn
import quimb.tensor.tensor_core as qtc
def add_local(tn,i,k,N,n,tensors):
    # between col:i,i+1 and row:k-1,k
    # indexed by i*n+k for i in [0,N-1], k in [1,n]
    size = n*N
    if k==1:
        t = tn.tensor_map[i*n]
        inds = list(t.inds).copy()
        inds[-1] = 'ir{},0,'.format(i) 
        t.modify(inds=inds)
        t = tn.tensor_map[(i+1)*n]
        inds = list(t.inds).copy()
        idx = -1 if i==N-2 else -2
        inds[idx] = 'il{},0,'.format(i+1) 
        t.modify(inds=inds)
    else:
        t = tn.tensor_map[i*n+k-1+size]
        inds = list(t.inds).copy()
        inds[-1] = 'kr{},{},'.format(i,k-1)
        t.modify(inds=inds)
        t = tn.tensor_map[(i+1)*n+k-1+size*2]
        inds = list(t.inds).copy()
        inds[-1] = 'kl{},{},'.format(i+1,k-1) 
        t.modify(inds=inds)

    t = tn.tensor_map[i*n+k]
    inds = list(t.inds).copy()
    inds[-1] = 'ir{},{},'.format(i,k) 
    t.modify(inds=inds)
    t = tn.tensor_map[(i+1)*n+k]
    inds = list(t.inds).copy()
    idx = -1 if i==N-2 else -2
    inds[idx] = 'il{},{},'.format(i+1,k) 
    t.modify(inds=inds)
    
    i1 = 'ir{},0,'.format(i) if k==1 else 'kr{},{},'.format(i,k-1)
    inds = i1,'ir{},{},'.format(i,k),'k{},{},'.format(i+1,k)
    tags = {'r','{},{},'.format(i,k)}
    tid = i*n+k+size
    tn.add_tensor(qtn.Tensor(data=tensors[0],inds=inds,tags=tags),tid=tid) 
    i1 = 'il{},0,'.format(i+1) if k==1 else 'kl{},{},'.format(i+1,k-1)
    inds = i1,'il{},{},'.format(i+1,k),'k{},{},'.format(i+1,k)
    tags = {'l','{},{},'.format(i+1,k)}
    tid = (i+1)*n+k+size*2
    tn.add_tensor(qtn.Tensor(data=tensors[1],inds=inds,tags=tags),tid=tid)
def fit_local(tn,tn_target,i,k,N,n,**kwargs):
    size = n*N
    for tid in [i*n+k+size,(i+1)*n+k+size*2]:
        t = tn.tensor_map[tid]
        tags = set(t.tags)
        tags.add('fit')
        t.modify(tags=tags)
#    print(tn)
    qtc.tensor_network_fit_autodiff(tn,tn_target,tags='fit',inplace=True,**kwargs)
    for tid in [i*n+k+size,(i+1)*n+k+size*2]:
        t = tn.tensor_map[tid]
        tags = set(t.tags).copy()
        tags.discard('fit')
        t.modify(tags=tags)
#    print(tn)
def get_tensor(k,max_bond):
    dim1 = 2 if k==1 else max_bond
    scale = 1e-3
    tmp = np.einsum('ka,ib->kiab',np.eye(dim1)+np.random.rand(dim1,dim1)*scale,
                                  np.eye(2)+np.random.rand(2,2)*scale)
    tmp = tmp.reshape(dim1,2,dim1*2)
    if dim1*2>max_bond:
        tmp = tmp[:,:,:max_bond]
    tmp += np.random.rand(*tmp.shape)*scale
    tmp = np.random.rand(*tmp.shape)
    return tmp
def fit_from_left(tn,N,n,max_bond=None,max_iter=100,thresh=1e-6,**kwargs):
    original = tn.copy()
    norm = abs((original|original.H).contract())**0.5
    print('norm=',norm)
    max_bond = 2**n if max_bond is None else max_bond
    for i in range(N-1):
        for k in range(1,n):
            tensor = get_tensor(k,max_bond)
            add_local(tn,i,k,N,n,(tensor,tensor))
            fit_local(tn,original,i,k,N,n,**kwargs)
    for it in range(max_iter):
        dist = qtc.tensor_network_distance(tn,original)
        print('iter={},dist={}'.format(it,dist/norm))
        if dist/norm < thresh:
            break
        for i in range(N-1):
            for k in range(1,n):
                fit_local(tn,original,i,k,N,n,**kwargs)
    return tn,dist/norm
def fit_from_top(tn,N,n,max_bond=None,max_iter=100,thresh=1e-6,**kwargs):
    original = tn.copy()
    norm = abs((original|original.H).contract())**0.5
    print('norm=',norm)
    scale = 1e-1
    max_bond = 2**n if max_bond is None else max_bond
    tensor = np.random.rand(max_bond,2,max_bond)*scale
    for k in range(1,n):
        for i in range(N-1):
            tmp = tensor[:2,:,:] if k==1 else tensor.copy()
            add_local(tn,i,k,N,n,(tmp,tmp))
            fit_local(tn,original,i,k,N,n,**kwargs)
    for it in range(max_iter):
        dist = qtc.tensor_network_distance(tn,original)
        print('iter={},dist={}'.format(it,dist/norm))
        if dist/norm < thresh:
            break
        for k in range(1,n):
            for i in range(N-1):
                fit_local(tn,original,i,k,N,n,**kwargs)
    return tn,dist/norm
