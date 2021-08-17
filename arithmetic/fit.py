import numpy as np
import quimb.tensor as qtn
import quimb.tensor.tensor_core as qtc
def add_local(tn,i,k,N,n,max_bond,guess='svd',**kwargs):
    # between col:i,i+1 and row:k-1,k
    # indexed by i*n+k for i in [0,N-1], k in [1,n]
    # guess = {'svd','fit','random'}
    size = n*N
    if guess!='random':
        target_left = get_local_left(tn,i,k,N,n)
        target_right = get_local_right(tn,i,k,N,n)
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

    dim1 = 2**k
    dim2 = dim1*2
    if dim2<=max_bond:
        dim1 = dim1
        dim2 = dim2
    elif dim1<=max_bond:
        dim1 = dim1
        dim2 = max_bond
    else:
        dim1 = max_bond
        dim2 = max_bond
    if guess=='random': 
        left = np.random.rand(dim1,2,dim2)
        right = np.random.rand(dim1,2,dim2)
    else:
        if guess=='svd':
            left  = target_left.contract()
            right = target_right.contract()
            _, shared_inds, _ = qtc.group_inds(left,right)
            o = shared_inds[0]
            qtc.tensor_compress_bond(left,right,max_bond=max_bond,cutoff=0.0)
        elif guess=='fit':
            o = 'i{},0,'.format(i+1) if k==1 else 'k{},{},'.format(i+1,k-1)
        fit_left = get_local_left(tn,i,k,N,n)
        fit_right = get_local_right(tn,i,k,N,n)
        i1 = 'ir{},0,'.format(i) if k==1 else 'kr{},{},'.format(i,k-1)
        inds = i1,'ir{},{},'.format(i,k),o
        tags = {'fit','left'}
        data = np.random.rand(dim1,2,dim2)
        fit_left.add_tensor(qtn.Tensor(data=data,inds=inds,tags=tags)) 
        i1 = 'il{},0,'.format(i+1) if k==1 else 'kl{},{},'.format(i+1,k-1)
        inds = i1,'il{},{},'.format(i+1,k),o
        tags = {'fit','right'}
        data = np.random.rand(dim1,2,dim2)
        fit_right.add_tensor(qtn.Tensor(data=data,inds=inds,tags=tags))
        if guess=='svd':
            qtc.tensor_network_fit_autodiff(fit_left,qtn.TensorNetwork([left]),
                                            tags='fit',inplace=True,**kwargs)
            qtc.tensor_network_fit_autodiff(fit_right,qtn.TensorNetwork([right]),
                                            tags='fit',inplace=True,**kwargs)
            left = fit_left.select_tensors({'fit','left'})[0].data 
            right = fit_right.select_tensors({'fit','right'})[0].data 
        elif guess=='fit':
            target = target_left.copy()
            target.add_tensor_network(target_right)
            fit = fit_left.copy()
            fit.add_tensor_network(fit_right)
            qtc.tensor_network_fit_autodiff(fit,target,
                                            tags='fit',inplace=True,**kwargs)
            left = fit.select_tensors({'fit','left'})[0].data 
            right = fit.select_tensors({'fit','right'})[0].data 

    i1 = 'ir{},0,'.format(i) if k==1 else 'kr{},{},'.format(i,k-1)
    inds = i1,'ir{},{},'.format(i,k),'k{},{},'.format(i+1,k)
    tags = {'r','{},{},'.format(i,k)}
    tid = i*n+k+size
    tn.add_tensor(qtn.Tensor(data=left,inds=inds,tags=tags),tid=tid) 
    i1 = 'il{},0,'.format(i+1) if k==1 else 'kl{},{},'.format(i+1,k-1)
    inds = i1,'il{},{},'.format(i+1,k),'k{},{},'.format(i+1,k)
    tags = {'l','{},{},'.format(i+1,k)}
    tid = (i+1)*n+k+size*2
    tn.add_tensor(qtn.Tensor(data=right,inds=inds,tags=tags),tid=tid)
def get_local_left(tn,i,k,N,n):
    size = N*n
    tids = [i*n+k-1,i*n+k]
    if k>1:
        tids.append(i*n+k-1+size)
    return qtn.TensorNetwork([tn.tensor_map[tid].copy() for tid in tids])
def get_local_right(tn,i,k,N,n):
    size = N*n
    tids = [(i+1)*n+k-1,(i+1)*n+k]
    if k>1:
        tids.append((i+1)*n+k-1+size*2)
    return qtn.TensorNetwork([tn.tensor_map[tid].copy() for tid in tids])
def get_local(tn,i,k,N,n):
    out = get_local_left(tn,i,k,N,n)
    out.add_tensor_network(get_local_right(tn,i,k,N,n)) 
    return out
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
def fit_from_left(tn,N,n,max_bond=None,max_iter=100,thresh=1e-6,**kwargs):
    original = tn.copy()
    norm = abs((original|original.H).contract())**0.5
    print('norm=',norm)
    max_bond = 2**n if max_bond is None else max_bond
    print('adding fit tensors')
    for i in range(N-1):
        for k in range(1,n):
            add_local(tn,i,k,N,n,max_bond,**kwargs)
            fit_local(tn,original,i,k,N,n,**kwargs)
    print('iterating fit tensors')
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
