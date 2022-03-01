import numpy as np
import quimb.tensor as qtn
import arithmetic.utils as utils
import functools
import multiprocessing
def morse(ri,rj,De=1.0,a=1.0,re=1.0):
    r = np.linalg.norm(ri-rj)
    return De*(1.0-np.exp(-a*(r-re)))**2
def get_data(key,r,beta,v_params,regularize=True):
    # ri.shape = rj.shape = g,3
    N,g,_ = r.shape
    i,j = key
    assert i<j
    data = np.zeros((g,)*2)
    for k in range(g):
        for l in range(g):
            hij = morse(r[i,k,:],r[j,l,:],**v_params)
            data[k,l] = np.exp(-beta*hij)
    if regularize:
        data_max = np.amax(data)
        data /= data_max
        expo = np.log10(data_max)
    else:
        expo = 0.0
    u,s,v = np.linalg.svd(data,full_matrices=False)
    s = np.sqrt(s)
    u = np.einsum('ik,k->ik',u,s)
    v = np.einsum('kj,k->jk',v,s)
    return key,u,v,expo
def get_data_map(r,beta,v_params,regularize=True,nworkers=5):
    N,g,_ = r.shape
    pool = multiprocessing.Pool(nworkers)
    ls = []
    for i in range(N):
        for j in range(i+1,N):
            ls.append((i,j))
    fxn = functools.partial(get_data,r=r,beta=beta,v_params=v_params,
                            regularize=regularize)
    ls = pool.map(fxn,ls)
    data_map = dict()
    exponent = 0.0
    for (key,xi,xj,expo) in ls:
        data_map[key] = xi,xj
        exponent += expo
    return data_map,exponent
def get_1D(i,tr,data_map):
    N,g = tr.shape
    cp = np.zeros((g,)*3)
    for gi in range(g):
        cp[(gi,)*3] = 1.0
    # Head
    H_arr = []
    H_pix = []
    for j in range(i+1,N):
        data = data_map[i,j][0]
        if j>i+1:
            data = np.einsum('gk,efg->efk',data,cp)
        H_arr.append(data)
        H_pix.append('r{},{}'.format(i,j))
    # Tail
    T_arr = []
    T_pix = []
    for j in range(i):
        data = data_map[j,i][1]
        if j<i-1:
            data = np.einsum('gk,efg->efk',data,cp)
        T_arr.append(data)
        T_pix.append('r{},{}'.format(j,i))
    # merge
    if i==0:
        H_arr[-1] = np.einsum('efk,f->ek',H_arr[-1],tr[i,:])
    elif i==N-1:
        T_arr[0] = np.einsum('efk,e->fk',T_arr[0],tr[i,:])
    else:
        T_arr[0] = np.einsum('e...,e->e...',T_arr[0],tr[i,:])
    arr = H_arr+T_arr
    pix = H_pix+T_pix
    ls = []
    tag = 'r{}'.format(i)
    for j,data in enumerate(arr):
        rix = tag+'_'+pix[j]  +'_'+pix[j+1] if j<N-2 else None
        lix = tag+'_'+pix[j-1]+'_'+pix[j]   if j>0 else None
        tags = tag,'I{}'.format(j)
        if j==0:
            inds = rix,pix[j]
        elif j==N-2:
            inds = lix,pix[j]
        else:
            inds = lix,rix,pix[j] 
        ls.append(qtn.Tensor(data=data,inds=inds,tags=tags))
    return ls
def get_tn(tr,data_map,nworkers=5):
    N,g = tr.shape
    pool = multiprocessing.Pool(nworkers)
    fxn = functools.partial(get_1D,tr=tr,data_map=data_map)
    ls = pool.map(fxn,range(N))
    arrs = []
    for arr_i in ls:
        arrs += arr_i
    tn = qtn.TensorNetwork(arrs)
    return tn
def contract_pair(pair):
    c1,c2 = node_pair # left/right child node
    assert set(c1).isdisjoint(set(c2))
    c1,c2 = list(c1),list(c2)
    c1.sort()
    c2.sort()
    # assert adjacent
    adj = None
    for i in c1:
        for j in c2:
            if (i-(j+1))%N==0:
                adj = c1,c2
            elif (j-(i+1))%N==0:
                adj = c2,c1
    assert adj is not None
    left,right = adjacent
    
