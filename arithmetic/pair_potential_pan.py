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
    rtag = 'r{}'.format(i)
    for j,data in enumerate(arr):
        rix = rtag+'_'+pix[j]  +'_'+pix[j+1] if j<N-2 else None
        lix = rtag+'_'+pix[j-1]+'_'+pix[j]   if j>0 else None
        tags = rtag,'I{}'.format(j)
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
def contract(left,right,edges,tn,**compress_opts):
    # merge tail of left
    left_tag = ['r{}'.format(i) for i in left]
    left_L = len(tn.select_tensors(left_tag,which='any'))
    tag0,tag1 = left_tag+['I{}'.format(left_L-1)],left_tag+['I{}'.format(left_L-2)]
    tn.contract_between(tag0,tag1)
    # merge head of right
    right_tag = ['r{}'.format(i) for i in right]
    right_L = len(tn.select_tensors(right_tag,which='any'))
    tag0,tag1 = right_tag+['I{}'.format(0)],right_tag+['I{}'.format(1)]
    tn.contract_between(tag0,tag1)
    tn.fuse_multibonds_()
    # 
    rmin,lmax = 1,left_L-2
    tag0,tag1 = left_tag+['I{}'.format(lmax)],right_tag+['I{}'.format(rmin)]
    assert len(tn[tag0].inds)==len(tn[tag1].inds)
    while len(tn[tag0].inds)<=2:
        if tn.num_tensors>2:
            tn.contract_between(tag0,tag1)
            tag = tag0+tag1
            rmin += 1
            lmax -= 1
            tag0,tag1 = left_tag+['I{}'.format(lmax)],right_tag+['I{}'.format(rmin)]
            tn[tag].modify(tags=tag0)
            tn.contract_tags(tag0,which='all',inplace=True)    
            tn.fuse_multibonds_()
        else:
            return tn

    left_tsr = []
    for j in range(lmax+1):
        tags = left_tag+['I{}'.format(j)]
        tid = tuple(tn._get_tids_from_tags(tags,which='all'))[0]
        left_tsr.append(tn._pop_tensor(tid))
    right_tsr = []
    for j in range(rmin,right_L):
        tags = right_tag+['I{}'.format(j)]
        tid = tuple(tn._get_tids_from_tags(tags,which='all'))[0]
        right_tsr.append(tn._pop_tensor(tid))
    full_tag = left_tag+right_tag
    # make full mps
    def transpose(ls):
        L = len(ls)
        for j in range(L):
            t = ls[j] 
            if j>0:
                lix = tuple(t.bonds(tsrs[j-1]))[0]
            if j<L-1:
                rix = tuple(t.bonds(tsrs[j+1]))[0]
            if j==0:
                vix = (rix,)
            elif j==L-1:
                vix = (lix,)
            else:
                vix = (lix,rix)
            pix = tuple(set(t.inds).difference(set(vix)))
            assert len(pix)==1
            output_inds = vix+pix
            t.transpose_(*output_inds)
        return ls
    tsrs = left_tsr+right_tsr
    tsrs = transpose(tsrs)
    arrs = [t.data for t in tsrs]
    mps = qtn.MatrixProductState(arrs,shape='lrp',tags=full_tag)
    pixs = [t.inds[-1] for t in tsrs]
    for e in edges: # edges in addition to the left-right edge
        assert mps.L==len(pixs)
        # get site 
        site = []
        for j in range(mps.L):
            if e==pixs[j]:
                site.append(j)
        assert len(site)==2
        jl,jr = site
        pixs = pixs[:jl]+pixs[jl+1:jr]+pixs[jr+1:]
        # mps swap & contract
        mps.swap_site_to(i=jr,f=jl+1,inplace=True,**compress_opts)
        old,new = mps._site_ind_id.format(jl+1),mps._site_ind_id.format(jl)
        mps[mps.site_tag(jl+1)].reindex_({old:new})
        mps.contract_between(mps.site_tag(jl),mps.site_tag(jl+1))
        mps.contract_between(mps.site_tag(jl+1),mps.site_tag(jl+2))
        tsrs = [mps[mps.site_tag(j)] 
                for j in tuple(range(jl))+tuple(range(jl+2,mps.L))]
        tsrs = transpose(tsrs)
        arrs = [t.data for t in tsrs]
        mps = qtn.MatrixProductState(arrs,shape='lrp',tags=full_tag)
    # add to tn
    for j in range(mps.L):
        tsr = mps[mps.site_tag(j)]
        tsr.reindex_({mps._site_ind_id.format(j):pixs[j]})
        tn.add_tensor(tsr)
    return tn
