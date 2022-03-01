import numpy as np
import quimb.tensor as qtn
import cotengra as ctg
import arithmetic.utils as utils
import itertools,functools,multiprocessing
np.set_printoptions(suppress=True,linewidth=100)
def morse(ri,rj,De=1.0,a=1.0,re=1.0):
    r = np.linalg.norm(ri-rj)
    return De*(1.0-np.exp(-a*(r-re)))**2
def get_data(key,r,beta,v_params,regularize=True,split=True):
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
    return key,data,expo
def split(info):
    key,data = info
    u,s,v = np.linalg.svd(data,full_matrices=False)
    s = np.sqrt(s)
    u = np.einsum('ik,k->ik',u,s)
    v = np.einsum('kj,k->jk',v,s)
    return key,u,v
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
    for (key,data,expo) in ls:
        data_map[key] = data 
        exponent += expo
    return data_map,exponent
def split_data_map(data_map,nworkers=5):
    pool = multiprocessing.Pool(nworkers)
    ls = [(key,val) for key,val in data_map.items()]
    fxn = split
    ls = pool.map(fxn,ls)
    data_map_split = dict()
    for (key,ri,rj) in ls:
        data_map_split[key] = ri,rj
    return data_map_split
def get_hypergraph(tr,data_map):
    # regularize: set the largest value of each tensor to 1
    N,g = tr.shape
    tn = qtn.TensorNetwork([])
    for (i,j),data in data_map.items():
        inds = 'r{}'.format(i),'r{}'.format(j)
        tags = set(inds).union({'exp'})
        tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags=tags))
    for i in range(N):
        inds = ('r{}'.format(i),)
        tags = set(inds).union({'w'})
        tn.add_tensor(qtn.Tensor(data=tr[i,:],inds=inds,tags=tags))
    return tn
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
def get_circular(tr,data_map,nworkers=5):
    N,g = tr.shape
    pool = multiprocessing.Pool(nworkers)
    fxn = functools.partial(get_1D,tr=tr,data_map=data_map)
    ls = pool.map(fxn,range(N))
    arrs = []
    for arr_i in ls:
        arrs += arr_i
    tn = qtn.TensorNetwork(arrs)
    return tn
def draw(tn,N,R=5.0):
    theta = 2.0*np.pi/tn.num_tensors
    cnt = 0
    fixer = dict()
    for i in range(N-1,-1,-1):
        for j in range(N-1):
            tags = 'r{}'.format(i),'I{}'.format(j)
            fixer[tags] = np.sin(cnt*theta),np.cos(cnt*theta)
            cnt += 1
    color = ['r{}'.format(i) for i in range(N)]
    fig = tn.draw(fix=fixer,color=color,show_inds=False,show_tags=False,
                  return_fig=True)
    return fig
def contract(tn,N,alternate=True,**compress_opts):
    ls = list(range(N))
    if alternate:
        order = []
        while len(ls)>1:
            order.append(ls.pop(0))
            order.append(ls.pop(-1))
        if len(ls)>0:
            assert len(ls)==1
            order.append(ls.pop())
    else:
        order = ls
        order.reverse()
    total_swap = 0
    for i in range(1,N):
        c0 = order[:i]
        c1 = [order[i]]
        tn,num_swap = contract_binary(c0,c1,tn,**compress_opts) 
        print('i={},num swap={},max_bond={}'.format(i,num_swap,tn.max_bond()))
        total_swap += num_swap
    print('total_swap=',total_swap)
    return tn.contract()
def contract_binary(c0,c1,tn,**compress_opts):
    # determin rel position of child nodes
    tag0 = ['r{}'.format(i) for i in list(c0)]
    tag1 = ['r{}'.format(i) for i in list(c1)]
    L0 = len(tn.select_tensors(tag0,which='any'))
    L1 = len(tn.select_tensors(tag1,which='any'))
    h0,t0 = tn[tag0+['I{}'.format(0)]],tn[tag0+['I{}'.format(L0-1)]]
    h1,t1 = tn[tag1+['I{}'.format(0)]],tn[tag1+['I{}'.format(L1-1)]]
    t0h1 = not set(t0.inds).isdisjoint(set(h1.inds))
    t1h0 = not set(t1.inds).isdisjoint(set(h0.inds))
    assert t0h1 or t1h0
    if t0h1:
        left_tag,right_tag = tag0,tag1
        left_L,right_L = L0,L1
    else:
        left_tag,right_tag = tag1,tag0
        left_L,right_L = L1,L0

    # merge tail of left
    tag0,tag1 = left_tag+['I{}'.format(left_L-1)],left_tag+['I{}'.format(left_L-2)]
    tn.contract_between(tag0,tag1)
    # merge head of right
    tag0,tag1 = right_tag+['I{}'.format(0)],right_tag+['I{}'.format(1)]
    tn.contract_between(tag0,tag1)
    # remove multibond & 2-leg tsrs 
    tn.fuse_multibonds_()
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
            return tn,0

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
    # get edges
    pixs = [t.inds[-1] for t in tsrs]
    edges = []
    dists = []
    for jl,e in enumerate(pixs):
        if e not in edges:
            for jr in range(jl+1,len(pixs)):
                if pixs[jr]==e:
                    edges.append(e)
                    dists.append(jr-jl)
    idxs = np.argsort(dists)
    edges = [edges[j] for j in idxs] 
    num_swap = 0
    for e in edges: # edges in addition to the left-right edge
        assert mps.L==len(pixs)
        site = []
        for j,pix in enumerate(pixs):
            if pix==e:
                site.append(j)
        assert len(site)==2
        jl,jr = site
        assert jl<jr
        num_swap += jr-(jl+1)
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
    return tn,num_swap
#def gen_path(N,g):
#    inputs = []
#    size_dict = {}
#    for i in range(N): # tensor
#        inds = []
#        for j in range(N):
#            order = (i,j) if i<j else (j,i) 
#            idx = 'b{},{}'.format(*order)
#            if i!=j:
#                inds.append(idx)
#                size_dict[idx] = g
#        inputs.append(tuple(inds))
#    output = []
#    hg = ctg.HyperGraph(inputs,output,size_dict)
#    opt = ctg.ReusableHyperOptimizer(
#        minimize='flops',
#        reconf_opts={'forested':True},
#        #slicing_reconf_opts={'target_size':2**30},
#        parallel='ray',
#        progbar=True,
#        directory='N{}_path'.format(N))
#    tree = opt.search(inputs,output,size_dict)
#    return hg,tree
delete_tn_from_disc = utils.delete_tn_from_disc
load_tn_from_disc = utils.load_tn_from_disc
write_tn_to_disc = utils.write_tn_to_disc
