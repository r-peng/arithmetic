import numpy as np
import matplotlib.pyplot as plt
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
    tag = 'r{}'.format(i)
    for j,data in enumerate(arr):
        rix = tag+'_'+pix[j]  +'_'+pix[j+1] if j<N-2 else None
        lix = tag+'_'+pix[j-1]+'_'+pix[j]   if j>0 else None
        if j==0:
            inds = rix,pix[j]
        elif j==N-2:
            inds = lix,pix[j]
        else:
            inds = lix,rix,pix[j] 
        ls.append(qtn.Tensor(data=data,inds=inds,tags={tag,'I{}'.format(j)}))
    return ls
def get_circular(tr,data_map,nworkers=5):
    N,g = tr.shape
    pool = multiprocessing.Pool(nworkers)
    fxn = functools.partial(get_1D,tr=tr,data_map=data_map)
    ls = pool.map(fxn,range(N))
    arrs = []
    for arr_ in ls:
        arrs += arr_
    tn = qtn.TensorNetwork(arrs)
    return tn
def draw(tn,R=5.0,N=None,nodes=None):
    nodes = [[i] for i in range(N)] if nodes is None else nodes
    nodes_map = dict()
    for node in nodes:
        nodes_map[min(node)] = node
    keys = list(nodes_map.keys())
    keys.sort(reverse=True)

    theta = 2.0*np.pi/tn.num_tensors
    cnt = 0
    fixer = dict()
    for key in keys:
        node = nodes_map[key]
        tag = 'r{}'.format(node[0])
        L = len(tn.select_tensors(tag,which='any'))
        for j in range(L):
            fixer[tag,'I{}'.format(j)] = np.sin(cnt*theta),np.cos(cnt*theta)
            cnt += 1
    color = ['r{}'.format(node[0]) for node in nodes]
    fig = tn.draw(fix=fixer,color=color,show_inds=False,show_tags=False,
                  return_fig=True)
    return fig
def gen_path(N):
    stop = False
    parents = (list(range(N)),)
    order = []
    while not stop:
        parents_new = tuple()
        pairs = [] 
        for p in parents:
           L = len(p)
           l = int(L/2)
           c1,c2 = p[:l],p[l:]
           parents_new += (c1,c2)
           pairs.append((c1,c2))
        parents = parents_new
        order.insert(0,pairs)
        stop = True
        for p in parents:
            if len(p)>1:
                stop = False
                break
    return order
def contract(tn,order,max_workers=20,contract_final=True,save_as=None,
             draw_tn=False,**compress_opts):
    pair_fn = order.pop(-1)
    pair_fn = pair_fn[0]
    N = len(pair_fn[0])+len(pair_fn[1])
    nodes = [[i] for i in range(N)]
    for node in nodes:
        fname = 'N{}_tmp_'.format(N)+','.join([str(i) for i in node])
        tag = ['r{}'.format(i) for i in node]
        tni = tn.select(tag,which='any')
        write_tn_to_disc(tni,fname)
        
    total_swap = 0
    for step,pairs in enumerate(order):
        print('step=',step,pairs)
        nworkers = min(max_workers,len(pairs))
        pool = multiprocessing.Pool(nworkers)
        fxn = functools.partial(contract_binary_wrapper,N=N,**compress_opts)
        ls = pool.map(fxn,pairs)
        pool.close()
        fnames = []
        for (fname,num_swap) in ls:
            total_swap += num_swap
            fnames.append(fname)
        if draw_tn:
            tn = qtn.TensorNetwork([]) 
            for fname in fnames:
                tn.add_tensor_network(load_tn_from_disc(fname))
            nodes = []
            for pair in pairs:
                nodes.append(pair[0]+pair[1])
            fig = draw(tn,nodes=nodes)
            fig.savefig('N{}_step{}.png'.format(N,step),dpi=300)
            plt.close(fig)
    print('num_swap=',total_swap)

    tn = qtn.TensorNetwork([])
    for (fname,_) in ls:
        tn.add_tensor_network(load_tn_from_disc(fname))
        delete_tn_from_disc(fname)
    if save_as is not None:
        write_tn_to_disc(tn,save_as) 
    if contract_final:
        tn,num_swap = contract_binary(tn,*pair_fn,**compress_opts) 
        total_swap += num_swap 
        print('total_swap=',total_swap)
        return tn.contract()
    return tn
def contract_binary_wrapper(pair,N,**compress_opts):
    c1,c2 = pair
    if len(c1)==0 or len(c2)==0:
        fname = 'N{}_tmp_'.format(N)+','.join([str(i) for i in c1+c2])
        return fname,0
    fname1 = 'N{}_tmp_'.format(N)+','.join([str(i) for i in c1])
    fname2 = 'N{}_tmp_'.format(N)+','.join([str(i) for i in c2])
    tn = load_tn_from_disc(fname1)
    tn.add_tensor_network(load_tn_from_disc(fname2))
    delete_tn_from_disc(fname1)
    delete_tn_from_disc(fname2)
    tn,num_swap = contract_binary(tn,c1,c2,**compress_opts)
    print('nodes={},{},num_swap={},max_bond={}'.format(c1,c2,num_swap,tn.max_bond()))
    p = c1+c2
    p.sort()
    fname = 'N{}_tmp_'.format(N)+','.join([str(i) for i in p])
    write_tn_to_disc(tn,fname)
    return fname,num_swap
def contract_binary(tn,c1,c2,**compress_opts):
    # determin rel position of child nodes
    c1,c2 = list(c1),list(c2)
    tag1 = 'r{}'.format(c1[0])
    tag2 = 'r{}'.format(c2[0])
    L1 = len(tn.select_tensors(tag1,which='any'))
    L2 = len(tn.select_tensors(tag2,which='any'))
    h1,t1 = tn[tag1,'I{}'.format(0)],tn[tag1,'I{}'.format(L1-1)]
    h2,t2 = tn[tag2,'I{}'.format(0)],tn[tag2,'I{}'.format(L2-1)]
    t1h2 = not set(t1.inds).isdisjoint(set(h2.inds))
    t2h1 = not set(t2.inds).isdisjoint(set(h1.inds))
    assert t1h2 or t2h1
    if t1h2 and t2h1:
        assert L1==L2
        # cyclic
        return contract_cyclic(tn,c1,c2,**compress_opts)
    else:
        cl,cr = (c1,c2) if t1h2 else (c2,c1)
        return contract_open(tn,cl,cr,**compress_opts) 
def contract_open(tn,cl,cr,**compress_opts):
    ltag = 'r{}'.format(cl[0])
    rtag = 'r{}'.format(cr[0])
    Ll = len(tn.select_tensors(ltag,which='any'))
    Lr = len(tn.select_tensors(rtag,which='any'))
    rmin,lmax = 0,Ll-1
    ltags = ltag,'I{}'.format(lmax)
    rtags = rtag,'I{}'.format(rmin)
    while len(tn[ltags].inds)<=2:
        assert len(tn[ltags].inds)==len(tn[rtags].inds)
        if tn.num_tensors<=2:
            return tn,0
        tn.contract_between(ltags,rtags)
        tags = ltags+rtags
        rmin += 1
        lmax -= 1
        ltags = ltag,'I{}'.format(lmax)
        rtags = rtag,'I{}'.format(rmin)
        tn[tags].modify(tags=ltags)
        tn.contract_tags(ltags,which='all',inplace=True)    
        tn.fuse_multibonds_()

    left_tsr = []
    for j in range(lmax+1):
        tags = ltag,'I{}'.format(j)
        tid = tuple(tn._get_tids_from_tags(tags,which='all'))[0]
        left_tsr.append(tn._pop_tensor(tid))
    right_tsr = []
    for j in range(rmin,Lr):
        tags = rtag,'I{}'.format(j)
        tid = tuple(tn._get_tids_from_tags(tags,which='all'))[0]
        right_tsr.append(tn._pop_tensor(tid))
    # make full mps
    def transpose(ls):
        L = len(ls)
        for j in range(L):
            t = ls[j] 
            if j>0:
                lix = tuple(t.bonds(ls[j-1]))[0]
            if j<L-1:
                rix = tuple(t.bonds(ls[j+1]))[0]
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
    full_tag = ['r{}'.format(i) for i in cl+cr]
    tsrs = left_tsr+right_tsr
    tsrs = transpose(tsrs)
    pixs = [t.inds[-1] for t in tsrs]
    num_swap = 0
    while True: 
        arrs = [t.data for t in tsrs]
        mps = qtn.MatrixProductState(arrs,shape='lrp',tags=full_tag)
        if mps.L<=2:
            break
        assert mps.L==len(pixs)
        # get edge
        edge_map = dict()
        for jl,e in enumerate(pixs):
            if e not in edge_map:
                for jr in range(jl+1,len(pixs)):
                    if pixs[jr]==e:
                        edge_map[e] = jl,jr,jr-jl-1
        if len(edge_map)==0:
            break
        edges = list(edge_map.keys())
        dists = [edge_map[e][-1] for e in edges]
        e = edges[np.argsort(dists)[0]]
        jl,jr,d = edge_map[e]
        pixs = pixs[:jl]+pixs[jl+1:jr]+pixs[jr+1:]
        if d==0:
            jt = jl-1 if jl>0 else jr+1
            tags = [mps.site_tag(j) for j in [jl,jr,jt]]
            mps.contract_tags(tags,which='any',inplace=True)
            tsrs = [mps[mps.site_tag(j)] for j in range(jl)]
            tsrs = tsrs+[mps[mps.site_tag(j)] for j in range(jr+1,mps.L)]
        elif d%2==1:
            jm = (jl+jr)//2
            num_swap += d-1
            mps.swap_site_to(i=jl,f=jm-1,inplace=True,**compress_opts)
            mps.swap_site_to(i=jr,f=jm+1,inplace=True,**compress_opts)
            old = mps._site_ind_id.format(jm-1)
            new = mps._site_ind_id.format(jm+1)
            tags = [mps.site_tag(j) for j in [jm-1,jm,jm+1]]
            mps[mps.site_tag(jm-1)].reindex_({old:new})
            mps.contract_tags(tags,which='any',inplace=True)
            tsrs = [mps[mps.site_tag(j)] for j in range(jm-1)]
            tsrs = tsrs+[mps[mps.site_tag(jm)]]
            tsrs = tsrs+[mps[mps.site_tag(j)] for j in range(jm+2,mps.L)]
        else:
            jml,jmr = jl+d//2,jr-d//2
            num_swap += d-2 
            mps.swap_site_to(i=jl,f=jml-1,inplace=True,**compress_opts)
            mps.swap_site_to(i=jr,f=jmr+1,inplace=True,**compress_opts)
            old = mps._site_ind_id.format(jml-1)
            new = mps._site_ind_id.format(jmr+1)
            mps[mps.site_tag(jml-1)].reindex_({old:new})
            mps.contract_between(mps.site_tag(jml),mps.site_tag(jml-1))
            mps.contract_between(mps.site_tag(jmr),mps.site_tag(jmr+1))
            mps.fuse_multibonds_()
            mps.left_canonize(jml-1)
            mps.right_canonize(jmr+1)
            mps.compress_between(mps.site_tag(jml),mps.site_tag(jmr),**compress_opts)
            tsrs = [mps[mps.site_tag(j)] for j in range(jml-1)]
            tsrs = tsrs+[mps[mps.site_tag(j)] for j in [jml,jmr]]
            tsrs = tsrs+[mps[mps.site_tag(j)] for j in range(jmr+2,mps.L)]
        tsrs = transpose(tsrs)
    # add to tn
    for j in range(mps.L):
        tsr = mps[mps.site_tag(j)]
        if j>0:
            lix = '_'.join(full_tag)+'_I{},{}'.format(j-1,j)
        if j<mps.L-1:
            rix = '_'.join(full_tag)+'_I{},{}'.format(j,j+1)
        if j==0:
            inds = rix,pixs[j]
        elif j==mps.L-1:
            inds = lix,pixs[j]
        else:
            inds = lix,rix,pixs[j]
        tsr.modify(inds=inds)
        tn.add_tensor(tsr)
    return tn,num_swap
def contract_cyclic(tn,c1,c2,**compress_opts):
    tag1 = 'r{}'.format(c1[0])
    tag2 = 'r{}'.format(c2[0])
    L = tn.num_tensors//2
    min2,max1 = 0,L-1
    tags1 = tag1,'I{}'.format(max1)
    tags2 = tag2,'I{}'.format(min2)
    while len(tn[tags1].inds)<=2:
        assert len(tn[tags1].inds)==len(tn[tags2].inds)
        if tn.num_tensors<=2:
            return tn,0
        tn.contract_between(tags1,tags2)
        tags = tags1+tags2
        min2 += 1
        max1 -= 1
        tags1 = tag1,'I{}'.format(max1)
        tags2 = tag2,'I{}'.format(min2)
        tn[tags].modify(tags=tags1)
        tn.contract_tags(tags1,which='all',inplace=True)    
        tn.fuse_multibonds_()
    min1,max2 = 0,L-1
    tags1 = tag1,'I{}'.format(min1)
    tags2 = tag2,'I{}'.format(max2)
    while len(tn[tags1].inds)<=2:
        assert len(tn[tags1].inds)==len(tn[tags2].inds)
        if tn.num_tensors<=2:
            return tn,0
        tn.contract_between(tags1,tags2)
        tags = tags1+tags2
        min1 += 1
        max2 -= 1
        tags1 = tag1,'I{}'.format(min1)
        tags2 = tag2,'I{}'.format(max2)
        tn[tags].modify(tags=tags2)
        tn.contract_tags(tags2,which='all',inplace=True)    
        tn.fuse_multibonds_()
    
    tsrs1 = []
    for j in range(min1,max1+1):
        tags = tag1,'I{}'.format(j)
        tid = tuple(tn._get_tids_from_tags(tags,which='all'))[0]
        tsrs1.append(tn._pop_tensor(tid))
    tsrs2 = []
    for j in range(min2,max2+1):
        tags = tag2,'I{}'.format(j)
        tid = tuple(tn._get_tids_from_tags(tags,which='all'))[0]
        tsrs2.append(tn._pop_tensor(tid))
    def transpose(ls):
        L = len(ls)
        for j in range(L):
            tl,t,tr = ls[(j-1)%L],ls[j],ls[(j+1)%L]
            lix = tuple(t.bonds(tl))[0]
            rix = tuple(t.bonds(tr))[0]
            pix = tuple(set(t.inds).difference({lix,rix}))[0]
            t.transpose_(lix,rix,pix)
        return ls
    tsrs = tsrs1+tsrs2
    tsrs = transpose(tsrs)
    pixs = [t.inds[-1] for t in tsrs]
    num_swap = 0
    while True:
        edge_map = dict()
        for jl,e in enumerate(pixs):
            if e not in edge_map:
                for jr in range(jl+1,len(pixs)):
                    if pixs[jr]==e:
                        d1 = jr-jl-1
                        d2 = len(pixs)-d1-2
                        edge_map[e] = (jl,jr),(d1,d2)
        edges = list(edge_map.keys())
        dists = [min(edge_map[e][-1]) for e in edges]
        e = edges[np.argsort(dists)[0]]
        (jl,jr),(d1,d2) = edge_map[e]
        d = min(d1,d2)
        if d1>d2:
            tsrs = tsrs[jl+1:]+tsrs[:jl+1]
            pixs = pixs[jl+1:]+pixs[:jl+1]
            jl,jr = d1,len(tsrs)-1
        arrs = [t.data for t in tsrs]
        mps = qtn.MatrixProductState(arrs,shape='lrp')
        if mps.L<=4:
            break
        assert mps.L==len(pixs)
#        print('edge=',e)
#        print(pixs)
#        print(mps)
#        print('len={},max_bond={}'.format(len(pixs),mps.max_bond()))
        pixs = pixs[:jl]+pixs[jl+1:jr]+pixs[jr+1:]
        if d==0:
            jt = jl-1 if jl>0 else jr+1
            tags = [mps.site_tag(j) for j in [jl,jr,jt]]
            mps.contract_tags(tags,which='any',inplace=True)
            tsrs = [mps[mps.site_tag(j)] for j in range(jl)]
            tsrs = tsrs+[mps[mps.site_tag(j)] for j in range(jr+1,mps.L)]
        elif d%2==1:
            jm = (jl+jr)//2
            num_swap += d-1
            mps.swap_site_to(i=jl,f=jm-1,inplace=True,**compress_opts)
            mps.swap_site_to(i=jr,f=jm+1,inplace=True,**compress_opts)
            old = mps._site_ind_id.format(jm-1)
            new = mps._site_ind_id.format(jm+1)
            tags = [mps.site_tag(j) for j in [jm-1,jm,jm+1]]
            mps[mps.site_tag(jm-1)].reindex_({old:new})
            mps.contract_tags(tags,which='any',inplace=True)
            tsrs = [mps[mps.site_tag(j)] for j in range(jm-1)]
            tsrs = tsrs+[mps[mps.site_tag(jm)]]
            tsrs = tsrs+[mps[mps.site_tag(j)] for j in range(jm+2,mps.L)]
        else:
            jml,jmr = jl+d//2,jr-d//2
            num_swap += d-2 
            mps.swap_site_to(i=jl,f=jml-1,inplace=True,**compress_opts)
            mps.swap_site_to(i=jr,f=jmr+1,inplace=True,**compress_opts)
            old = mps._site_ind_id.format(jml-1)
            new = mps._site_ind_id.format(jmr+1)
            mps[mps.site_tag(jml-1)].reindex_({old:new})
            mps.contract_between(mps.site_tag(jml),mps.site_tag(jml-1))
            mps.contract_between(mps.site_tag(jmr),mps.site_tag(jmr+1))
            mps.fuse_multibonds_()
            mps.left_canonize(jml-1)
            mps.right_canonize(jmr+1)
            mps.compress_between(mps.site_tag(jml),mps.site_tag(jmr),**compress_opts)
            tsrs = [mps[mps.site_tag(j)] for j in range(jml-1)]
            tsrs = tsrs+[mps[mps.site_tag(j)] for j in [jml,jmr]]
            tsrs = tsrs+[mps[mps.site_tag(j)] for j in range(jmr+2,mps.L)]
        tsrs = transpose(tsrs)
    assert(len(tsrs))==len(pixs) 
    for j,t in enumerate(tsrs):
        t.reindex_({t.inds[-1]:pixs[j]}) 
        tn.add_tensor(t)
    return tn,num_swap
def contract_serial(tn,N,alternate=True,**compress_opts):
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
        c1 = order[:i]
        c2 = [order[i]]
        tn,num_swap = contract_binary(c1,c2,tn,**compress_opts) 
        print('i={},num swap={},max_bond={}'.format(i,num_swap,tn.max_bond()))
        total_swap += num_swap
    print('total_swap=',total_swap)
    return tn.contract()
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
