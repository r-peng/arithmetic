
def get_mps(args,tr,data_map):
    i,fname = args
    N,g = tr.shape
    cp = np.zeros((g,)*3)
    for gi in range(g):
        cp[(gi,)*3] = 1.0
    # Head
    H_arr = []
    H_tag = []
    for j in range(i+1,N):
        data = data_map[i,j][0]
        if j>i+1:
            data = np.einsum('gk,efg->efk',data,cp)
        H_arr.append(data)
        H_tag.append(tag.format(j))
    # Tail
    T_arr = []
    T_tag = []
    for j in range(0,i):
        data = data_map[j,i][1]
        if j<i-1:
            data = np.einsum('gk,efg->efk',data,cp)
        T_arr.append(data)
        T_tag.append(tag.format(j))
    # merge
    if i==0:
        H_arr[-1] = np.einsum('efk,f->ek',H_arr[-1],tr[i,:])
    elif i==N-1:
        T_arr[0] = np.einsum('efk,e->fk',T_arr[0],tr[i,:])
    else:
        T_arr[0] = np.einsum('efk,e->efk',T_arr[0],tr[i,:])
    arrays = H_arr+T_arr
    tags = H_tag+T_tag

    mps = qtn.MatrixProductState(arrays=arrays,shape='lrp',tags='f{}'.format(i))
#    assert canonize in {None,'left','right'}
#    if canonize is not None:
#        if canonize=='left':
#            mps.left_canonize(normalize=False)
#        else:
#            mps.right_canonize(normalize=False) 
    for j in range(mps.L):
        mps[mps.site_tag(j)].add_tag('t{}'.format(tags[j]))

    write_tn_to_disc(mps,fname)
    return i,fname
def init_mps_map_parallel(tr,data_map,directory):
    N,g = tr.shape
    pool = multiprocessing.Pool(nworkers)
    ls = [(i,directory+'mps_{}'.format(i)) for i in range(N)]  
    fxn = functools.partial(get_mps,tr=tr,data_map=data_map)
    ls = pool.map(fxn,ls)
    mps_map = dict()
    for key,fname in ls:
        mps_map[key] = fname
    return mps_map
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
def merge2(tn,N):
    L = N-1
    for i in range(N):
        rtag = 'r{}'.format(i)
        for pair in [(0,1),(L-1,L-2)]:
            tag0,tag1 = 'I{}'.format(pair[0]),'I{}'.format(pair[1])
            t0,t1 = tn[rtag,tag0],tn[rtag,tag1]
            t0.add_tag('contract')
            t1.add_tag('contract')
            tn.contract_tags('contract',which='any',inplace=True)
            tsr = tn[rtag,tag0,tag1,'contract']
            tsr.modify(tags=(rtag,tag1))
        for j in range(1,L-1):
            old,new = 'I{}'.format(j),'I{}'.format(j-1)
            tsr = tn[rtag,old]
            tsr.retag_({old:new})
    return tn 
def merge(nodes,idxs,tn):
    rtags = ['r{}'.format(i) for i in rnodes]
    L = len(tn.select_tensors(rtag,which='any'))
    tag0,tag1 = 'I{}'.format(idxs),'I{}'.format(L-2)
    t0,t1 = tn[rtag+[tag0]],tn[rtag+[tag1]]
    t0.add_tag('contract')
    t1.add_tag('contract')
    tn.contract_tags('contract',which='any',inplace=True)
    t = tn['contract']
    t.modify(tags=rtag+[tag1])
    return tn
def merge_head(rnodes,tn): 
    rtags = ['r{}'.format(i) for i in rnodes]
    L = len(tn.select_tensors(rtag,which='any'))
    tag0,tag1 = 'I{}'.format(0),'I{}'.format(1)
    t0,t1 = tn[rtag+[tag0]],tn[rtag+[tag1]]
    t0.add_tag('contract')
    t1.add_tag('contract')
    tn.contract_tags('contract',which='any',inplace=True)
    t = tn['contract']
    t.modify(tags=rtag+[tag1])
    for j in range(1,L):
        old,new = 'I{}'.format(j),'I{}'.format(j-1+shift)
        t = tn[rtags+[old]]
        t.retag_({old:new})
    return tn,L-1
def get_mps(rtags,tn):
    L = len(tn.select_tensors(rtags,which='any'))
    arrs = [None for j in range(L)]
    for j in range(L):
        t = subtn['I{}'.format(j)]
        if j>0:
            tl = subtn['I{}'.format(j-1)]
            lix = t.bonds(tl)[0]
        if j<L-1:
            tr = subtn['I{}'.format(j+1)]
            rix = t.bonds(tr)[0]
        if j==0:
            vix = (rix,)
        elif j==L-1:
            vix = (lix,)
        else:
            vix = (lix,rix)
        pix = tuple(set(t.inds).difference(set(vix)))
        t.transpose_(vix+pix)
        arrs[j] = t.data.copy()
    mps = qtn.MatrixProductState(arrs,shape='lrp')
    return mps
def contract(left,right,egdes,tn,N):
    left_tag = ['r{}'.format(i) for i in left]
    right_tag = ['r{}'.format(i) for i in right]
    full_tag = left_tag+right_tag
    # merge I{L-1},I{L-2} of left
    L_left = len(tn.select_tensors(left_tag,which='any'))
    tag0,tag1 = 'I{}'.format(L_left-1),'I{}'.format(L_left-2)
    t0,t1 = tn[left_tag+[tag0]],tn[left_tag+[tag1]]
    t0.add_tag('contract')
    t1.add_tag('contract')
    tn.contract_tags('contract',which='any',inplace=True)
    for j in range(L_left-1):
        tag = 'I{}'.format(j)
        t = tn[left_tag+[tag]]
        t.modify(tags=full_tag+[tag])
    # merge I0,I1 of right
    L_right = len(tn.select_tensors(right_tag,which='any'))
    tag0,tag1 = 'I{}'.format(0),'I{}'.format(1)
    t0,t1 = tn[right_tag+[tag0]],tn[right_tag+[tag1]]
    t0.add_tag('contract')
    t1.add_tag('contract')
    tn.contract_tags('contract',which='any',inplace=True)
    for j in range(1,L_right):
        old,new = 'I{}'.format(j),'I{}'.format(j-1+L_left)
        t = tn[left_tag+[old]]
        t.modify(tags=full_tag+[new])

    # retag mps(left+blob) 
    L = len(tn.select_tensors(full_tag,which='any')) 
    assert L==L_left+L_right-2
    for edge in egdes:
        # find the idx of the tsr pair
        sites = []
        for j in range(L):
            tags = full_tag+['I{}'.format(j)]
            if edge in t.inds:
                sites.append(j)
        assert len(sites)==2
        mps = get_mps(full_tags,tn)
    return tn
def contract(left,right,edges,tn,**compress_opts):
    # merge tail of left
    left_tag = ['r{}'.format(i) for i in left]
    left_L = len(tn.select_tensors(left_tag,which='any'))
    left_tsr = [None for j in range(left_L)]
    for j in range(left_L):
        tags = left_tag+['I{}'.format(j)]
        tid = tuple(tn._get_tids_from_tags(tags,which='all'))[0]
        left_tsr[j] = tn._pop_tensor(tid)
#    for tsr in left_tsr:
#        print(tsr)
#    exit()
    left_tsr[-2] = qtn.tensor_contract(left_tsr[-2],left_tsr[-1])
    left_tsr.pop(-1)
#    for tsr in left_tsr:
#        print(tsr)
#    exit()
    # merge head of right
    right_tag = ['r{}'.format(i) for i in right]
    right_L = len(tn.select_tensors(right_tag,which='any'))
    right_tsr = [None for j in range(right_L)]
    for j in range(right_L):
        tags = right_tag+['I{}'.format(j)]
        tid = tuple(tn._get_tids_from_tags(tags,which='all'))[0]
        right_tsr[j] = tn._pop_tensor(tid)
#    for tsr in right_tsr:
#        print(tsr)
#    exit()
    right_tsr[1] = qtn.tensor_contract(right_tsr[1],right_tsr[0])
    right_tsr.pop(0)
#    for tsr in right_tsr:
#        print(tsr)
#    exit()
#    print(tn)
    # make full mps
    L = left_L+right_L-2
    tsrs = left_tsr+right_tsr
    pixs = [None for j in range(L)]
    arrs = [None for j in range(L)]
#    for tsr in tsrs:
#        print(tsr)
    for j in range(L):
        t = tsrs[j] 
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
#        print(t)
#        print(output_inds,vix,pix)
        t.transpose_(*output_inds)
        arrs[j] = t.data.copy()
        pixs[j] = pix[0]
    mps = qtn.MatrixProductState(arrs,shape='lrp',tags=left_tag+right_tag)
    for e in edges: # edges in addition to the left-right edge
        assert mps.L==len(pixs)
        # get site 
        site = []
        for j in range(L):
            if e==pixs[j]:
                site.append(j)
        assert len(site)==2
        jl,jr = site
        # mps swap & contract
        mps.swap_site_to(i=jr,f=jl+1,inplace=True,**compress_opts)
        idxs = [jl,jl+1,jl+2]
        for j in idxs:
            mps[mps.site_tag(j)].add_tag('contract')
        mps.contract_tags('contract',which='any',inplace=True)
        drop_tags = ['contract']+[mps.site_tag(j) for j in idxs[:2]]
        mps['contract'].drop_tags(drop_tags)
        for j in range(jl+2,L):
            old_idx = mps._site_ind_id.format(j)
            new_idx = mps._site_ind_id.format(j-2)
            old_tag = mps.site_tag(j)
            new_tag = mps.site_tag(j-2)
            t = mps[old_tag]
            t.retag_({old_tag:new_tag}) 
            t.reindex_({old_idx:new_idx})
        pixs = pixs[:jl]+pixs[jl+1:jr]+pixs[jr+1:]
        mps.L = mps.num_tensors
    # add to tn
    print(pixs)
    for j in range(mps.L):
        tsr = mps[mps.site_tag(j)]
        tsr.reindex_({mps._site_ind_id.format(j):pixs[j]})
        tn.add_tensor(tsr)
    print(tn)
    return tn
def contract(left,right,edges,tn,**compress_opts):
    # merge tail of left
    left_tag = ['r{}'.format(i) for i in left]
    left_L = len(tn.select_tensors(left_tag,which='any'))
    left_tsr = [None for j in range(left_L)]
    for j in range(left_L):
        tags = left_tag+['I{}'.format(j)]
        tid = tuple(tn._get_tids_from_tags(tags,which='all'))[0]
        left_tsr[j] = tn._pop_tensor(tid)
    left_tsr[-2] = qtn.tensor_contract(left_tsr[-2],left_tsr[-1])
    left_tsr.pop(-1)
    # merge head of right
    right_tag = ['r{}'.format(i) for i in right]
    right_L = len(tn.select_tensors(right_tag,which='any'))
    right_tsr = [None for j in range(right_L)]
    for j in range(right_L):
        tags = right_tag+['I{}'.format(j)]
        tid = tuple(tn._get_tids_from_tags(tags,which='all'))[0]
        right_tsr[j] = tn._pop_tensor(tid)
    right_tsr[1] = qtn.tensor_contract(right_tsr[1],right_tsr[0])
    right_tsr.pop(0)
    # make full mps
    full_tag = left_tag+right_tag
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
        print(e)
        print(pixs)
        print(mps)
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
        idxs = [jl,jl+1,jl+2]
#        print(site)
#        print(pixs)
#        print(mps)
        for j in idxs:
            mps[mps.site_tag(j)].add_tag('contract')
        old,new = mps._site_ind_id.format(jl+1),mps._site_ind_id.format(jl)
        mps[mps.site_tag(jl+1)].reindex_({old:new})
        mps.contract_tags('contract',which='any',inplace=True)
#        print(mps)
        tsrs = [mps[mps.site_tag(j)] 
                for j in tuple(range(jl))+tuple(range(jl+2,mps.L))]
        tsrs = transpose(tsrs)
        arrs = [t.data for t in tsrs]
        mps = qtn.MatrixProductState(arrs,shape='lrp',tags=full_tag)
    # add to tn
#    print(pixs)
    for j in range(mps.L):
        tsr = mps[mps.site_tag(j)]
        tsr.reindex_({mps._site_ind_id.format(j):pixs[j]})
        tn.add_tensor(tsr)
    print(tn)
    return tn
