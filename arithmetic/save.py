
def get_backbone_mps(T,n,tag,**compress_opts):
    norb = T.shape[T.inds.index('p')]
    oix = tuple(set(T.inds).difference({'p','q','k'}))[0]
    Ttag = tuple(T.tags)

    sCP2 = np.zeros((2,)*3)
    sCP2[0,0,0] = 1./norb
    sCP2[1,1,1] = 1.

    tn = qtn.TensorNetwork([])
    for i in range(n):
        pnew = 'p' if i==n-1 else f'{tag}{i},{i+1}'
        qnew = 'q' if i==0 else f'{tag}{i-1},{i}'
        Ti = T.reindex({'p':pnew,'q':qnew,'k':f'i{i}',oix:f'{oix}_{tag}{i}'})
        Ti.add_tag(f'{tag}{i}')
        tn.add_tensor(Ti)
    for i in range(1,n):
        lix = 'i0' if i==1 else f'i{i-1},{i}_'
        rix = 'k' if i==n-1 else f'i{i},{i+1}_'
        tn.add_tensor(qtn.Tensor(data=sCP2,inds=(lix,f'i{i}',rix),
                                 tags={'a',f'{tag}{i}'}))
    ix_map = {0:'q',n-1:'p'}
    ls = []
    # move q-leg
    for i in ix_map:
        Ti = tn[Ttag+(f'{tag}{i}',)]
        Tl,Tr = Ti.split(left_inds=None,right_inds=(f'i{i}',ix_map[i]),
                         absorb='right',**compress_opts)
        Ti.modify(data=Tl.data,inds=Tl.inds)

        ix_map[i] = Tr.inds[0]
        ls.append(Tr)
    for i in range(1,nstep-1):
        Ti = tn['a',f'{tag}{i}']
        blob = qtn.tensor_contract(ls[0],Ti)
        Tl,ls[0] = blob.split(left_inds=None,right_inds=('q',f'i{i},{i+1}_'),
                              absorb='right',bond_ind=f'i{i},{i+1}',**compress_opts)
        Ti.modify(data=Tl.data,inds=Tl.inds)
    Ti = tn_['a',f'{tag}{n-1}']
    blob = qtn.tensor_contract(*ls,Ti)
    Ti.modify(data=blob.data,inds=blob.inds)
    # compress involved tensors
    T = tn_[Ttag+(f'{tag}{n-1}',)]
    T.retag_({f'{tag}{n-1}':f'{tag}{n}'})
    tn_ = compress(tn_,tag,**compress_opts)
    T.retag_({f'{tag}{n}':f'{tag}{n-1}'})
    # seperate p,q,k
    Tl,Tr = Ti.split(left_inds=None,right_inds=('p','q','k'),
                     absorb='right',**compress_opts)
    Ti.modify(data=Tl.data,inds=Tl.inds)
    Tr.modify(tags={})
    tn.add_tensor(Tr)
    tn.reindex_({bix:f'i{i}' for i,bix in ix_map.items()})
    return tn
def get_cheb_coeff(fxn,order):
    N = order + 1
    c = []
    theta = [np.pi*(k-0.5)/N for k in range(1,N+1)]
    for j in range(order+1):
        v1 = np.array([fxn(np.cos(thetak)) for thetak in theta])
        v2 = np.array([np.cos(j*thetak) for thetak in theta])
        c.append(np.dot(v1,v2)*2./N)
    coeff = np.polynomial.chebyshev.cheb2poly(c)
    coeff[0] -= 0.5*c[0]
    print(coeff)
    print([1./scipy.math.factorial(i) for i in range(order+1)])
    return list(coeff)
def _get_full_propagator(tni,nstep):
    nf = tni.num_tensors-1
    norb = tni['T0'].shape[tni['T0'].inds.index('p')]
    ng = tni['T1'].shape[tni['T1'].inds.index('x0')]
    tn = qtn.TensorNetwork([])
    sCP2 = np.zeros((2,)*3)
    sCP2[0,0,0] = 1./norb
    sCP2[1,1,1] = 1.
    for i in range(1,nstep):
        lix = 'i0' if i==1 else f'i{i-1},{i}'
        rix = 'k' if i==nstep-1 else f'i{i},{i+1}'
        tn.add_tensor(qtn.Tensor(data=sCP2,inds=(lix,f'i{i}',rix),
                                 tags={'CP',f's{i}'}))
    vix = [qtn.rand_uuid() for i in range(nstep-1)]
    for i in range(nstep):
        tni_ = tni.copy()
        tni_.add_tag(f's{i}')
        pnew = 'p' if i==nstep-1 else vix[i]
        qnew = 'q' if i==0 else vix[i-1]
        index_map = {'p':pnew,'q':qnew,'k':f'i{i}'}
        index_map.update({f'x{g}':f's{i},x{g}' for g in range(nf)})
        tni_.reindex_(index_map)
        tn.add_tensor_network(tni_,check_collisions=True)
    return tn
def get_full_propagator(tni,nstep,**compress_opts):
    nf = tni.num_tensors-1
    norb = tni['T0'].shape[tni['T0'].inds.index('p')]
    ng = tni['T1'].shape[tni['T1'].inds.index('x0')]
    tn = qtn.TensorNetwork([])
    vix = [qtn.rand_uuid() for i in range(nstep-1)]
    sdel = np.eye(2)    
    sdel[0,0] *= 1./norb
    for i in range(nstep):
        tni_ = tni.copy()
        tni_.add_tag(f's{i}')
        pnew = 'p' if i==nstep-1 else vix[i]
        qnew = 'q' if i==0 else vix[i-1]
        index_map = {'p':pnew,'q':qnew,'k':f'i{i}'}
        index_map.update({f'x{g}':f's{i},x{g}' for g in range(nf)})
        tni_.reindex_(index_map)
        tn.add_tensor_network(tni_,check_collisions=True)
        # add 1/norb scaling
        if i>0: 
            tn['T0',f's{i}'].reindex_({f'i{i}':f'i{i}_'})
            tn.add_tensor(qtn.Tensor(data=sdel,inds=(f'i{i}_',f'i{i}'),
                                     tags={'T0',f's{i}'}))
            tn.contract_tags(('T0',f's{i}'),which='all',inplace=True)
            lix = 'i0' if i==1 else f'i{i-1},{i}'
            rix = 'k' if i==nstep-1 else f'i{i},{i+1}'
            tn.add_tensor(qtn.Tensor(data=CP2,inds=(lix,f'i{i}',rix),
                                     tags={'CP2',f's{i}'}))
    print(tn)
    Ti = tn['T0','s0']
    Tl,Tq = Ti.split(left_inds=None,right_inds=('q','i0'),absorb='right',**compress_opts)
    Ti.modify(data=Tl.data,inds=Tl.inds)
    xix = Tq.inds[0]
    for i in range(1,nstep-1):
        Ti = tn['CP2',f's{i}']
        blob = qtn.tensor_contract(Tq,Ti)
        Tl,Tq = blob.split(left_inds=None,right_inds=('q',f'i{i},{i+1}'),absorb='right',**compress_opts)
        Ti.modify(data=Tl.data,inds=Tl.inds)
    Ti = tn['T0',f's{nstep-1}']
    Tl,Tp = Ti.split(left_inds=None,right_inds=('p',f'i{nstep-1}'),absorb='right',**compress_opts)
#    Ti.modify(data=Tl.data,inds=Tl.inds)
#    yix = Tp.inds[0]
#    Ti = tn['CP2',f's{i}']
#    blob = qtn.tensor_contract()
def get_full_propagator(tni,nstep,**compress_opts):
    nf = tni.num_tensors-1
    norb = tni['T0'].shape[tni['T0'].inds.index('p')]
    ng = tni['T1'].shape[tni['T1'].inds.index('x0')]
    tn = qtn.TensorNetwork([])
    vix = [qtn.rand_uuid() for i in range(nstep-1)]
    sdel = np.eye(2)    
    sdel[0,0] *= 1./norb
    for i in range(nstep):
        tni_ = tni.copy()
        tni_.add_tag(f's{i}')
        pnew = 'p' if i==nstep-1 else vix[i]
        qnew = 'q' if i==0 else vix[i-1]
        index_map = {'p':pnew,'q':qnew,'k':f'i{i}'}
        index_map.update({f'x{g}':f's{i},x{g}' for g in range(nf)})
        tni_.reindex_(index_map)
        tn.add_tensor_network(tni_,check_collisions=True)
        # add 1/norb scaling
        if i>0: 
            tn['T0',f's{i}'].reindex_({f'i{i}':f'i{i}_'})
            tn.add_tensor(qtn.Tensor(data=sdel,inds=(f'i{i}_',f'i{i}'),
                                     tags={'T0',f's{i}'}))
            tn.contract_tags(('T0',f's{i}'),which='all',inplace=True)
            lix = 'i0' if i==1 else f'i{i-1},{i}'
            rix = 'k' if i==nstep-1 else f'i{i},{i+1}'
            tn.add_tensor(qtn.Tensor(data=CP2,inds=(lix,f'i{i}',rix)))
    print(tn)
    lixs = [('i0','q'),(f'i{nstep-1}','p')]
    tags = ['s0',f's{nstep-1}']
    U,S,V = dict(),dict(),dict() 
    for lix,tag in zip(lixs,tags):
        Tl,Tr = tn['T0',tag].split(lix,absorb='left',**compress_opts)
        tn['T0',tag].modify(data=Tr.data,inds=Tr.inds)
        Tl.transpose_(lix[0],lix[1],Tl.inds[-1])
        for i in [0,1]:
            data = Tl.data[i,:,:]
            U[tag,i],S[tag,i],V[tag,i] = np.linalg.svd(data,full_matrix=False)
        
    return tn
def get_cheb_coeff(fxn,order,a=-1.,b=1.,m=None):
    if m is None:
        m = order + 1
    r = [-np.cos((2.*k-1.)*np.pi/(2.*m)) for k in range(1,m+1)]
    x = [rk*(b-a)/2.+(b+a)/2. for rk in r]
    y = np.array([fxn(xk) for xk in x])
    c = []
    for i in range(order+1):
        vec = np.array([scipy.special.eval_chebyt(i,rk) for rk in r])
        c.append(np.dot(y,vec)/np.dot(vec,vec))
    c = np.polynomial.chebyshev.cheb2poly(c)
    A,B = 2./(b-a),-(b+a)/(b-a)
    coeff = [0. for i in range(order+1)]
    fac = [1]
    for i in range(1,order+1):
        fac.append(fac[-1]*i)
    for i in range(order+1):
        for j in range(i+1):
            coeff[j] += c[i]*A**j*B**(i-j)*fac[i]/(fac[j]*fac[i-j])
    return coeff
def get_B(v0,vg,tau,xs,max_bond=None,cutoff=1e-15,equalize_norms=True):
    norb,_,nf = vg.shape
    ng = len(xs)
    sqrt_tau = np.sqrt(tau)
    tn = qtn.TensorNetwork([])
    CP = np.zeros((norb,)*3)
    for i in range(norb):
        CP[i,i,i] = 1.

    data = np.ones((norb,norb,2))
    data[:,:,1] = -tau*v0
    inds = [qtn.rand_uuid() for i in range(3)]
    tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags={'T0','v0'}))
    xix = []
    for g in range(nf):
        data = np.ones((norb,norb,ng,2),dtype=vg.dtype)
        for i in range(ng):
            data[:,:,i,1] = sqrt_tau*xs[i]*vg[:,:,g]
        inds = [qtn.rand_uuid() for i in range(4)]
        xix.append(inds[2])
        tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags={f'T{g+1}',f'v{g+1}'}))

    pix = [qtn.rand_uuid() for i in range(nf)]
    for g in range(nf):
        rix = tn[f'v{g+1}'].inds[0] if g==nf-1 else pix[g+1]
        inds = pix[g],rix,tn[f'v{g}'].inds[0]
        tn.add_tensor(qtn.Tensor(data=CP,inds=inds,tags={f'T{g}',f'p{g}'}))
    qix = [qtn.rand_uuid() for i in range(nf)]
    for g in range(nf):
        rix = tn[f'v{g+1}'].inds[1] if g==nf-1 else qix[g+1]
        inds = qix[g],rix,tn[f'v{g}'].inds[1]
        tn.add_tensor(qtn.Tensor(data=CP,inds=inds,tags={f'T{g}',f'q{g}'}))
    iix = [qtn.rand_uuid() for i in range(nf)]
    for g in range(nf):
        rix = tn[f'v{g+1}'].inds[-1] if g==nf-1 else iix[g+1]
        inds = rix,tn[f'v{g}'].inds[-1],iix[g]
        tn.add_tensor(qtn.Tensor(data=ADD,inds=inds,tags={f'T{g}',f'a{g}'}))
    print(tn)

    for g in range(nf):
        tn.contract_tags(f'T{g}',which='any',inplace=True)
    tn.fuse_multibonds_()
    print(tn)

#    # canonize from right
#    for g in range(nf,0,-1):
#        tn.canonize_between(f'T{g-1}',f'T{g}',absorb='left')
#    # compress from right
#    for g in range(nf):
#        tn.compress_between(f'T{g}',f'T{g+1}',absorb='right',
#                            max_bond=max_bond,cutoff=cutoff)
#    # canonize from left
#    for g in range(nf):
#        tn.canonize_between(f'T{g}',f'T{g+1}',absorb='right')
#    # compress from right
#    for g in range(nf,0,-1):
#        tn.compress_between(f'T{g-1}',f'T{g}',absorb='left',
#                            max_bond=max_bond,cutoff=cutoff)
    return tn,pix[0],qix[0],iix[0],xix 
if __name__=='__main__':
    norb = 3
    t = 1.
    u = 1.

    h1 = np.zeros((norb,)*2)
    for i in range(norb):
        if i-1>0:
            h1[i,i-1] = -t
        if i+1<norb:
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
def get_tn1(x,y,z,beta,v_params,regularize=True):
    N,g = x.shape
    ls = list(itertools.product(range(g),repeat=3))
    tn = qtn.TensorNetwork([])
    coords = ['x','y','z']
    for i in range(N):
        for j in range(i+1,N):
            data = np.zeros((g,)*6)
            for (xi,yi,zi) in ls:
                ri = np.array([x[i,xi],y[i,yi],z[i,zi]])
                for (xj,yj,zj) in ls:
                    rj = np.array([x[j,xj],y[j,yj],z[j,zj]])
                    hij = morse(ri,rj,**v_params)
                    data[xi,yi,zi,xj,yj,zj] = np.exp(-beta*hij)
            if regularize:
                data_max = np.amax(data)
                data /= data_max
                expo = np.log10(data_max)
            else:
                expo = 0.0
            inds = [c+str(i) for c in coords]+[c+str(j) for c in coords] 
            tags = set(inds).union({'exp'})
            tn.add_tensor(qtn.Tensor(data=data.copy(),inds=inds,tags=tags))
            tn.exponent = tn.exponent + expo
    for i in range(N):
        for c in ['x','y','z']:
            inds = (c+str(i),)
            tags = set(inds).union({'w'})
            tn.add_tensor(qtn.Tensor(data=np.ones(g),inds=inds,tags=tags))
    expo = tn.exponent
    tn.exponent = 0.0
    return tn,expo
