
#def block_compressed(xs,coeff,ins=None,contract_left=[],contract_right=[],
#                       cutoff=0.0,max_bond=None):
#    # coeff: m*n*N tensor
#    m,n,N = coeff.shape
#    m -= 1
#    xs = [xpowers(x,m) for x in xs]
#    data = make_ins(xs,ins)
#    def make_c(cki):
#        c = np.zeros((m+1,2))
#        c[0,0] = 1
#        c[:,1] = cki.copy()
#        return c
#    ls = []
#    ADD = np.zeros((2,)*3)
#    ADD[1,0,1] = ADD[0,1,1] = ADD[0,0,0] = 1
#    for i in range(N):
#        q = np.einsum('dm,mi->di',xs[i],make_c(coeff[:,0,i]))
#        fi = np.einsum('di,ijk->djk',q,ADD)
#        for k in range(1,n):
#            q = np.einsum('dm,mi->di',xs[i],make_c(coeff[:,k,i]))
#            q = np.einsum('di,ijk->djk',q,ADD)
#            fi = np.einsum('d...,djk->d...jk',fi,q)
#        ls.append(np.einsum('d...,d->...',fi,data[i]))
#    tn = qtn.TensorNetwork([])
#    linds = [qtc.rand_uuid() for i in range(n)] 
#    rinds = [qtn.rand_uuid() for i in range(n)]
#    for i in range(N):
#        new = []
#        for k in range(n):
#            lind = linds[k] if i==0 else old[2*k+1]
#            rind = rinds[k] if i==N-1 else qtc.rand_uuid()
#            new += [lind,rind]
#        tn.add_tensor(qtn.Tensor(data=ls[i],inds=new))
#        old = new.copy()
#    linds_new = linds.copy()
#    for i in contract_left:
#        tn.add_tensor(qtn.Tensor(data=np.array([1,0]),inds=[linds[i]]))
#        linds_new.remove(linds[i]) 
#    rinds_new = rinds.copy()
#    for i in contract_right:
#        tn.add_tensor(qtn.Tensor(data=np.array([0,1]),inds=[rinds[i]]))
#        rinds_new.remove(rinds[i]) 
#    output_inds = linds_new+rinds_new
#    opt = ctg.HyperOptimizer()
#    tn = tn.contract_compressed(output_inds=output_inds,optimize=opt,cutoff=cutoff,max_bond=max_bond)
#    out = tn.contract(output_inds=output_inds,optimize=opt)
#    return out 
def general_compressed(xs,coeff,ins=None,cutoff=0.0,max_bond=None):
    m,n,N = coeff.shape
    contract_left = list(range(n))
    contract_right = list(range(n))
    tn, linds, rinds = gen2D(xs,coeff,ins=ins,
                             contract_left=contract_left,
                             contract_right=contract_right)
    out = compress(tn,N,n,linds,rinds,cutoff=cutoff,max_bond=max_bond)
    return out.data
def compressed_tn(xs,coeff,var_tensors,coords=None):
    # coeff: m*n*N tensor
    m,n,N = coeff.shape
    m -= 1
    if coords is not None: 
        data = get_input_vectors(xs,coords)
    P,ADD,v = var_tensors
    size = N*n

    tn = qtn.TensorNetwork([])
    for i in range(N):
        x = get_powers(xs[i],m)
        d = x.shape[0]
        CP = np.zeros((d,)*3)
        for p in range(d):
            CP[p,p,p] = 1
        for k in range(n):
            j = 'i{},{},'.format(i,k)
            t = 'd{},{},'.format(i,k)
            b = 'd{},{},'.format(i,k+1)
            tags = {'q','{},{},'.format(i,k)}
            q = get_poly1(x,coeff[:,k,i])
            if k==n-1:
                inds = t,j
            elif k==0 and coords is not None:
                q = np.einsum('di,d->di',q,data[i])
                inds = b,j
            else:
                q = np.einsum('di,def->efi',q,CP)
                inds = t,b,j
            tn.add_tensor(qtn.Tensor(data=q,inds=inds,tags=tags),tid=i*n+k)
        for k in range(1,n):
            i1 = 'i{},{},'.format(i,k-1) if k==1 else 'p{},{},'.format(i,k-1)
            inds = i1,'i{},{},'.format(i,k),'p{},{},'.format(i,k)
            tags = {'fit','p','{},{},'.format(i,k)}
            tn.add_tensor(qtn.Tensor(data=P(k),inds=inds,tags=tags),tid=size+i*n+k)
    for i in range(1,N):
        i1 = 'p{},{},'.format(i-1,n-1) if i==1 else '+{},'.format(i-1)
        inds = i1,'p{},{},'.format(i,n-1),'+{},'.format(i)
        tags = {'fit','+','{},'.format(i)}
        tn.add_tensor(qtn.Tensor(data=ADD,inds=inds,tags=tags),tid=2*size+i)
    inds = ['+{},'.format(N-1)]
    tn.add_tensor(qtn.Tensor(data=v,inds=inds,tags='fit'),tid=2*size+N)
    return tn
