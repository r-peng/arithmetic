
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
def next_add_inds(ks):
    mod = len(ks)
    new = []
    for k in ks:
        kl,kr = [],[]
        for i,j in k:
            i_,j_ = i+mod,j+mod
            kl.append((i_,j))
            kr.append((i,j_))
        new.append(kl+kr)
    return ks+new
def get_add_inds(max_iter):
    ks = [[(0,0)],[(1,0),(0,1)]]
    for i in range(max_iter):
        ks = next_add_inds(ks)
    return ks
def get_projector_exp(k,max_bond=None):
    dim = 2**k
    if max_bond is None:
        out = np.zeros((dim,2,dim*2))
        out[:,0,:dim] = np.eye(dim)
        out[:,1,dim:] = np.eye(dim)
    else:
        dim1 = 2 if k==1 else max_bond
        out = np.zeros((dim1,2,max_bond))
        if max_bond<=dim:
            out[:,0,:] = np.eye(max_bond)
        elif max_bond<=2*dim:
            out[:dim,0,:dim] = np.eye(dim)
            remain = max_bond-dim
            out[:remain,1,dim:] = np.eye(remain)
        else: 
            out[:dim,0,:dim] = np.eye(dim)
            out[:dim,1,dim:2*dim] = np.eye(dim)
    return out
def get_add_exp(n,max_bond=None):
    ks = get_add_inds(n-1)
    max_bond = len(ks) if max_bond is None else max_bond
    out = np.zeros((max_bond,)*3)
    for k in range(max_bond):
        for i,j in ks[k]:
            out[i,j,k] = 1
    return out
def get_terminal_exp(n,max_bond=None):
    max_bond = 2**n if max_bond is None else max_bond
    out = np.zeros(max_bond)
    out[-1] = 1
    return out

def get_projector_lin(k,max_bond=None):
    if max_bond is None:
        out = np.zeros((k+1,2,k+2))
    else: 
        dim1 = 2 if k==1 else max_bond
        out = np.zeros((dim1,2,max_bond))
    out[:k+1,0,:k+1] = np.eye(k+1)
    out[k,1,k+1] = 1.0
    return out
def get_add_lin(n,max_bond=None):
    max_bond = n+1 if max_bond is None else max_bond
    fac = [math.factorial(k) for k in range(n+1)]
    out = np.zeros((max_bond,)*3)
    for k in range(n+1):
        for i in range(k+1):
            j = k-i
            out[i,j,k] = fac[k]/(fac[i]*fac[j])
    return out
def get_terminal_lin(n,max_bond=None):
    max_bond = n+1 if max_bond is None else max_bond
    out = np.zeros(max_bond)
    out[n] = 1
    return out

def get_projector_uniform(k,max_bond):
    dim1 = 2 if k==1 else max_bond
    return np.zeros((dim1,2,max_bond))
def get_add_uniform(max_bond):
    return np.zeros((max_bond,)*3)
def get_terminal_uniform(max_bond):
    return np.zeros(max_bond)

scale = 1e-3
scale = 0.0
def get_projector(k,max_bond=None,scheme='lin'):
    if scheme=='lin':
        out = get_projector_lin(k,max_bond)
    if scheme=='exp':
        out = get_projector_exp(k,max_bond)
    if scheme=='uniform':
        out = get_projector_uniform(k,max_bond)
    out += np.ones_like(out)*scale
    return out
def get_add(n,max_bond=None,scheme='lin'):
    if scheme=='lin':
        out = get_add_lin(n,max_bond)
    if scheme=='exp':
        out = get_add_exp(n,max_bond)   
    if scheme=='uniform':
        out = get_add_uniform(max_bond)   
    out += np.ones_like(out)*scale
    return out
def get_terminal(n,max_bond=None,scheme='lin'):
    if scheme=='lin':
        out = get_terminal_lin(n,max_bond)
    if scheme=='exp':
        out = get_terminal_exp(n,max_bond)   
    if scheme=='uniform':
        out = get_terminal_uniform(max_bond)   
    out += np.ones_like(out)*scale
    return out
def get_permutation(col_inds):
    # col_list = [j1,...,jn]
    n = len(col_inds)
    L0, L1 = [],[]
    for k in range(n):
        if col_inds[k]==0:
            L0.append(k)
        else:
            L1.append(k)
#    sigma = {}
#    for i in range(len(L0)):
#        sigma.update({L0[i]:i})
#    for i in range(len(L1)):
#        sigma.update({L1[i]:i+len(L0)})
#    sigma_ = {}
    sigma = L0+L1
    inds = []
    for i in range(2**n):
        init_inds = to_bin(i)
        target_inds = [sigma[j] for j in init_inds]
        inds.append(to_int(target_inds))
def to_int(bin_inds):
    string = ['{}'.format(j) for j in bin_inds]
    return int(str(string),'2')
def to_bin(int_ind):
    string = format(int_ind,'b')
    return [int(j) for j in string] 
 
if __name__=='__main__': 
    import functools
    n = 5
    N = 7
    def poly(qs):
        out = 1.0
        for k in range(n):
            pk = 0.0
            for i in range(N):
                pk += qs[k,i]
            out *= pk
        return out
    def tn(qs,scheme):
        tn = qtn.TensorNetwork([])
        if scheme=='lin':
            P = functools.partial(get_projector_lin,max_bond=None)
            ADD = get_add_lin(n)
            v = get_terminal_lin(n)
        if scheme=='exp':
            P = functools.partial(get_projector_exp,max_bond=None)
            ADD = get_add_exp(n)
            v = get_terminal_exp(n)
        for i in range(N):
            for k in range(n):
                data = np.array([1,qs[k,i]])
                inds = ('q{},{},'.format(i,k),)
                tn.add_tensor(qtn.Tensor(data=data,inds=inds))
            for k in range(1,n):
                i1 = 'q{},{},'.format(i,k-1) if k==1 else 'p{},{},'.format(i,k-1)
                inds = i1,'q{},{},'.format(i,k),'p{},{},'.format(i,k)
                tn.add_tensor(qtn.Tensor(data=P(k),inds=inds))
        for i in range(1,N):
            i1 = 'p{},{},'.format(i-1,n-1) if i==1 else '+{},'.format(i-1)
            inds = i1,'p{},{},'.format(i,n-1),'+{},'.format(i)
            tn.add_tensor(qtn.Tensor(data=ADD,inds=inds))
        tn.add_tensor(qtn.Tensor(data=v,inds=('+{},'.format(N-1),)))
        return tn.contract(output_inds=[])
    print('#### exp ####')
    qs = np.random.rand(n,N)
    print(poly(qs)-tn(qs,scheme='exp'))
    print('#### lin ####')
    qs = np.zeros((n,N))
    qs[0,:] = np.random.rand(N)
    for i in range(1,n):
        qs[i,:] = qs[0,:]
    print(poly(qs)-tn(qs,scheme='lin'))

