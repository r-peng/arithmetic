
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
