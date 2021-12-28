import numpy as np
import quimb.tensor as qtn
import math
ADD = np.zeros((2,)*3)
ADD[0,0,0] = ADD[1,0,1] = ADD[0,1,1] = 1.0
CP2 = np.zeros((2,)*3)
CP2[0,0,0] = CP2[1,1,1] = 1.0
def get_energy(A,xs,contract=False,**split_opts):
    N,d = xs.shape 
    # N=number of variable
    # d=physical dimension
    assert A.shape==(N,N)
    Xs = [np.ones((d,2)) for i in range(N)]
    for i in range(N):
        Xs[i][:,1] = xs[i,:]

    tn = qtn.TensorNetwork([])
    # diagonal terms
    for i in range(N):
        data = np.multiply(Xs[i],Xs[i])
        data[:,1] *= A[i,i]
        if i==0:
            rix = qtn.rand_uuid()
            inds = qtn.rand_uuid(),rix # p,r
        else:
            lix,rix = rix,qtn.rand_uuid() # p,l,r
            data = np.einsum('di,ijk->djk',data,ADD)
            inds = qtn.rand_uuid(),lix,rix
        tags = 'x{}'.format(i)
        tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags=tags),tid=i,virtual=True)

    # off_diagonal terms, NN
    CPd = np.zeros((d,)*3)
    for i in range(d):
        CPd[i,i,i] = 1.0
    for i in range(N-1):
        j = i+1
#        print(i,j)
        Ti,Tj = tn.tensor_map[i],tn.tensor_map[j]

        data = np.einsum('di,ei,i->dei',Xs[i],Xs[j],np.array([1.0,A[i,j]+A[j,i]]))
        inds = [qtn.rand_uuid() for i in data.shape] # pi,pj,oi
        T = qtn.Tensor(data=data,inds=inds)

        inds_i = Ti.inds[0],T.inds[0],qtn.rand_uuid() # pi
        inds_j = Tj.inds[0],T.inds[1],qtn.rand_uuid() # pj
        CPi = qtn.Tensor(data=CPd,inds=inds_i)
        CPj = qtn.Tensor(data=CPd,inds=inds_j)

        tns = Ti,Tj,T,CPi,CPj
        lix = [CPi.inds[-1],T.inds[-1]]
        if i>0:
            lix.append(Ti.inds[-2])
        rix = [CPj.inds[-1],Tj.inds[-1]]
        Tij = qtn.tensor_contract(*tns,output_inds=lix+rix)
        Ti_,Tj_ = qtn.tensor_split(Tij,lix,right_inds=rix,get='tensors',**split_opts)
        
        Ti.modify(data=Ti_.data,inds=Ti_.inds)
        Tj.modify(data=Tj_.data,inds=Tj_.inds)
        Tj.transpose_(*(Tj.inds[ax] for ax in [1,0,2]))
    i,j = N-1,0
#    print(i,j)
    Ti,Tj = tn.tensor_map[i],tn.tensor_map[j]
    data = np.einsum('di,ijk->djk',Xs[i],CP2)
    data = np.einsum('dli,djr,ijk->dklr',Ti.data,data,ADD)
    Ti.modify(data=data,inds=[Ti.inds[ax] for ax in [0,2,1]]+[qtn.rand_uuid()])

    data = Xs[j].copy()
    data[:,1] *= A[i,j]+A[j,i]
    data = np.einsum('dkr,dl->dklr',Tj.data,data)
    Tj.modify(data=data,inds=Tj.inds[:2]+(Ti.inds[-1],Tj.inds[-1]))
#    return tn

    # off-diagonal terms, xi<xj
    # tid(xi) = i
#    print(tn)
    for i in range(N):
        data_i = np.einsum('di,ijk->djk',Xs[i],CP2) 
        Ti = tn.tensor_map[i]
        j_range = range(i+2,N-1) if i==0 else range(i+2,N)
        for j in j_range:
#            print(i,j)

            inds = [qtn.rand_uuid() for i in data_i.shape]
            ti = qtn.Tensor(data=data_i,inds=inds)
            inds = Ti.inds[1],ti.inds[1],qtn.rand_uuid()
            ai = qtn.Tensor(data=ADD,inds=inds)
            inds = Ti.inds[0],ti.inds[0],qtn.rand_uuid()
            CPi = qtn.Tensor(data=CPd,inds=inds)
            tns = CPi,Ti,ti,ai
            lix = CPi.inds[-1],ai.inds[-1],Ti.inds[2]
            rix = ti.inds[2],Ti.inds[3] 
            T = qtn.tensor_contract(*tns,output_inds=lix+rix)
            Li,Ri = qtn.tensor_split(T,lix,right_inds=rix,get='tensors',
                                     rtags='A{},{}'.format(i,j),**split_opts) 
            Ti.modify(data=Li.data,inds=Li.inds)
            tn.add_tensor(Ri,virtual=True)

            Tj = tn.tensor_map[j]
            data_j = Xs[j].copy()
            data_j[:,1] *= A[i,j]+A[j,i]
            inds = qtn.rand_uuid(),Ri.inds[1]
            tj = qtn.Tensor(data=data_j,inds=inds)
            inds = Tj.inds[0],tj.inds[0],qtn.rand_uuid()
            CPj = qtn.Tensor(data=CPd,inds=inds)
            tns = CPj,Tj,tj
            lix = CPj.inds[-1],Tj.inds[1],Tj.inds[2]
            rix = tj.inds[1],Tj.inds[3]
            T = qtn.tensor_contract(*tns,output_inds=lix+rix)
            Lj,Rj = qtn.tensor_split(T,lix,right_inds=rix,get='tensors',
                                     rtags='A{},{}'.format(i,j),**split_opts) 
            Tj.modify(data=Lj.data,inds=Lj.inds)
            tn.add_tensor(Rj,virtual=True)
#    print(tn)

    inds = [None,]*3
    for i in range(1,N):
        i1 = tn.tensor_map[i-1].inds[1] if i==1 else inds[-1]
        inds = i1,tn.tensor_map[i].inds[1],qtn.rand_uuid()
        tn.add_tensor(qtn.Tensor(data=ADD,inds=inds))

    if contract:
        inds = [tn.tensor_map[tn.num_tensors-1].inds[-1]]
        data = np.array([0.0,1.0])
        tn.add_tensor(qtn.Tensor(data=data,inds=inds))
        for i in range(N):
            inds = [tn.tensor_map[i].inds[0]]
            data = np.ones(d)/d
            tn.add_tensor(qtn.Tensor(data=data,inds=inds))
    return tn
def get_pol(A,xs,coeff,**split_opts):
    # coeff[i] = ai
    n = len(coeff)-1
    N,d = xs.shape 
    E = get_energy(A,xs,**split_opts)
    tr = np.ones(d)/d
    tmp = np.einsum('ijk,klm->ijlm',CP2,ADD)
    CPd = np.zeros((d,)*3)
    for i in range(d):
        CPd[i,i,i] = 1.0

    tn = qtn.TensorNetwork([])
    for i in range(n-1,-1,-1):
        Ei = E.copy()
        T = Ei.tensor_map[Ei.num_tensors-1]
        data = np.einsum('ijlm,...i,l->...jm',tmp,T.data,np.array([1.0,coeff[i]]))
        if i==n-1:
            data = np.einsum('...jm,j->...m',data,np.array([1.0,coeff[n]]))
            ainds = T.inds[:2]+(qtn.rand_uuid(),)
        elif i==0:
            data = np.einsum('...jm,m->...j',data,np.array([0.0,1.0]))
            ainds = T.inds[:2]+(ainds[-1],)
        else:
            ainds = T.inds[:2]+(ainds[-1],qtn.rand_uuid())
        T.modify(data=data,inds=ainds)

        if i==n-1: 
            xids = [qtn.rand_uuid() for j in range(N)]
            index_map = {Ei.tensor_map[j].inds[0]:xids[j] for j in range(N)}
            Ei.reindex_(index_map)
        elif i==0:
            xids_ = xids.copy()
            for j in range(N):
                T = Ei.tensor_map[j]
                data = np.einsum('i...,ijk,k->j...',T.data,CPd,tr)
                inds = (xids_[j],)+T.inds[1:]
                T.modify(data=data,inds=inds)
        else:
            xids_ = xids.copy()
            xids = [qtn.rand_uuid() for j in range(N)]
            for j in range(N):
                T = Ei.tensor_map[j]
                data = np.einsum('i...,ijk->jk...',T.data,CPd)
                inds = (xids_[j],xids[j])+T.inds[1:]
                T.modify(data=data,inds=inds)
        tn.add_tensor_network(Ei,virtual=True)
    return tn
def _get_exp_2d(A,xs,**split_opts): # get a 2d tn
    N,d = xs.shape 
    tr = np.ones(d)/d
    # N=number of variable
    # d=physical dimension
    assert A.shape==(N,N)

    tn = qtn.TensorNetwork([])
    # diagonal terms
    for i in range(N):
        data = np.ones((d,2))
        for j in range(d):
            data[j,1] = np.exp(A[i,i]*xs[i,j]**2)
        if i==0:
            rix = qtn.rand_uuid()
            inds = qtn.rand_uuid(),rix # p,r
        else:
            lix,rix = rix,qtn.rand_uuid() # p,l,r
            data = np.einsum('di,ijk->djk',data,CP2)
            inds = qtn.rand_uuid(),lix,rix
        tags = 'x{}'.format(i)
        tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags=tags),tid=i,virtual=True)
#    print(tn)

    # off_diagonal terms, NN
    CPd = np.zeros((d,)*3)
    for i in range(d):
        CPd[i,i,i] = 1.0
    for i in range(N-1):
        j = i+1
        Ti,Tj = tn.tensor_map[i],tn.tensor_map[j]

        data = np.ones((d,d,2))
        for k in range(d):
            for l in range(d):
                data[k,l,1] = np.exp((A[i,j]+A[j,i])*xs[i,k]*xs[j,l])
        inds = [qtn.rand_uuid() for i in data.shape] # pi,pj,oi
        T = qtn.Tensor(data=data,inds=inds)

        inds_i = Ti.inds[0],T.inds[0],qtn.rand_uuid() # pi
        inds_j = Tj.inds[0],T.inds[1],qtn.rand_uuid() # pj
        CPi = qtn.Tensor(data=CPd,inds=inds_i)
        CPj = qtn.Tensor(data=CPd,inds=inds_j)

        tns = Ti,Tj,T,CPi,CPj
        lix = [CPi.inds[-1],T.inds[-1]]
        if i>0:
            lix.append(Ti.inds[-2])
        rix = [CPj.inds[-1],Tj.inds[-1]]
        Tij = qtn.tensor_contract(*tns,output_inds=lix+rix)
        Ti_,Tj_ = qtn.tensor_split(Tij,lix,right_inds=rix,get='tensors',**split_opts)
        
        Ti.modify(data=Ti_.data,inds=Ti_.inds)
        Tj.modify(data=Tj_.data,inds=Tj_.inds)
        Tj.transpose_(*(Tj.inds[ax] for ax in [1,0,2]))
    i,j = N-1,0
    Ti,Tj = tn.tensor_map[i],tn.tensor_map[j]
    data = np.ones((d,2,d))
    for k in range(d):
        for l in range(d):
            data[k,1,l] = np.exp((A[i,j]+A[j,i])*xs[i,k]*xs[j,l])
    inds = [qtn.rand_uuid() for ax in data.shape]
    T = qtn.Tensor(data=data,inds=inds)
    L,R = qtn.tensor_split(T,T.inds[:2],right_inds=T.inds[2:],get='arrays',**split_opts)
    data = np.einsum('dli,djr,ijk->dklr',Ti.data,L,CP2)
    Ti.modify(data=data,inds=[Ti.inds[ax] for ax in [0,2,1]]+[qtn.rand_uuid()])
    data = np.einsum('dkr,dl->dklr',Tj.data,R.T)
    Tj.modify(data=data,inds=Tj.inds[:2]+(Ti.inds[-1],Tj.inds[-1]))
    print(tn)
    
    # off-diagonal terms, xi<xj
    # tid(xi) = i
    for i in range(N):
        Ti = tn.tensor_map[i]
        j_range = range(i+2,N-1) if i==0 else range(i+2,N)
        for j in j_range:
            data = np.ones((d,2,d))
            for k in range(d):
                for l in range(d):
                    data[k,1,l] = np.exp((A[i,j]+A[j,i])*xs[i,k]*xs[j,l])
            inds = [qtn.rand_uuid() for ax in data.shape]
            T = qtn.Tensor(data=data,inds=inds)
            data_i,data_j = qtn.tensor_split(T,T.inds[:2],right_inds=T.inds[2:],
                            get='arrays',**split_opts)
            data_j = data_j.T

            inds = [qtn.rand_uuid() for i in data_i.shape]
            ti = qtn.Tensor(data=data_i,inds=inds)
            inds = Ti.inds[1],ti.inds[1],qtn.rand_uuid()
            ai = qtn.Tensor(data=CP2,inds=inds)
            inds = Ti.inds[0],ti.inds[0],qtn.rand_uuid()
            CPi = qtn.Tensor(data=CPd,inds=inds)
            tns = CPi,Ti,ti,ai
            lix = CPi.inds[-1],ai.inds[-1],Ti.inds[2]
            rix = ti.inds[2],Ti.inds[3] 
            T = qtn.tensor_contract(*tns,output_inds=lix+rix)
            Li,Ri = qtn.tensor_split(T,lix,right_inds=rix,get='tensors',
                                     rtags='A{},{}'.format(i,j),**split_opts) 
            Ti.modify(data=Li.data,inds=Li.inds)
            tn.add_tensor(Ri,virtual=True)

            Tj = tn.tensor_map[j]
            inds = qtn.rand_uuid(),Ri.inds[1]
            tj = qtn.Tensor(data=data_j,inds=inds)
            inds = Tj.inds[0],tj.inds[0],qtn.rand_uuid()
            CPj = qtn.Tensor(data=CPd,inds=inds)
            tns = CPj,Tj,tj
            lix = CPj.inds[-1],Tj.inds[1],Tj.inds[2]
            rix = tj.inds[1],Tj.inds[3]
            T = qtn.tensor_contract(*tns,output_inds=lix+rix)
            Lj,Rj = qtn.tensor_split(T,lix,right_inds=rix,get='tensors',
                                     rtags='A{},{}'.format(i,j),**split_opts) 
            Tj.modify(data=Lj.data,inds=Lj.inds)
            tn.add_tensor(Rj,virtual=True)

    inds = [None,]*3
    for i in range(1,N):
        i1 = tn.tensor_map[i-1].inds[1] if i==1 else inds[-1]
        inds = i1,tn.tensor_map[i].inds[1],qtn.rand_uuid()
        tn.add_tensor(qtn.Tensor(data=CP2,inds=inds,tags=tags))

    oid = tn.tensor_map[tn.num_tensors-1].inds[-1]
    tn.add_tensor(qtn.Tensor(data=np.array([0.0,1.0]),inds=[oid]))
    for i in range(N):
        tn.add_tensor(qtn.Tensor(data=tr,inds=[tn.tensor_map[i].inds[0]]))
    return tn
def get_exp_2d(A,xs,**split_opts): # get a 2d tn
    N,d = xs.shape 
    tr = np.ones(d)/d
    # N=number of variable
    # d=physical dimension
    assert A.shape==(N,N)

    tn = qtn.TensorNetwork([])
    # off_diagonal terms, NN
#    CPd = np.zeros((d,)*3)
#    for i in range(d):
#        CPd[i,i,i] = 1.0
    for i in range(N):
        j = (i+1)%N
        data = np.zeros((d,d))
        for k in range(d):
            for l in range(d):
                data[k,l] = np.exp((A[i,j]+A[j,i])*xs[i,k]*xs[j,l])
        inds = [qtn.rand_uuid() for i in data.shape] # pi,pj,oi
        T = qtn.Tensor(data=data,inds=inds)
        ti,tj = qtn.tensor_split(T,T.inds[:1],right_inds=T.inds[1:],
                        get='tensors',**split_opts)
        tj.transpose_(*(tj.inds[1],tj.inds[0]))
        if i==0:
            tn.add_tensor(ti,tid=i)
        else:
            Ti = tn.tensor_map[i]
            data = np.einsum('dl,dr->dlr',Ti.data,ti.data)
            inds = Ti.inds+(ti.inds[1],)
            Ti.modify(data=data,inds=inds)
        if j==0:
            Tj = tn.tensor_map[j]
            data = np.einsum('dl,dr->dlr',tj.data,Tj.data)
            inds = tj.inds+(Tj.inds[-1],)
            Tj.modify(data=data,inds=inds)
        else:
            tn.add_tensor(tj,tid=j)
    # diagonal terms
    for i in range(N):
        data = np.zeros(d)
        for k in range(d):
            data[k] = np.exp(A[i,i]*xs[i,k]**2)
        Ti = tn.tensor_map[i]
        data = np.einsum('dij,d->dij',Ti.data,data)
        Ti.modify(data=data)
    # off-diag terms
    for i in range(N):
        j_range = range(i+2,N-1) if i==0 else range(i+2,N)
        for j in j_range:
            data = np.zeros((d,d))
            for k in range(d):
                for l in range(d):
                    data[k,l] = np.exp((A[i,j]+A[j,i])*xs[i,k]*xs[j,l])
            inds = [qtn.rand_uuid() for i in data.shape] # pi,pj,oi
            T = qtn.Tensor(data=data,inds=inds)
            ti,tj = qtn.tensor_split(T,T.inds[:1],right_inds=T.inds[1:],
                            get='tensors',**split_opts)
            tj.transpose_(*(tj.inds[1],tj.inds[0]))
            group = {tn.tensor_map[i]:ti,tn.tensor_map[j]:tj}
            for T,t in group.items(): 
                data = np.einsum('dlr,dk->dlkr',T.data,t.data)
                inds = T.inds[:2]+(t.inds[1],T.inds[2])
                tmp = qtn.Tensor(data=data,inds=inds)
                L,R = qtn.tensor_split(tmp,inds[:2],right_inds=inds[2:],
                                       get='tensors',rtags='A{},{}'.format(i,j),
                                       **split_opts)
                T.modify(data=L.data,inds=L.inds)
                tn.add_tensor(R,virtual=True)
    for i in range(N):
        tn.add_tensor(qtn.Tensor(data=tr,inds=[tn.tensor_map[i].inds[0]]))
    return tn        
def get_exp_3d(A,xs,**split_opts): # get a 2d tn
    N,d = xs.shape 
    tr = np.ones(d)/d
    # N=number of variable
    # d=physical dimension
    assert A.shape==(N,N)

    tn = qtn.TensorNetwork([])
    # off_diagonal terms, NN
    CPd = np.zeros((d,)*3)
    for i in range(d):
        CPd[i,i,i] = 1.0
    for i in range(N):
        j = (i+1)%N
        data = np.zeros((d,d))
        for k in range(d):
            for l in range(d):
                data[k,l] = np.exp((A[i,j]+A[j,i])*xs[i,k]*xs[j,l])
        inds = [qtn.rand_uuid() for i in data.shape] # pi,pj,oi
        T = qtn.Tensor(data=data,inds=inds)
        ti,tj = qtn.tensor_split(T,T.inds[:1],right_inds=T.inds[1:],
                        get='tensors',**split_opts)
        tj.transpose_(*(tj.inds[1],tj.inds[0]))
        if i==0:
            tn.add_tensor(ti,tid=i)
        else:
            Ti = tn.tensor_map[i]
            data = np.einsum('dl,dr->dlr',Ti.data,ti.data)
            inds = Ti.inds+(ti.inds[1],)
            Ti.modify(data=data,inds=inds)
        if j==0:
            Tj = tn.tensor_map[j]
            data = np.einsum('dl,dr->dlr',tj.data,Tj.data)
            inds = tj.inds+(Tj.inds[-1],)
            Tj.modify(data=data,inds=inds)
        else:
            tn.add_tensor(tj,tid=j)
    # diagonal terms
    for i in range(N):
        data = np.zeros(d)
        for k in range(d):
            data[k] = np.exp(A[i,i]*xs[i,k]**2)
        Ti = tn.tensor_map[i]
        data = np.einsum('dij,d->dij',Ti.data,data)
        Ti.modify(data=data)
    # off-diag terms
    pix = [tn.tensor_map[i].inds[0] for i in range(N)]
    for i in range(N):
        j_range = range(i+2,N-1) if i==0 else range(i+2,N)
        for j in j_range:
            data = np.zeros((d,d))
            for k in range(d):
                for l in range(d):
                    data[k,l] = np.exp((A[i,j]+A[j,i])*xs[i,k]*xs[j,l])
            inds = [qtn.rand_uuid() for i in data.shape] # pi,pj,oi
            T = qtn.Tensor(data=data,inds=inds)
            ti,tj = qtn.tensor_split(T,T.inds[:1],right_inds=T.inds[1:],
                            get='tensors',**split_opts)
            tj.transpose_(*(tj.inds[1],tj.inds[0]))
            for ix,t in {i:ti,j:tj}.items():
                data = np.einsum('di,def->efi',t.data,CPd)
                inds = qtn.rand_uuid(),pix[ix],t.inds[-1]
                tags = {'x{}'.format(ix),'A{},{}'.format(i,j)}
                tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags=tags))
                pix[ix] = inds[0]
    for i in range(N):
        tn.add_tensor(qtn.Tensor(data=tr,inds=[pix[i]]))
    return tn        
  
def get_coeffs(a,M):
    # M degree taylor approximation of exp(-x) from [0,2*a] centered at a
    coeff = []
    fac = [math.factorial(k) for k in range(M+1)]
    for k in range(M+1):
        out = 0.0
        for l in range(M-k+1):
            out += a**l/fac[l]
        coeff.append(np.exp(a)*out/fac[k])
    return coeff
