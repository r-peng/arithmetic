import numpy as np
import quimb.tensor as qtn
import arithmetic.utils as utils
import itertools,functools
from autoray import do
np.set_printoptions(precision=6,suppress=True,linewidth=200)
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
#        data,expo = get_data((A[i,j]+A[j,i]),xs[i,:]*xs[j,:])
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
#        tn.exponent = tn.exponent+expo
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
def get_exp_2d(A,xs,tr,regularize=False,**split_opts): # get a 2d tn
    N,d = xs.shape 
    # N=number of variable
    # d=physical dimension
    assert A.shape==(N,N)

    tn = qtn.TensorNetwork([])
    # off_diagonal terms, NN
    for i in range(N):
        j = (i+1)%N

        fac = (A[i,j]+A[j,i])
        xi,xj = xs[i,:],xs[j,:]
        data,expo = get_data(fac,xi,xj,regularize=regularize)

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
        if regularize:
            tn.exponent = tn.exponent + expo
            tn.equalize_norms_(1.0)
            tn.balance_bonds_()

    # diagonal terms
    for i in range(N):
        fac = A[i,i]
        xi = np.square(xs[i,:])
        data,expo = get_data(fac,xi,regularize=regularize)

        Ti = tn.tensor_map[i]
        data = np.einsum('dij,d->dij',Ti.data,data)
        Ti.modify(data=data)
        if regularize:
            tn.exponent = tn.exponent + expo
            tn.equalize_norms_(1.0)
            tn.balance_bonds_()

    # off-diag terms
    for i in range(N):
        j_range = range(i+2,N-1) if i==0 else range(i+2,N)
        for j in j_range:
            fac = (A[i,j]+A[j,i])
            xi,xj = xs[i,:],xs[j,:]
            data,expo = get_data(fac,xi,xj,rgularize=regularize)

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
            if regularize:
                tn.exponent = tn.exponent + expo
                tn.equalize_norms_(1.0)
                tn.balance_bonds_()
    
    for i in range(N):
        Ti = qtn.Tensor(data=tr[i,:],inds=[tn.tensor_map[i].inds[0]])
        tn.add_tensor(Ti)
    return tn        
def decompose_exp(tn,**split_opts):
    gauges = dict()
    tensors = []
    for tid,T in tn.tensor_map.items():
        if len(T.inds)==2:
            ls,gs = decompose_mps(T,**split_opts)
        elif len(T.inds)==4:
            ls,gs = decompose_mps(T,**split_opts)
            #ls,gs = decompose_tree(T,**split_opts)
        else:
            ls,gs = [T],dict()
        tensors += ls
        gauges.update(gs)
    return qtn.TensorNetwork(tensors),gauges
def decompose_mps(T,**split_opts):
    ls = []
    gauges = dict()
    for i,pix in enumerate(T.inds[:-1]):
        if i==0:
            lix = [pix]
        else:
            lix = [S.inds[0],pix]
        L,S,T = T.split(lix,absorb=None,get='tensors',**split_opts)
        print(S.data)
        ls.append(L)
        gauges[S.inds[0]] = S.data
    ls.append(T)
    return ls,gauges
def decompose_tree(T,**split_opts):
    assert len(T.inds)==4
    gauges = dict()
    for ix in [1,2,3]:
        lix = [T.inds[i] for i in [0,ix]]
        L,S,R = T.split(lix,absorb=None,get='tensors',**split_opts)
        if ix==1:
            Lmin,Smin,Rmin = L,S,R
        else:
            if len(S.data)<len(Smin.data):
                Lmin,Smin,Rmin = L,S,R
    gauges[Smin.inds[0]] = Smin.data 
    ls = []
    for t in [Lmin,Rmin]:
        inds = list(t.inds)
        inds.remove(Smin.inds[0])
        for i,lix in enumerate(inds):
            l,s,r = t.split([lix],absorb=None,get='tensors',**split_opts)
            if i==0:
                lmin,smin,rmin = l,s,r
            else:
                if len(s.data)<len(smin.data):
                    lmin,smin,rmin = l,s,r
        gauges[smin.inds[0]] = smin.data
        ls += [lmin,rmin]
    return ls,gauges 
def gauge_all_simple(tn,max_iterations=5,tol=1e-6,smudge=1e-12,power=1.0,
                     cutoff=1e-15):
    """Iterative gauge all the bonds in this tensor network with a 'simple
    update' like strategy.
    """
    # every index in the TN
    inds = list(tn.ind_map)
    # the vector 'gauges' that will live on the bonds
    gauges = {}
    # for retrieving singular values
    info = {}
    # accrue scaling to avoid numerical blow-ups
    nfact = 0.0
    it = 0
    not_converged = True
    while not_converged and it < max_iterations:
        # can only converge if tol > 0.0
        all_converged = tol > 0.0
        for ind in inds:
            try:
                tid1, tid2 = tn.ind_map[ind]
            except (KeyError, ValueError):
                # fused multibond (removed) or not a bond (len(tids != 2))
                continue
            t1 = tn.tensor_map[tid1]
            t2 = tn.tensor_map[tid2]
            lix, bix, rix = qtn.group_inds(t1, t2)
            bond = bix[0]
            assert len(bix)==1
            if len(bix) > 1:
                # first absorb separate gauges
                for ind in bix:
                    s = gauges.pop(ind, None)
                    if s is not None:
                        t1.multiply_index_diagonal_(ind, s**0.5)
                        t2.multiply_index_diagonal_(ind, s**0.5)
                # multibond - fuse it
                t1.fuse_({bond: bix})
                t2.fuse_({bond: bix})
            # absorb 'outer' gauges into tensors
            inv_gauges = []
            for t, ixs in ((t1, lix), (t2, rix)):
                for ix in ixs:
                    try:
                        s = (gauges[ix] + smudge)**power
                    except KeyError:
                        continue
                    t.multiply_index_diagonal_(ix, s)
                    # keep track of how to invert gauge
                    inv_gauges.append((t, ix, 1 / s))
            # absorb the inner gauge, if it exists
            if bond in gauges:
                t1.multiply_index_diagonal_(bond, gauges[bond])
            # perform SVD to get new bond gauge
            qtn.tensor_compress_bond(
                t1, t2, absorb=None, info=info, cutoff=cutoff)
            s = info['singular_values'].data
            smax = s[0]
            new_gauge = s / smax
            nfact = do('log10', smax) + nfact
            if tol > 0.0:
                # check convergence
                old_gauge = gauges.get(bond,np.ones(1))
                lold,lnew = len(old_gauge),len(new_gauge)
                if lold<lnew:
                    old_gauge_ = np.zeros(lnew)
                    old_gauge_[:lold] = old_gauge
                    sdiff = do('linalg.norm', old_gauge_ - new_gauge)
                    sdiff /= np.sqrt(lnew)
                elif lnew<lold:
                    new_gauge_ = np.zeros(lold)
                    new_gauge_[:lnew] = new_gauge
                    sdiff = do('linalg.norm', old_gauge - new_gauge_)
                    sdiff /= np.sqrt(lold)
                else:
                    sdiff = do('linalg.norm', old_gauge - new_gauge)
                    sdiff /= np.sqrt(lnew)
                all_converged &= sdiff < tol
            # update inner gauge and undo outer gauges
            gauges[bond] = new_gauge
            for t, ix, inv_s in inv_gauges:
                t.multiply_index_diagonal_(ix, inv_s)
        not_converged = not all_converged
        it += 1
    # redistribute the accrued scaling
    tn.multiply_each_(10**(nfact / tn.num_tensors))
    # absorb all bond gauges
    for ix, s in gauges.items():
        t1, t2 = map(tn.tensor_map.__getitem__, tn.ind_map[ix])
        s_1_2 = s**0.5
        t1.multiply_index_diagonal_(ix, s_1_2)
        t2.multiply_index_diagonal_(ix, s_1_2)
    return tn
################# useful fxn below ###########################
def get_1body(i,A,B,xs,regularize=False):
    fac = A[i,i]
    xi = np.square(xs[i,:])
    t,e = get_data(fac,xi,regularize=regularize)
    if B is not None:
        fac = B[i,i,i,i]
        xi = np.square(xi)
        t_,e_ = get_data(fac,xi,regularize=regularize)
        t,e = np.einsum('i,i->i',t,t_),e+e_
    return i,(t,e)
def get_2body(idx,A,B,xs,regularize=False):
    i,j = idx
    fac = A[i,j]+A[j,i]
    xi = xs[i,:].copy()
    xj = xs[j,:].copy()
    t,e = get_data(fac,xi,xj,regularize=regularize)
    if B is not None:
        #xi^3xj
        fac = get_fac_iiij(B,i,j)
        xi = np.square(xs[i,:])
        xi = np.multiply(xi,xs[i,:])
        xj = xs[j,:].copy()
        t_,e_ = get_data(fac,xi,xj,regularize=regularize)
        t,e = np.einsum('ij,ij->ij',t,t_),e+e_
        #xixj^3
        fac = get_fac_iiij(B,j,i)
        xi = xs[i,:].copy()
        xj = np.square(xs[j,:])
        xj = np.multiply(xj,xs[j,:])
        t_,e_ = get_data(fac,xi,xj,regularize=regularize)
        t,e = np.einsum('ij,ij->ij',t,t_),e+e_
        #xi^2xj^2
        fac = get_fac_iijj(B,i,j)
        xi = np.square(xs[i,:])
        xj = np.square(xs[j,:])
        t_,e_ = get_data(fac,xi,xj,regularize=regularize)
        t,e = np.einsum('ij,ij->ij',t,t_),e+e_
    return tuple(idx),(t,e)
def get_3body(idx,B,xs,regularize=False): 
    i,j,k = idx
    # xi^2xjxk
    fac = get_fac_iijk(B,i,j,k)
    xi = np.square(xs[i,:])
    xj = xs[j,:].copy()
    xk = xs[k,:].copy()
    t,e = get_data(fac,xi,xj,xk,regularize=regularize)
    # xixj^2xk
    fac = get_fac_iijk(B,j,i,k)
    xi = xs[i,:].copy()
    xj = np.square(xs[j,:])
    xk = xs[k,:].copy()
    t_,e_ = get_data(fac,xi,xj,xk,regularize=regularize)
    t,e = np.einsum('ijk,ijk->ijk',t,t_),e+e_
    # xixjxk^2
    fac = get_fac_iijk(B,k,i,j)
    xi = xs[i,:].copy()
    xj = xs[j,:].copy()
    xk = np.square(xs[k,:])
    t_,e_ = get_data(fac,xi,xj,xk,regularize=regularize)
    t,e = np.einsum('ijk,ijk->ijk',t,t_),e+e_
    return tuple(idx),(t,e)
def get_4body(idx,B,xs,regularize=False):
    i,j,k,l = idx
    fac = get_fac_ijkl(B,i,j,k,l)
    xi = xs[i,:].copy()
    xj = xs[j,:].copy()
    xk = xs[k,:].copy()
    xl = xs[l,:].copy()
    t,e = get_data(fac,xi,xj,xk,xl,regularize=regularize)
    return tuple(idx),(t,e)
def get_data(fac,*x_ls,regularize=False):
    nb = len(x_ls)
    d = len(x_ls[0])
    id_ls = list(itertools.product(range(d),repeat=nb))
    expo = np.zeros((d,)*nb)
    for ids in id_ls:
        x = np.array([x_ls[i][ids[i]] for i in range(nb)])
        expo[ids] = fac*np.prod(x)
    if regularize:
        max_expo = np.amax(expo)
        expo -= max_expo
        max_expo /= np.log(10.0)   
    else:
        max_expo = 0.0
    data = np.zeros((d,)*nb)
    for ids in id_ls:
        data[ids] = np.exp(expo[ids])
    return data,max_expo
def get_map(xs,A,B=None,regularize=False):
    N,d = xs.shape
    Tmap = dict()
    print('get 1-body terms...')
    for i in range(N):
        key,val = get_1body(i,A,B,xs,regularize=regularize)
        Tmap[key] = val
    print('get 2-body terms...')
    for i in range(N):
        for j in range(i+1,N):
            key,val = get_2body([i,j],A,B,xs,regularize=regularize)
            Tmap[key] = val
    if B is not None:
        print('get 3-body terms...')
        for i in range(N):
            for j in range(i+1,N):
                for k in range(j+1,N):
                    key,val = get_3body([i,j,k],B,xs,regularize=regularize)
                    Tmap[key] = val
        print('get 4-body terms...')
        for i in range(N):
            for j in range(i+1,N):
                for k in range(j+1,N):
                    for l in range(k+1,N):
                        key,val = get_4body([i,j,k,l],B,xs,regularize=regularize)
                        Tmap[key] = val
    return Tmap
def get_map_parallel(xs,A,B=None,regularize=False,nworkers=5):
    import multiprocessing
    N,d = xs.shape
    pool = multiprocessing.Pool(nworkers)
    Tmap = dict()
    print('get 1-body terms...')
    keys = range(N)
    get_term = functools.partial(get_1body,A=A,B=B,xs=xs,regularize=regularize)
    ls = pool.map(get_term,keys)
    for item in ls:
        Tmap[item[0]] = item[1]
    print('get 2-body terms...')
    keys = []
    for i in range(N):
        for j in range(i+1,N):
            keys.append((i,j))
    get_term = functools.partial(get_2body,A=A,B=B,xs=xs,regularize=regularize)
    ls = pool.map(get_term,keys)
    for item in ls:
        Tmap[item[0]] = item[1]
    print('get 3-body terms...')
    keys = []
    for i in range(N):
        for j in range(i+1,N):
            for k in range(j+1,N):
                keys.append((i,j,k))
    get_term = functools.partial(get_3body,B=B,xs=xs,regularize=regularize)
    ls = pool.map(get_term,keys)
    for item in ls:
        Tmap[item[0]] = item[1]
    print('get 4-body terms...')
    keys = []
    for i in range(N):
        for j in range(i+1,N):
            for k in range(j+1,N):
                for l in range(k+1,N):
                    keys.append((i,j,k,l))
    get_term = functools.partial(get_4body,B=B,xs=xs,regularize=regularize)
    ls = pool.map(get_term,keys)
    for item in ls:
        Tmap[item[0]] = item[1]
    return Tmap
def merge_terms(Tmap,N):
    print('merge 1-body terms...')
    for i in range(N):
        t1,e1 = Tmap[i]
        if i==N-1:
            j = 0
            t2,e2 = Tmap[j,i]
            Tmap[j,i] = np.einsum('i,ji->ji',t1,t2),e1+e2
        else:
            j = N-1
            t2,e2 = Tmap[i,j]
            Tmap[i,j] = np.einsum('i,ij->ij',t1,t2),e1+e2
        Tmap.pop(i)
    tmp = Tmap.get((0,1,2,3),None)
    if tmp is not None:
        print('merge 2-body terms...')
        for i in range(N):
            for j in range(i+1,N):
                t2,e2 = Tmap[i,j]
                if j==N-1:
                    if i==N-2:
                        k = 0
                        t3,e3 = Tmap[k,i,j]
                        Tmap[k,i,j] = np.einsum('ij,kij->kij',t2,t3),e2+e3
                    else:
                        k = N-2
                        t3,e3 = Tmap[i,k,j]
                        Tmap[i,k,j] = np.einsum('ij,ikj->ikj',t2,t3),e2+e3
                else:
                    k = N-1
                    t3,e3 = Tmap[i,j,k]
                    Tmap[i,j,k] = np.einsum('ij,ijk->ijk',t2,t3),e2+e3
                Tmap.pop((i,j))
        print('merge 3-body terms...')
        for i in range(N):
            for j in range(i+1,N):
                for k in range(j+1,N):
                    t3,e3 = Tmap[i,j,k]
                    if k==N-1:
                        if j==N-2:
                            if i==N-3:
                                l = 0
                                t4,e4 = Tmap[l,i,j,k]
                                t,e = np.einsum('ijk,lijk->lijk',t3,t4),e3+e4
                                Tmap[l,i,j,k] = t,e
                            else:
                                l = N-3
                                t4,e4 = Tmap[i,l,j,k]
                                t,e = np.einsum('ijk,iljk->iljk',t3,t4),e3+e4
                                Tmap[i,l,j,k] = t,e
                        else:
                            l = N-2
                            t4,e4 = Tmap[i,j,l,k]
                            Tmap[i,j,l,k] = np.einsum('ijk,ijlk->ijlk',t3,t4),e3+e4
                    else:
                        l = N-1 
                        t4,e4 = Tmap[i,j,k,l]
                        Tmap[i,j,k,l] = np.einsum('ijk,ijkl->ijkl',t3,t4),e3+e4
                    Tmap.pop((i,j,k))
    return Tmap
def get_exp(xs,tr,A,B=None,regularize=False,nworkers=None):
    N,d = xs.shape
    if nworkers is None:
        Tmap = get_map(xs,A,B=B,regularize=regularize)
    else:
        Tmap = get_map_parallel(xs,A,B=B,regularize=regularize,nworkers=nworkers)
    Tmap = merge_terms(Tmap,N)
    tn = qtn.TensorNetwork([])
    for key,(data,expo) in Tmap.items():
        inds = ['x{}'.format(i) for i in key]
        tags = set(inds).union({'exp'})
        tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags=tags))
        tn.exponent = tn.exponent + expo
    for i in range(N):
        tn.add_tensor(qtn.Tensor(data=tr[i,:],inds=['x{}'.format(i)],tags='tr'))
    return tn

def exact(D):
    N = len(D)
    num = N/2.0*np.log10(2.0*np.pi)
    denom = 1.0/2.0*sum([np.log10(Di) for Di in D])
    return 10**(num-denom)
def diag(xs,tr,Ad,Bd=None,regularize=False): 
    N,d = xs.shape
    prod = 1.0
    for i in range(N):
        fac = Ad[i]
        xi = np.square(xs[i,:])
        t,e = get_data(fac,xi,regularize=regularize)
        if Bd is not None:
            fac = Bd[i]
            xi = np.square(xi)
            t_,e_ = get_data(fac,xi,regularize=regularize)
            t,e = np.einsum('i,i->i',t,t_),e+e_
        prod *= np.einsum('i,i->',t,tr[i,:])
        prod *= 10**e
    return prod
get_fac_ijkl = utils.get_fac_ijkl
get_fac_iijk = utils.get_fac_iijk
get_fac_iijj = utils.get_fac_iijj
get_fac_iiij = utils.get_fac_iiij
quad = utils.quad
delete_tn_from_disc = utils.delete_tn_from_disc
load_tn_from_disc = utils.load_tn_from_disc
write_tn_to_disc = utils.write_tn_to_disc
