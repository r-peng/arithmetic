import numpy as np
import scipy
import quimb.tensor as qtn
ADD = np.zeros((2,)*3)
ADD[0,1,1] = ADD[1,0,1] = ADD[0,0,0] = 1.
CP2 = np.zeros((2,)*3)
CP2[0,0,0] = CP2[1,1,1] = 1.
def get_Hmc(h1,eri,method='eigh'):
    # eri = (pr|qs)
    norb = h1.shape[0]
    v0 = h1 - 0.5*np.einsum('prrs->ps',eri)

    eri = eri.reshape((norb**2,)*2)
    print('check hermitian:',np.linalg.norm(eri-eri.T))
    if method=='eigh':
        D,L = np.linalg.eigh(eri)
        L = np.einsum('mg,g->mg',L,np.sqrt(D))
    elif method=='cholesky':
        L = np.linalg.cholesky(eri)
    else:
        raise NotImplementedError(f'{method} decomposition not implemented!')
    L = L.reshape((norb,norb,L.shape[-1]))
    vg = 1j*L
    return v0,vg
def compress(tn,tag,maxiter=10,**compress_opts):
    L = tn.num_tensors
    max_bond = tn.max_bond()
    for i in range(maxiter):
        # canonize from right
        for i in range(L-1,0,-1):
            tn.canonize_between(tag+str(i-1),tag+str(i),absorb='left')
        # compress from left 
        for i in range(L-1):
            tn.compress_between(tag+str(i),tag+str(i+1),absorb='right',
                                **compress_opts)
        # canonize from left
        for i in range(L-1):
            tn.canonize_between(tag+str(i),tag+str(i+1),absorb='right')
        # compress from right
        for g in range(L-1,0,-1):
            tn.compress_between(tag+str(i-1),tag+str(i),absorb='left',
                                **compress_opts)
        max_bond_new = tn.max_bond()
        if max_bond==max_bond_new:
            break
        max_bond = max_bond_new
    return tn
def scalar_multiply(tn,c):
    if isinstance(tn,qtn.Tensor):
        T = tn
    else:
        tid = tuple(tn.ind_map['k'])[0]
        T = tn.tensor_map[tid]
    T.reindex_({'k':'i1'})

    ls = [T]
    ls.append(qtn.Tensor(data=np.array([1.,c]),inds=('i2',)))
    ls.append(qtn.Tensor(data=CP2,inds=('i1','i2','k')))
    Tnew = qtn.tensor_contract(*ls)
    T.modify(data=Tnew.data,inds=Tnew.inds)
    return tn
def add(tn,c):
    if isinstance(tn,qtn.Tensor):
        T = tn
    else:
        tid = tuple(tn.ind_map['k'])[0]
        T = tn.tensor_map[tid]
    dim = T.shape[T.inds.index('p')]
    T.reindex_({'p':'p1','q':'q1','k':'i1'})

    CP = np.zeros((dim,)*3)
    for i in range(dim):
        CP[i,i,i] = 1.
    data = np.ones((norb,norb,2))
    data[:,:,1] = np.eye(norb)*c

    ls = [T]
    ls.append(qtn.Tensor(data=data,inds=('p2','q2','i2')))
    ls.append(qtn.Tensor(data=CP,inds=('p1','p2','p')))
    ls.append(qtn.Tensor(data=CP,inds=('q1','q2','q')))
    ls.append(qtn.Tensor(data=ADD,inds=('i1','i2','k')))
    Tnew = qtn.tensor_contract(*ls)
    T.modify(data=Tnew.data,inds=Tnew.inds)
    return tn
def matrix_multiply(tn,U,back=True):
    dim1,dim2 = U.shape
    if isinstance(tn,qtn.Tensor):
        T = tn
    else:
        tid = tuple(tn.ind_map['k'])[0]
        T = tn.tensor_map[tid]
    dim = T.shape[T.inds.index('p')]

    sCP2 = np.zeros((2,)*3)
    sCP2[0,0,0] = 1./dim
    sCP2[1,1,1] = 1.
    data = np.ones((dim1,dim2,2))
    data[:,:,1] = U
    if back: # B*U
        T.reindex_({'q':'_','k':'i1'})
        ls = [T,qtn.Tensor(data=data,inds=('_','q','i2'))]
    else: # U*B
        T.reindex_({'p':'_','k':'i1'})
        ls = [T,qtn.Tensor(data=data,inds=('p','_','i2'))]
    ls.append(qtn.Tensor(data=sCP2,inds=('i1','i2','k')))
    Tnew = qtn.tensor_contract(*ls)
    T.modify(data=Tnew.data,inds=Tnew.inds)
    return tn

def get_exponent(v0,vg,xs,tau,**compress_opts):
    norb,_,nf = vg.shape
    ng = len(xs)

    CP = np.zeros((norb,)*3)
    for i in range(norb):
        CP[i,i,i] = 1.
    tn = qtn.TensorNetwork([])

    data = np.ones((norb,norb,2))
    data[:,:,1] = -np.sqrt(tau)*v0
    data = np.einsum('PQi,Ppr,Qqs,ijk->pqkrsj',data,CP,CP,ADD)
    inds = 'p','q','k','v0,1','v0,1_','v0,1__'
    tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags={'v0'}))
    for i in range(1,nf+1):
        data = np.zeros((2,norb,norb,2),dtype=vg.dtype)
        data[1,:,:,1] = vg[:,:,i-1]
        data[0,:,:,0] = np.ones((norb,)*2)
        inds = (f'x{i}_',) + inds[-3:]
        if i<nf:
            data = np.einsum('xPQi,Ppr,Qqs,ijk->xpqkrsj',data,CP,CP,ADD)
            inds = inds + (f'v{i},{i+1}',f'v{i},{i+1}_',f'v{i},{i+1}__')
        tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags={f'v{i}'}))
    tn.fuse_multibonds_()
    tn = compress(tn,'v',**compress_opts)
    tn = scalar_multiply(tn,np.sqrt(tau))
    for i in range(1,nf+1):
        data = np.ones((ng,2))
        data[:,1] = xs.copy()
        inds = f'x{i}',f'x{i}_'
        tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags={f'v{i}'}))
        tn.contract_tags(f'v{i}',which='any',inplace=True)
    return tn
def get_exponential(tni,order=3,**compress_opts):
    # tni = -tau*v0+sqrt(tau)*sum(xi*vi) 
    nf = tni.num_tensors-1
    norb = tni['v0'].shape[tni['v0'].inds.index('p')]
    ng = tni['v1'].shape[tni['v1'].inds.index('x1')]
    coeff = [1./np.math.factorial(i) for i in range(order,-1,-1)]

    sCP2 = np.zeros((2,)*3)
    sCP2[0,0,0] = 1./norb
    sCP2[1,1,1] = 1.
    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1.

    tn = tni.copy()
    tn = scalar_multiply(tn,coeff[0])
    tn = add(tn,coeff[1])
    for i in range(2,len(coeff)):
        T1 = tn['v0']
        T2 = tni['v0'].copy()
        T1.reindex_({'p':'_','k':'i1'})
        T2.reindex_({'q':'_','k':'i2','v0,1':'v0,1_'})
        ls = [T1,T2]
        ls.append(qtn.Tensor(data=sCP2,inds=('i1','i2','k')))
        Tnew = qtn.tensor_contract(*ls)
        T1.modify(data=Tnew.data,inds=Tnew.inds)
        for j in range(1,nf+1):
            T1 = tn[f'v{j}']
            T2 = tni[f'v{j}'].copy()
            T1.reindex_({f'x{j}':f'x{j}_1'})
            T2.reindex_({f'x{j}':f'x{j}_2',f'v{j-1},{j}':f'v{j-1},{j}_'})
            if j<nf:
                rix = tuple(tni[f'v{j}'].bonds(tni[f'v{j+1}']))[0]
                T2.reindex_({f'v{j},{j+1}':f'v{j},{j+1}_'})
            ls = [T1,T2]
            ls.append(qtn.Tensor(data=CP,inds=(f'x{j}_1',f'x{j}_2',f'x{j}')))
            Tnew = qtn.tensor_contract(*ls)
            T1.modify(data=Tnew.data,inds=Tnew.inds)
        tn = compress(tn,'v',**compress_opts)
        tn = add(tn,coeff[i])
    return tn
def get_nstep_backbone(T0,nstep,**compress_opts):
    # tni = exp(-tau*v0+sqrt(tau)*sum(xi*vi))
    norb = T0.shape[T0.inds.index('p')]

    sCP2 = np.zeros((2,)*3)
    sCP2[0,0,0] = 1./norb
    sCP2[1,1,1] = 1.

    tn = qtn.TensorNetwork([])
    for i in range(nstep):
        Ti = T0.copy()
        Ti.add_tag(f's{i}')
        pnew = 'p' if i==nstep-1 else f'v0_s{i},{i+1}'
        qnew = 'q' if i==0 else f'v0_s{i-1},{i}'
        Ti.reindex_({'p':pnew,'q':qnew,'k':f'i{i}','v0,1':f's{i}_v0,1'})
        tn.add_tensor(Ti)
        if i>0: 
            lix = 'i0' if i==1 else f'i{i-1},{i}_'
            rix = 'k' if i==nstep-1 else f'i{i},{i+1}_'
            tn.add_tensor(qtn.Tensor(data=sCP2,inds=(lix,f'i{i}',rix),
                                     tags={'a',f's{i}'}))
    tn_ = tn.select('a',which='any',virtual=True) 
    ix_map = {0:'q',nstep-1:'p'}
    ls = []
    # move q-leg
    for i in ix_map:
        Ti = tn['v0',f's{i}']
        Tl,Tr = Ti.split(left_inds=None,right_inds=(f'i{i}',ix_map[i]),
                         absorb='right',**compress_opts)
        Ti.modify(data=Tl.data,inds=Tl.inds)

        ix_map[i] = Tr.inds[0]
        ls.append(Tr)
        tn_.add_tensor(Ti,virtual=True)
    for i in range(1,nstep-1):
        Ti = tn_[f's{i}']
        blob = qtn.tensor_contract(ls[0],Ti)
        Tl,ls[0] = blob.split(left_inds=None,right_inds=('q',f'i{i},{i+1}_'),
                              absorb='right',bond_ind=f'i{i},{i+1}',**compress_opts)
        Ti.modify(data=Tl.data,inds=Tl.inds)
    Ti = tn_['a',f's{nstep-1}']
    blob = qtn.tensor_contract(*ls,Ti)
    Ti.modify(data=blob.data,inds=blob.inds)
    # compress involved tensors
    T = tn_['v0',f's{nstep-1}']
    T.retag_({f's{nstep-1}':f's{nstep}'})
    tn_ = compress(tn_,'s',**compress_opts)
    T.retag_({f's{nstep}':f's{nstep-1}'})
    # seperate p,q,k
    Tl,Tr = Ti.split(left_inds=None,right_inds=('p','q','k'),
                     absorb='right',**compress_opts)
    Ti.modify(data=Tl.data,inds=Tl.inds)
    Tr.modify(tags={})
    tn.add_tensor(Tr)
    tn.reindex_({bix:f'i{i}' for i,bix in ix_map.items()})
    return tn
def get_inverse_backbone(T,order):
    norb = T.shape[T.inds.index('p')]
    sCP2 = np.zeros((2,)*3)
    sCP2[0,0,0] = 1./norb
    sCP2[1,1,1] = 1.

    T = scalar_multiply(T,-1.)
    T = add(T,1.)

    tn = qtn.TensorNetwork([T])
    tn = add(tn,1.)
    tn.add_tag('o0') 
    for i in range(1,order):
        tn[f'o{i-1}'].reindex_({'p':'o{i-1},{i}','k':'o{i-1},{i}_'})
        ls = [T.reindex({'q':'o{i-1},{i}','k':'_'})]
        ls.append(qtn.Tensor(data=sCP2,inds=('k','_','o{i-1},{i}_')))
        Ti = qtn.tensor_contract(*ls)
        Ti.add_tag(f'o{i}')
    return
class BackBone:
    def __init__(self,T0,nstep,**compress_opts):
        self.nstep = nstep
        self.tn = get_nstep_backbone(T0,nstep,**compress_opts)
class Propagator:
    def __init__(self,v0,vg,xs,tau,**compress_opts):
        tn = get_exponent(v0,vg,xs,tau,**compress_opts)
        self.B = get_exponential(tn,**compress_opts)

        self.field = self.B.copy()
        tid = tuple(self.field._get_tids_from_tags('v0',which='any'))[0]
        self.T0 = self.field._pop_tensor(tid)

        self.nstep_backbone = None
        self.inverse_backbone = None
        
    def init_nstep_backbone(self,nstep,**compress_opts):
        self.nstep_backbone = BackBone(self.T0,nstep,**compress_opts)
        return self.nstep_backbone.tn
    def get_nstep_propagator_from_backbone(self):
        nf = self.field.num_tensors
        tn = self.nstep_backbone.tn.copy() 
        for i in range(nstep):
            tni_ = self.field.copy()
            tni_.add_tag(f's{i}')
            tni_.reindex_({f'x{j}':f's{i},x{j}' for j in range(1,nf+1)})
            tni_['v1'].reindex_({'v0,1':f's{i}_v0,1'})
            tn.add_tensor_network(tni_,check_collisions=True)
        return tn
        
if __name__=='__main__':
    norb = 3
    t = 1.
    u = 1.

    h1 = np.zeros((norb,)*2)
    for i in range(norb):
        if i-1>0:
            h1[i,i-1] = -t
        if i+1<norb:
            h1[i,i+1] = -t
    eri = np.zeros((norb,)*4)
    for i in range(norb):
        eri[i,i,i,i] = u
    v0,vg = get_Hmc(h1,eri)
    nf = vg.shape[-1]
    ng = 4
    xs = np.random.rand(ng)
    idxs = {i:np.random.randint(low=0,high=ng) for i in range(1,nf+1)}

    cutoff = 1e-15
    order = 3
    coeff = [1./scipy.math.factorial(i) for i in range(order,-1,-1)]
    for tau in [1e-1,1e-2,1e-3]:
        A = get_exponent(v0,vg,xs,tau,cutoff=cutoff)
        B = get_exponential(A,order=order,cutoff=cutoff)
        print(A)
        print(B)
        for i in range(1,nf+1):
            vec = np.zeros(ng)
            vec[idxs[i]] = 1.
            A.add_tensor(qtn.Tensor(data=vec,inds=(f'x{i}',)))
            B.add_tensor(qtn.Tensor(data=vec,inds=(f'x{i}',)))
        A = A.contract(output_inds=('p','q','k')).data
        B = B.contract(output_inds=('p','q','k')).data

        A_ = np.array(-tau*v0,dtype=vg.dtype)
        for i in range(1,nf+1):
            A_ += np.sqrt(tau)*xs[idxs[i]]*vg[:,:,i-1]
        exp = scipy.linalg.expm(A_) 
        pol = A_*coeff[0]+np.eye(norb)*coeff[1]
        for ci in coeff[2:]:
            pol = np.einsum('pr,rq->pq',A_,pol)+ci*np.eye(norb)
        print(f'tau={tau}')
        print('check A[0]=',np.linalg.norm(np.ones((norb,)*2)-A[:,:,0]))
        print('check B[0]=',np.linalg.norm(np.ones((norb,)*2)-B[:,:,0]))
        print('check A[1]=',np.linalg.norm(A_-A[:,:,1]))
        print('check pol[1]=',np.linalg.norm(pol-B[:,:,1]))
        print('check exp[1]=',np.linalg.norm(exp-B[:,:,1]))
    print()
    # check full propagator
    tau = 0.01
    nstep = 5
    cls = Propagator(v0,vg,xs,tau,cutoff=cutoff)
    tn = cls.init_nstep_backbone(nstep,cutoff=cutoff)
    Bn = cls.get_nstep_propagator_from_backbone()
    print(tn)
    for i in range(1,nf+1):
        vec = np.zeros(ng)
        vec[idxs[i]] = 1.
        for j in range(nstep):
            Bn.add_tensor(qtn.Tensor(data=vec,inds=(f's{j},x{i}',)))
    Bn = Bn.contract(output_inds=('p','q','k')).data
    A_ = np.array(-tau*v0,dtype=vg.dtype)
    for i in range(1,nf+1):
        A_ += np.sqrt(tau)*xs[idxs[i]]*vg[:,:,i-1]
    exp = scipy.linalg.expm(A_) 
    pol = A_*coeff[0]+np.eye(norb)*coeff[1]
    for ci in coeff[2:]:
        pol = np.einsum('pr,rq->pq',A_,pol)+ci*np.eye(norb)
    expn = np.eye(norb)
    poln = np.eye(norb)
    for i in range(nstep):
        expn = np.dot(exp,expn)
        poln = np.dot(pol,poln)
    print('check Bn[0]=',np.linalg.norm(np.ones((norb,)*2)-Bn[:,:,0]))
    print('check poln[1]=',np.linalg.norm(poln-Bn[:,:,1]))
    print('check expn[1]=',np.linalg.norm(expn-Bn[:,:,1]))
    print()
    # check matrix multiply
    nocc = 2
    U = np.random.rand(norb,nocc)
    cls.nstep_backbone.tn = matrix_multiply(cls.nstep_backbone.tn,U,back=True)
    cls.nstep_backbone.tn = matrix_multiply(cls.nstep_backbone.tn,U.T,back=False)
    Bn = cls.get_nstep_propagator_from_backbone()
    for i in range(1,nf+1):
        vec = np.zeros(ng)
        vec[idxs[i]] = 1.
        for j in range(nstep):
            Bn.add_tensor(qtn.Tensor(data=vec,inds=(f's{j},x{i}',)))
    Bn = Bn.contract(output_inds=('p','q','k')).data
    expn = np.linalg.multi_dot([U.T,expn,U])
    poln = np.linalg.multi_dot([U.T,poln,U])
    print('check UBnU[0]=',np.linalg.norm(np.ones((nocc,)*2)-Bn[:,:,0]))
    print('check UpolnU[1]=',np.linalg.norm(poln-Bn[:,:,1]))
    print('check UexpnU[1]=',np.linalg.norm(expn-Bn[:,:,1]))
    exit()
    # check inverse
    tau = 1e-3
    order = 5
    coeff = [1. for i in range(order+1)]
    A = np.array(-tau*v0,dtype=vg.dtype)
    for g in range(vg.shape[-1]):
        A += np.sqrt(tau)*xs[idxs[g]]*vg[:,:,g]
    A_ = -A.copy()
    A = scipy.linalg.expm(A)
    A_ = scipy.linalg.expm(A_)
    w,v = np.linalg.eig(A)
    print('eigval(A)=',w)
    Dinv = np.array([1./w for w in np.diag(A)])
    print('1/D=',Dinv)
    print('diag(A_)=',np.diag(A_))
    inv = np.linalg.inv(A)
    phi0 = np.eye(norb)
    phi1 = np.diag(Dinv)
    phi2 = np.eye(norb)/(2.*norb)
    for i,phi in enumerate([phi0,phi1,phi2]):
        print(f'neumann_{i}')
        A_ = np.eye(norb)-np.dot(phi,A)
        w_,v = np.linalg.eig(A_)
        print('eigval(I-phi*A)=',w_)
        A1 = A_*coeff[0]+np.eye(norb)*coeff[1]
        for j,cj in enumerate(coeff[2:]):
            A1 = np.einsum('pr,rq->pq',A_,A1)+cj*np.eye(norb)
            print(f'order={j+2},inv={np.linalg.norm(inv-np.dot(A1,phi))}')
