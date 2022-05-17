import numpy as np
import scipy
import quimb.tensor as qtn
import functools
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
def compress1D(tn,tag,maxiter=10,final='left',**compress_opts):
    L = tn.num_tensors
    max_bond = tn.max_bond()
    def canonize_from_left():
        for i in range(L-1):
            tn.canonize_between(f'{tag}{i}',f'{tag}{i+1}',absorb='right')
    def canonize_from_right():
        for i in range(L-1,0,-1):
            tn.canonize_between(f'{tag}{i-1}',f'{tag}{i}',absorb='left')
    def compress_from_left():
        for i in range(L-1):
            tn.compress_between(f'{tag}{i}',f'{tag}{i+1}',absorb='right',
                                **compress_opts)
    def compress_from_right():
        for i in range(L-1,0,-1):
            tn.compress_between(f'{tag}{i-1}',f'{tag}{i}',absorb='left',
                                **compress_opts)
    if final=='left':
        canonize_from_left()
        def sweep():
            compress_from_right()
            compress_from_left()
    elif final=='right':
        canonize_from_right()
        def sweep():
            compress_from_left()
            compress_from_right()
    else:
        raise NotImplementedError(f'{final} canonical form not implemented!')
    for i in range(maxiter):
        sweep()
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
    data = np.ones((dim,dim,2))
    data[:,:,1] = np.eye(dim)*c

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
    tn = compress1D(tn,'v',final='right',**compress_opts)
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
        tn = compress1D(tn,'v',final='right',**compress_opts)
        tn = add(tn,coeff[i])
    return tn
def get_backbone(T,n,tag,coeff=None,make_mps=True,**compress_opts):
    norb = T.shape[T.inds.index('p')]
    oix = tuple(set(T.inds).difference({'p','q','k'}))[0]

    sCP2 = np.zeros((2,)*3)
    sCP2[0,0,0] = 1./norb
    sCP2[1,1,1] = 1.

    tn = qtn.TensorNetwork([])
    bix = tag
    L,R = T.split(left_inds=None,right_inds=('p','k'),absorb='right',
                  bond_ind=bix,**compress_opts)
    tmp = T
    if coeff is not None:
        tmp = add(tmp,coeff[0])
    for i in range(1,n):
        L1,R1 = tmp.split(left_inds=None,right_inds=('q','k'),absorb='right',
                          **compress_opts)
        if i==1:
            L1.reindex_({oix:f'{tag}0'})
        L1.reindex_({'p':f'{tag}{i-1},{i}_'})
        R1.reindex_({'k':'i1'})
        L2 = L.reindex({'q':f'{tag}{i-1},{i}_',oix:f'{tag}{i}',bix:f'{tag}{i}_'})
        R2 = R.reindex({'k':'i2',bix:f'{tag}{i}_'})
        ls = [R1,R2]
        ls.append(qtn.Tensor(data=sCP2,inds=('i1','i2','k')))
        blob = qtn.tensor_contract(*ls)
        Tl,tmp = blob.split(left_inds=None,right_inds=('p','q','k'),absorb='right',
                            bond_ind=f'{tag}{i},{i+1}',**compress_opts)
        if make_mps:
            blob = qtn.tensor_contract(L1,L2,Tl)
            blob.modify(tags=f'{tag}{i}')
            tn.add_tensor(blob)
        else:
            L1.modify(tags=f'{tag}{i}')
            L2.modify(tags=f'{tag}{i}')
            Tl.modify(tags=f'{tag}{i}')
            tn.add_tensor(L1,virtual=True)
            tn.add_tensor(L2,virtual=True)
            tn.add_tensor(Tl,virtual=True)
        if coeff is not None:
            tmp = add(tmp,coeff[i])
    tn.add_tensor(tmp,virtual=True)
    if make_mps:
        tmp.modify(tags=f'{tag}{n}')

        T1 = tn[f'{tag}1']
        Tl,Tr = T1.split((f'{tag}0',),absorb='both',bond_ind=f'{tag}0,1',
                         **compress_opts)
        Tl.modify(tags=f'{tag}0')
        tn.add_tensor(Tl)
        T1.modify(data=Tr.data,inds=Tr.inds)
        tn = compress1D(tn,tag,final='left',**compress_opts)
        tmp.modify(tags={})
    return tn
def compress_field(tni,order,tr,**compress_opts):
    nf = tni.num_tensors-1
    ng = len(tr)
    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1. 
    def get_tnxi(i):
        tnxi = qtn.TensorNetwork([])
        bix1 = tuple(tni[f'v{i}'].bonds(tni[f'v{i-1}']))[0]
        bix2 = tuple(tni[f'v{i}'].bonds(tni[f'v{i+1}']))[0] if i<nf else None
        oix = tuple(set(tni[f'v{i}'].inds).difference({bix1,bix2}))[0]
        for j in range(order):
            T = tni[f'v{i}'].reindex({bix1:f'{bix1}_o{j}'})
            if i<nf:
                T.reindex_({bix2:f'{bix2}_o{j}'})
            if j==0:
                T.reindex_({oix:f'{oix}_o0,1'})
            else:
                t = qtn.Tensor(data=CP,
                               inds=(oix,f'{oix}_o{j-1},{j}',f'{oix}_o{j},{j+1}'))
                T = qtn.tensor_contract(T,t)
            if j==order-1:
                t = qtn.Tensor(data=tr,inds=(f'{oix}_o{j},{j+1}',))
                T = qtn.tensor_contract(T,t)
            T.modify(tags=f'o{j}')
            tnxi.add_tensor(T)
        return tnxi
    tn = get_tnxi(nf)
    for i in range(nf-1,0,-1):
        tn.add_tensor_network(get_tnxi(i))
        for j in range(order):
            tn.contract_tags(f'o{j}',which='any',inplace=True)
        tn.fuse_multibonds_()
        tn = compress1D(tn,'o',**compress_opts)
    for i in range(order):
        bix1 = tuple(tn[f'o{i}'].bonds(tn[f'o{i+1}']))[0] if i<order-1 else None
        bix2 = tuple(tn[f'o{i}'].bonds(tn[f'o{i-1}']))[0] if i>0 else None
        oix = tuple(set(tn[f'o{i}'].inds).difference({bix1,bix2}))[0]
        if i<order-1:
            tn[f'o{i}'].reindex_({bix1:f'o{i},{i+1}'})
            tn[f'o{i+1}'].reindex_({bix1:f'o{i},{i+1}'})
        tn[f'o{i}'].reindex_({oix:f'o{i}'})
    return tn
class Propagator:
    def __init__(self,v0,vg,xs,tau,**compress_opts):
        tn = get_exponent(v0,vg,xs,tau,**compress_opts)
        self.B = get_exponential(tn,**compress_opts)
        self.Bn_backbone = None
        self.inverse_backbone = None
    def init_nstep_backbone(self,nstep,**compress_opts):
        tn = self.B
        tid = tuple(tn._get_tids_from_inds('k',which='any'))[0]
        T = tn.tensor_map[tid].copy()
        self.Bn_backbone = get_backbone(T,nstep,'s',**compress_opts)
        return self.Bn_backbone
    def apply_propagator(self,mo_coeff,full=True):
        # mo_coeff=norb*nocc
        tn = self.Bn_backbone if full else self.B
        self.mo_coeff = mo_coeff

        tid = tuple(tn.ind_map['k'])[0]
        self.Tprop = tn.tensor_map[tid].copy()
        self.Tprop.modify(tags={})
        self.Tdet  = matrix_multiply(self.Tprop.copy(),mo_coeff,  back=True)
        self.Tovlp = matrix_multiply(self.Tdet.copy(), mo_coeff.T,back=False)
    def init_inverse_backbone(self,order,**compress_opts):
        self.order = order

        T = self.Tovlp.copy()
        T = scalar_multiply(T,-1.)
        T = add(T,1.)
        coeff = [1. for i in range(order)]
        self.inverse_backbone = get_backbone(T,order,'o',coeff=coeff,**compress_opts)
        return self.inverse_backbone
    def get_G1_backbone(self,**compress_opts):
        tn = self.inverse_backbone.copy()
        tid = tuple(tn.ind_map['k'])[0]
        T1 = tn.tensor_map[tid]
        T2 = self.Tdet.copy()
        dim = T1.shape[T1.inds.index('p')]

        sCP2 = np.zeros((2,)*3)
        sCP2[0,0,0] = 1./dim
        sCP2[1,1,1] = 1.

        n1 = self.inverse_backbone.num_tensors-1
        n2 = self.Bn_backbone.num_tensors-1

        T1.reindex_({'p':'_','k':'i1'})
        T2.reindex_({f's{n2-1},{n2}':f'o{n1}','q':'_','k':'i2'})
        T2.modify(tags={})

        T = qtn.Tensor(data=sCP2,inds=('i1','i2','k'))
        blob = qtn.tensor_contract(T1,T2,T)
        Tl,Tr = blob.split(left_inds=None,right_inds=('p','q','k'),absorb='both',
                           bond_ind=f'o{n1},{n1+1}',**compress_opts)
        Tr.modify(tags=f'o{n1+1}')
        T1.modify(data=Tl.data,inds=Tl.inds,tags=f'o{n1}') 
        tn.add_tensor(Tr,virtual=True)
        tn = compress1D(tn,'o',final='left',**compress_opts)
        Tr.modify(tags={})
        self.G1_backbone = tn 
        return tn
     
    def contract_nlayer_tn_from_backbone(self,tr,typ,**compress_opts):
        if typ=='inv':
            order = self.order
            backbone = self.inverse_backbone
        elif typ=='rdm1':
            order = 1+self.order
            backbone = self.get_G1_backbone(**compress_opts)
        elif typ=='rdm2':
            order=4*(1+self.order)
        else:
            raise NotImplementedError(f'{typ} contraction not implemented!')
        tnf = compress_field(self.B,order,tr,**compress_opts)
        print(tnf) 
        return
    def get_nstep_tn_from_backbone(self,typ):
        nf = self.B.num_tensors-1
        nstep = self.Bn_backbone.num_tensors-1

        tni = self.B.copy() 
        tid = tuple(tni.ind_map['k'])[0]
        T = tni._pop_tensor(tid)
        oix = tuple(set(T.inds).difference({'p','q','k'}))[0]

        tn = self.Bn_backbone.copy()
        tid = tuple(tn.ind_map['k'])[0]
        T = tn._pop_tensor(tid)
        if typ=='prop':
            tn.add_tensor(self.Tprop)
        elif typ=='det':
            tn.add_tensor(self.Tdet)
        elif typ=='ovlp':
            tn.add_tensor(self.Tovlp)
        else:
            raise NotImplementedError(f'{typ} tn not implemented!')
        for i in range(nstep):
            tni_ = tni.copy()
            tni_.add_tag(f's{i}')
            tni_.reindex_({f'x{j}':f's{i},x{j}' for j in range(1,nf+1)})
            tni_['v1'].reindex_({oix:f's{i}'})
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

    tau = 0.001
    nocc = 2
    mo_coeff = np.random.rand(norb,nocc)

    cutoff = 1e-15
    nstep = 5
    coeff = [1./scipy.math.factorial(i) for i in range(3,-1,-1)]

    cls = Propagator(v0,vg,xs,tau,cutoff=cutoff)
    Bn = cls.init_nstep_backbone(nstep,cutoff=cutoff)
    print('B')
    print(cls.B)
    print('Bn_backbone')
    print(Bn)
    cls.apply_propagator(mo_coeff)
    det  = cls.get_nstep_tn_from_backbone('det')
    for i in range(1,nf+1):
        vec = np.zeros(ng)
        vec[idxs[i]] = 1.
        for j in range(nstep):
            det.add_tensor(qtn.Tensor(data=vec,inds=(f's{j},x{i}',)))
    det = det.contract(output_inds=('p','q','k')).data

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
    print('check Bn[0]=',np.linalg.norm(np.ones(det.shape[:2])-det[:,:,0]))
    print('check poln[1]=',np.linalg.norm(np.dot(poln,mo_coeff)-det[:,:,1]))
    print('check expn[1]=',np.linalg.norm(np.dot(expn,mo_coeff)-det[:,:,1]))
    print()
    # check inverse
    order = 5
    Winv = np.linalg.inv(pol)
    W_ = np.eye(norb)-pol
    Winv_ = W_+np.eye(norb)
    for j in range(order):
        Winv_ = np.dot(W_,Winv_)+np.eye(norb)
        print(f'order={j+2},inv={np.linalg.norm(Winv-Winv_)}')
    inv_back = cls.init_inverse_backbone(order,cutoff=cutoff)
    print('inverse_backbone')
    print(inv_back)
    tr = np.ones(ng)/ng
    cls.contract_nlayer_tn_from_backbone(tr,'rdm1',cutoff=cutoff)
