import numpy as np
import scipy
import quimb.tensor as qtn
import functools
np.set_printoptions(suppress=True,precision=6,linewidth=1000)
def get_Hmc(h1,eri,tau,method='eigh'):
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
    H = [-tau*v0]+[np.sqrt(tau)*1j*L[:,:,i] for i in range(L.shape[-1])]
    return H
def get_Hmc_Hubbard(t,U,sites,pairs,tau):
    nsites = len(sites)
    v0 = np.zeros((nsites,)*2)
    for (site1,site2) in pairs:
        i = sites.index(site1)
        j = sites.index(site2)
        v0[i,j] = -t
        v0[j,i] = -t
    for i in range(nsites):
        v0[i,i] = U/2.
    H = [-tau*v0]

    b = np.arccosh(np.exp(tau*U/2.)) 
    for i in range(nsites):
        vi = np.zeros((nsites,)*2) 
        vi[i,i] = b
        H.append(vi)
    return H
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
    dim = dim1 if back else dim2 

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

def get_exponent(H,xs,**compress_opts):
    norb = H[0].shape[0]
    nf = len(H)-1
    ng = len(xs)

    CP = np.zeros((norb,)*3)
    for i in range(norb):
        CP[i,i,i] = 1.
    tn = qtn.TensorNetwork([])

    data = np.ones((norb,norb,2))
    data[:,:,1] = H[0].copy() 
    data = np.einsum('PQi,Ppr,Qqs,ijk->pqkrsj',data,CP,CP,ADD)
    inds = 'p','q','k','x0,1','x0,1_','x0,1__'
    tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags={'x0'}))
    for i in range(1,nf+1):
        data = np.zeros((2,norb,norb,2),dtype=H[i].dtype)
        data[1,:,:,1] = H[i].copy()
        data[0,:,:,0] = np.ones((norb,)*2)
        inds = (f'x{i}_',) + inds[-3:]
        if i<nf:
            data = np.einsum('xPQi,Ppr,Qqs,ijk->xpqkrsj',data,CP,CP,ADD)
            inds = inds + (f'x{i},{i+1}',f'x{i},{i+1}_',f'x{i},{i+1}__')
        tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags={f'x{i}'}))
    tn.fuse_multibonds_()
    tn = compress1D(tn,'x',final='right',**compress_opts)
    for i in range(1,nf+1):
        data = np.ones((ng,2))
        data[:,1] = xs.copy()
        inds = f'x{i}',f'x{i}_'
        tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags={f'x{i}'}))
        tn.contract_tags(f'x{i}',which='any',inplace=True)
    return tn
def get_exponential(tni,n=3,**compress_opts):
    # tni = -tau*v0+sqrt(tau)*sum(xi*vi) 
    nf = tni.num_tensors-1
    norb = tni['x0'].shape[tni['x0'].inds.index('p')]
    ng = tni['x1'].shape[tni['x1'].inds.index('x1')]
    coeff = [1./np.math.factorial(i) for i in range(n+1)]

    sCP2 = np.zeros((2,)*3)
    sCP2[0,0,0] = 1./norb
    sCP2[1,1,1] = 1.
    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1.

    tn = tni.copy()
    tn = scalar_multiply(tn,coeff[n])
    tn = add(tn,coeff[n-1])
    for i in range(n-2,-1,-1):
        T1 = tn['x0'].reindex_({'p':'_','k':'i1'})
        T2 = tni['x0'].reindex({'q':'_','k':'i2','x0,1':'x0,1_'})
        T = qtn.tensor_contract(T1,T2,qtn.Tensor(data=sCP2,inds=('i1','i2','k')))
        T1.modify(data=T.data,inds=T.inds)
        tn = add(tn,coeff[i])
        for j in range(1,nf+1):
            T1 = tn[f'x{j}'].reindex_({f'x{j}':'a'})
            T2 = tni[f'x{j}'].reindex({f'x{j}':'b',f'x{j-1},{j}':f'x{j-1},{j}_',
                                       f'x{j},{j+1}':f'x{j},{j+1}_'})
            T = qtn.tensor_contract(T1,T2,qtn.Tensor(data=CP,inds=('a','b',f'x{j}')))
            T1.modify(data=T.data,inds=T.inds)
        tn.fuse_multibonds_()
        tn = compress1D(tn,'x',final='right',**compress_opts)
        tn = scale(tn)
    return tn
def get_backbone(T,coeff,tag,**compress_opts):
    n = len(coeff)-1
    dim = T.shape[T.inds.index('p')]
    oix = tuple(set(T.inds).difference({'p','q','k'}))[0]

    sCP2 = np.zeros((2,)*3)
    sCP2[0,0,0] = 1./dim
    sCP2[1,1,1] = 1.

    tn = qtn.TensorNetwork([])
    T1 = T.copy()
    T1.modify(tags=f'{tag}0')
    T1 = scalar_multiply(T1,coeff[n])
    T1 = add(T1,coeff[n-1])
    for i in range(n-2,-1,-1):
        T1.reindex_({'p':'_','k':'i1'})
        T2 = T.reindex({'q':'_','k':'i2',oix:f'{tag}{i+1}'})
        T2.modify(tags={})
        blob = qtn.tensor_contract(T1,T2,qtn.Tensor(data=sCP2,inds=('i1','i2','k')))
        T1,S,T2 = blob.split(left_inds=('p','q','k'),absorb=None,
                          bond_ind=f'{tag}{i},{i+1}',**compress_opts)
        fac = np.amax(S.data)
        T1.modify(data=np.einsum('...s,s->...s',T1.data,S.data/fac))
        tn.exponent += np.log10(fac)
        T2.modify(tags=f'{tag}{i+1}')
        tn.add_tensor(T2)
        T1 = add(T1,coeff[i])
    tn.add_tensor(T1,virtual=True)

    tn[f'{tag}{n-1}'].reindex_({oix:f'{tag}{n}'})
    t0,t1 = tn[f'{tag}{n-1}'].split((f'{tag}{n}',),absorb='both',
                                    bond_ind=f'{tag}{n-1},{n}',**compress_opts)
    tn[f'{tag}{n-1}'].modify(data=t1.data,inds=t1.inds)
    t0.modify(tags=f'{tag}{n}')
    tn.add_tensor(t0)

    tn = compress1D(tn,tag,final='right',**compress_opts)
    return tn
###########################################################################
# compression strategy 1
###########################################################################
def trace_field(tni,nl,tr,**compress_opts):
    nf = tni.num_tensors-1
    fac = 10**(tni.exponent/nf)
    ng = tni['x1'].shape[tni['x1'].inds.index('x1')]
    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1. 
    def get_col(i):
        col = qtn.TensorNetwork([])
        for j in range(nl):
            T = tni[f'x{i}'].reindex({f'x{i-1},{i}':f'x{i-1},{i}_l{j}',
                                      f'x{i},{i+1}':f'x{i},{i+1}_l{j}'})
            if j==0:
                T.reindex_({f'x{i}':f'x{i}_l0,1'})
            else:
                t = qtn.Tensor(data=CP,
                               inds=(f'x{i}',f'x{i}_l{j-1},{j}',f'x{i}_l{j},{j+1}'))
                T = qtn.tensor_contract(T,t)
            if j==nl-1:
                t = qtn.Tensor(data=tr[i],inds=(f'x{i}_l{j},{j+1}',))
                T = qtn.tensor_contract(T,t)
            T.modify(data=T.data*fac,tags=f'l{j}')
            col.add_tensor(T)
        return col
    tn = get_col(nf)
    for i in range(nf-1,0,-1):
        tn.add_tensor_network(get_col(i))
        for j in range(nl):
            tn.contract_tags(f'l{j}',which='any',inplace=True)
        tn.fuse_multibonds_()
        try:
            tn = compress1D(tn,'l',**compress_opts)
        except ValueError:
            tn = tn
    for i in range(nl):
        tn[f'l{i}'].reindex_({f'x{nf}_l{i-1},{i}':f'l{i-1},{i}',
                              f'x{nf}_l{i},{i+1}':f'l{i},{i+1}',
                              f'x0,1_l{i}':f'l{i}'})
    return tn
def compress_step(tnxi,tnsi,**compress_opts):
    nl = tnxi.num_tensors
    ns = tnsi.num_tensors-1
    fac = 10**(tnsi.exponent/ns)
    def get_col(i):
        col = qtn.TensorNetwork([])
        for j in range(nl):
            T = tnsi[f's{i}'].reindex({f's{i},{i+1}':f's{i},{i+1}_l{j}',
                                       f's{i-1},{i}':f's{i-1},{i}_l{j}',
                                       f's{i}':f'l{j}'})
            t = tnxi[f'l{j}'].reindex({f'l{j-1},{j}':f's{i}_l{j-1},{j}',
                                       f'l{j},{j+1}':f's{i}_l{j},{j+1}'})
            T = qtn.tensor_contract(T,t)
            T.modify(data=T.data*fac,tags=f'l{j}')
            col.add_tensor(T)
        return col
    tn = get_col(ns)
    for i in range(ns-1,0,-1):
        tn.add_tensor_network(get_col(i))
        for j in range(nl):
            tn.contract_tags(f'l{j}',which='any',inplace=True)
        tn.fuse_multibonds_()
        try:
            tn = compress1D(tn,'l',**compress_opts)
        except ValueError:
            tn = tn
    for i in range(nl):
        tn[f'l{i}'].reindex_({f's{ns}_l{i-1},{i}':f'l{i-1},{i}',
                              f's{ns}_l{i},{i+1}':f'l{i},{i+1}',
                              f's0,1_l{i}':f'l{i}'})
    return tn
###########################################################################
# compression strategy 2
###########################################################################
def add_layer(tn,tni,tag,data):
    n = tni.num_tensors-1
    start,stop,step = (n,0,-1) if tag=='x' else (0,n,1)
    for j in range(start,stop,step):
        T1 = tn[f'{tag}{j}']
        T2 = tni[f'{tag}{j}'].copy()
        T1.reindex_({f'{tag}{j}':f'a'})
        T2.reindex_({f'{tag}{j}':f'b'})
        T2.reindex_({f'{tag}{j-1},{j}':f'{tag}{j-1},{j}_'})
        T2.reindex_({f'{tag}{j},{j+1}':f'{tag}{j},{j+1}_'})
        T = qtn.tensor_contract(T1,T2,qtn.Tensor(data=data,inds=('a','b',f'{tag}{j}')))
        T1.modify(data=T.data,inds=T.inds)
    return tn
def compress_nlayer(tnxi,tnsi,inv_back,**compress_opts):
    ns = tnsi.num_tensors-1
    nl = inv_back.num_tensors-1
    ng = tnxi['x1'].shape[tnxi['x1'].inds.index('x1')]
    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1.

    tns = tnsi.copy()
    tns._pop_tensor(tuple(tns.ind_map['k'])[0])
    T = inv_back[f'o0'].copy()
    T.reindex_({f'o0':f's{ns-1},{ns}'})
    T.modify(tags=f's{ns}')
    tns.add_tensor(T)

    tnx = tnxi.copy()
    tnx._pop_tensor(tuple(tnx.ind_map['k'])[0])
    for i in range(1,nl):
        # compress tnx
        tnx = add_layer(tnx,tnxi,'x',CP)
        tnx.fuse_multibonds_()
        tnx['x1'].reindex_({'x0,1':'a','x0,1_':'b'})
        t0,t1 = tnx['x1'].split(left_inds=('a','b'),absorb='left',
                                bond_ind='x0,1',**compress_opts)
        tnx['x1'].modify(data=t1.data,inds=t1.inds)
        t0.modify(tags='x0')
        tnx.add_tensor(t0) 
        tnx = compress1D(tnx,'x',final='right',**compress_opts)
        t0 = tnx._pop_tensor(tuple(tnx.ind_map['a'])[0])
        t0.transpose_('a','b','x0,1')
        # compress tns
        tns = add_layer(tns,tnsi,'s',t0.data)
        T2 = inv_back[f'o{i}'].reindex({f'o{i}':f's{ns-1},{ns}_'})
        T = qtn.tensor_contract(tns[f's{ns}'],T2)
        tns[f's{ns}'].modify(data=T.data,inds=T.inds)
        tns.fuse_multibonds_()
        tns = compress1D(tns,'s',final='right',**compress_opts)
    T = inv_back.tensor_map[tuple(inv_map.ind_map['k'])[0]]
    T = qtn.tensor_contract(tns[f's{ns}'],T)
    tns[f's{ns}'].modify(data=T.data,inds=T.inds)
    return tnx,tns
            
class Propagator:
    def __init__(self,H,xs,order=5,fac=1.0,**compress_opts):
        tn = get_exponent(H,xs,**compress_opts)
        self.B = get_exponential(tn,n=order,**compress_opts)
        self.B = scalar_multiply(self.B,fac)
        self.B = scale(self.B)
        self.Bn_backbone = None
        self.inverse_backbone = None
    def apply_propagator(self,mo_coeff,full=True):
        # mo_coeff=norb*nocc
        tn = self.Bn_backbone if full else self.B
        tag = 's' if full else 'x'
        self.mo_coeff = mo_coeff

        self.Tprop = tn[f'{tag}0'].copy()
        self.Tdet  = matrix_multiply(self.Tprop.copy(),mo_coeff,  back=True)
        self.Tovlp = matrix_multiply(self.Tdet.copy(), mo_coeff.T,back=False)
    def init_nstep_backbone(self,nstep,**compress_opts):
        T = self.B['x0'].copy()
        coeff = [0. for i in range(nstep)] + [1.]
        self.Bn_backbone = get_backbone(T,coeff,'s',**compress_opts)
        return self.Bn_backbone
    def init_inverse_backbone(self,order=3,phi=None,**compress_opts):
        T = self.Tovlp.copy()
        phi = np.eye(T.shape[T.inds.index('p')]) if phi is None else phi
        print(phi)
        T = matrix_multiply(T,-phi,back=False)
        T = add(T,1.)
        coeff = [1. for i in range(order+1)]
        self.inverse_backbone = get_backbone(T,coeff,'o',**compress_opts)
        self.inverse_backbone = matrix_multiply(self.inverse_backbone,phi,back=True)
        return self.inverse_backbone
    def trace_field(self,tr):
        tn = self.B.copy()
        nf = tn.num_tensors-1
        fac = 10**(tn.exponent/nf)
        for i in range(1,nf+1):
            tn.add_tensor(qtn.Tensor(data=tr[i]*fac,inds=(f'x{i}',),tags=f'x{i}'))
            tn.contract_tags(f'x{i}',which='any',inplace=True)
        for i in range(nf,1,-1):
            tn.contract_tags((f'x{i}',f'x{i-1}'),which='any',inplace=True)
        return tn
    def trace_nstep_tn(self,tr,typ):
        ns = self.Bn_backbone.num_tensors-1

        tni = self.trace_field(tr) 

        tn = self.Bn_backbone.copy()
        fac = 10**(tn.exponent/nstep)
        if typ=='prop':
            tn[f's0'].modify(data=self.Tprop.data,inds=self.Tprop.inds)
        elif typ=='det':
            tn[f's0'].modify(data=self.Tdet.data, inds=self.Tdet.inds)
        elif typ=='ovlp':
            tn[f's0'].modify(data=self.Tovlp.data,inds=self.Tovlp.inds)
        else:
            raise NotImplementedError(f'{typ} tn not implemented!')
        for i in range(1,ns+1):
            ti = tni['x1'].reindex({f'x0,1':f's{i}'})
            ti.modify(data=ti.data*fac,tags=f's{i}')
            tn.add_tensor(ti)
            tn.contract_tags(f's{i}',which='any',inplace=True)
        for i in range(ns,1,-1):
            tn.contract_tags((f's{i}',f's{i-1}'),which='any',inplace=True)
        return tn.contract(output_inds=('p','q','k')).data
    def compute_inverse(self,tr,**compress_opts):
        #tnx,tns = compress_nlayer(self.B,self.Bn_backbone,self.inverse_backbone,
        #                          **compress_opts)  
        nl = self.inverse_backbone.num_tensors-1
        tnxi = trace_field(self.B,nl,tr,**compress_opts)
        tn = compress_step(tnxi,self.Bn_backbone,**compress_opts)
        fac = 10**(self.inverse_backbone.exponent/nl)
        tn.add_tensor_network(self.inverse_backbone)
        for i in range(nl-1,-1,-1):
            tn[f'o{i+1}'].reindex_({f'o{i+1}':f'l{i}'})
            tn[f'o{i+1}'].modify(data=tn[f'o{i+1}'].data*fac)
            tn.contract_tags((f'l{i}',f'o{i+1}'),which='any',inplace=True)
        for i in range(nl-1,1,-1):
            tn.contract_tags((f'l{i}',f'l{i-1}'),which='any',inplace=True)
        return tn.contract(output_inds=('p','q','k')).data
    def compute_rdm1(self,tr,**compress_opts):
        #tnx,tns = compress_nlayer(self.B,self.Bn_backbone,self.inverse_backbone,
        #                          **compress_opts)  
        nl = self.inverse_backbone.num_tensors
        tnxi = trace_field(self.B,nl,tr,**compress_opts)
        tn = compress_step(tnxi,self.Bn_backbone,**compress_opts)
        fac = 10**(self.inverse_backbone.exponent/(nl-1))
        for i in range(nl-1,0,-1):
            ti = self.inverse_backbone[f'o{i}'].reindex({f'o{i}':f'l{i}'})
            ti.modify(data=ti.data*fac)
            tn.add_tensor(ti)
            tn.contract_tags((f'l{i}',f'o{i}'),which='any',inplace=True)
        for i in range(nl-1,1,-1):
            tn.contract_tags((f'l{i}',f'l{i-1}'),which='any',inplace=True)

        T1 = self.Tdet.reindex({'q':'_','k':'i1','s0,1':'l0'})
        tn.add_tensor(T1)
        T2 = self.inverse_backbone['o0'].copy()
        T2 = matrix_multiply(T2,self.mo_coeff.T,back=True) 
        T2.reindex_({'p':'_','k':'i2'})
        tn.add_tensor(T2)

        dim = T2.shape[T2.inds.index('_')]
        sCP2 = np.zeros((2,)*3)
        sCP2[0,0,0] = 1./dim
        sCP2[1,1,1] = 1.
        tn.add_tensor(qtn.Tensor(data=sCP2,inds=('i1','i2','k')))        
        return tn.contract(output_inds=('p','q','k')).data

if __name__=='__main__':
    nsites = 3
    norb,nocc = nsites,2
    sites = [i for i in range(nsites)]
    pairs = [(i,i+1) for i in range(nsites-1)] 
    t = 1.
    u = 1.
    from pyscf import gto,scf,ao2mo
    mol = gto.M()
    mol.nelectron = 2*nocc
    h1 = np.zeros((nsites,)*2)
    for (site1,site2) in pairs:
        i = sites.index(site1)
        j = sites.index(site2)
        h1[i,j] = -t
        h1[j,i] = -t
    eri = np.zeros((nsites,)*4)
    for i in range(nsites):
        eri[i,i,i,i] = u
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(nsites)
    # ao2mo.restore(8, eri, n) to get 8-fold permutation symmetry of the integrals
    # ._eri only supports two-electron integrals with 4-fold or 8-fold symmetry.
    mf._eri = ao2mo.restore(8, eri, nsites)
    mf.init_guess = '1e'
    mf.kernel()
    mo_coeff = mf.mo_coeff[:,:nocc]
    E0 = mf.e_tot
    print(mf.make_rdm1())
    print(np.dot(mo_coeff,mo_coeff.T))

    tau = 0.01
    xs = np.array([1.,-1.])
    H = get_Hmc_Hubbard(t,u,sites,pairs,tau)

    nf = len(H)-1
    ng = len(xs)
#    idxs = {i:np.random.randint(low=0,high=ng) for i in range(1,nf+1)}
    idxs = {1:1,2:1,3:1}
    tr = {i:np.zeros(ng) for i in range(1,nf+1)}
    for i in range(1,nf+1):
        tr[i][idxs[i]] = 1.

    cutoff = 1e-15
    tn = get_exponent(H,xs,cutoff=cutoff)
    for i in range(1,nf+1):
        tn.add_tensor(qtn.Tensor(data=tr[i],inds=(f'x{i}',)))
    out = tn.contract(output_inds=('p','q','k')).data
    A = H[0].copy()
    for i in range(1,nf+1):
        A += xs[idxs[i]]*H[i].copy()
    print(f'max_bond(A)={tn.max_bond()},exponent={tn.exponent}')
    print('check A[0]=',np.linalg.norm(np.ones_like(out[:,:,0])-out[:,:,0]))
    print('check A[1]=',np.linalg.norm(A-out[:,:,1]))
    print()

    n = 10
    fac = np.exp(tau*E0)
    cls = Propagator(H,xs,order=n,fac=fac,cutoff=cutoff)
    coeff = [1./np.math.factorial(i) for i in range(n+1)]
    print(f'max_bond(B)={cls.B.max_bond()},exponent={cls.B.exponent}')
    tn = cls.trace_field(tr)
    out = tn.contract(output_inds=('p','q','k')).data 
    exp = scipy.linalg.expm(A) 
    pol = A*coeff[n]+np.eye(norb)*coeff[n-1]
    for i in range(n-2,-1,-1):
        pol = np.einsum('pr,rq->pq',A,pol)+coeff[i]*np.eye(norb)
    exp *= fac
    pol *= fac
    print('check B[0]=',np.linalg.norm(np.ones_like(out[:,:,0])-out[:,:,0]))
    print('check pol[1]=',np.linalg.norm(pol-out[:,:,1]))
    print('check exp[1]=',np.linalg.norm(exp-out[:,:,1]))
    print()

    nstep = 10
    cls.init_nstep_backbone(nstep,cutoff=cutoff)
    print(f'max_bond(Bn)={cls.Bn_backbone.max_bond()},exponent={cls.Bn_backbone.exponent}')
    cls.apply_propagator(mo_coeff)
    out = cls.trace_nstep_tn(tr,'det')

    expn = np.eye(norb)
    poln = np.eye(norb)
    for i in range(nstep):
        expn = np.dot(exp,expn)
        poln = np.dot(pol,poln)
    print('check Bn[0]=',np.linalg.norm(np.ones_like(out[:,:,0])-out[:,:,0]))
    print('check poln[1]=',np.linalg.norm(np.dot(poln,mo_coeff)-out[:,:,1]))
    print('check expn[1]=',np.linalg.norm(np.dot(expn,mo_coeff)-out[:,:,1]))
    print()

    # check inverse
    n = 5
    print(poln)
    W = np.linalg.multi_dot([mo_coeff.T,poln,mo_coeff])
    print(idxs)
    print(W)
    w,v = np.linalg.eig(W)
    print(w)
    Winv = np.linalg.inv(W)
    phi = np.eye(nocc)#/nocc
    phi = np.linalg.multi_dot([mo_coeff.T,np.linalg.inv(poln),mo_coeff])
    W_ = np.eye(nocc)-np.dot(phi,W)
    w_,v = np.linalg.eig(W_)
    print(w_)
    Winv_ = W_+np.eye(nocc)
    for j in range(2,n+1):
        Winv_ = np.dot(W_,Winv_)+np.eye(nocc)
        print(f'order={j},inv={np.linalg.norm(Winv-np.dot(Winv_,phi))}')
    print(Winv_)
    #coeff = get_cheb_coeff(lambda x:1./x,n,a=0.1,b=1.)
    #Winv_ = W*coeff[n]+np.eye(nocc)*coeff[n-1]
    #for i in range(n-2,-1,-1):
    #    Winv_ = np.einsum('pr,rq->pq',W,Winv_)+coeff[i]*np.eye(nocc)
    #    print(f'order={n-i},inv={np.linalg.norm(Winv-np.dot(Winv_,phi))}')
    Winv_ = np.dot(Winv_,phi)
    print(Winv)
    exit()
    cls.init_inverse_backbone(order=n,phi=phi,cutoff=cutoff)
    print(f'max_bond(inv)={cls.inverse_backbone.max_bond()},exponent={cls.inverse_backbone.exponent}')
    out = cls.compute_inverse(tr,cutoff=cutoff)
    print('check inv[0]=',np.linalg.norm(np.ones_like(out[:,:,0])-out[:,:,0]))
    print('check inv[1]=',np.linalg.norm(Winv_-out[:,:,1]))

    out = cls.compute_rdm1(tr,cutoff=cutoff)
    rdm1 = np.linalg.multi_dot([poln,mo_coeff,Winv_,mo_coeff.T])
    print('check rdm1[0]=',np.linalg.norm(np.ones_like(out[:,:,0])-out[:,:,0]))
    print('check rdm1[1]=',np.linalg.norm(rdm1-out[:,:,1]))

    W = np.linalg.multi_dot([mo_coeff.T,expn,mo_coeff])
    Winv = np.linalg.inv(W)
    rdm1 = np.linalg.multi_dot([expn,mo_coeff,Winv,mo_coeff.T])
    print('check rdm1[1]=',np.linalg.norm(rdm1-out[:,:,1]))
    print(rdm1)
