import numpy as np
import networkx as nx
import quimb.tensor as qtn
class HD:
    def __init__(self,idx,sym):
        self.idx = idx
        self.n = len(idx) // 2
        self.sym = sym
        self.pix = self.n + 1 # dummy

        b1 = []
        for i in range(self.n):
            li,ri = idx[2*i],idx[2*i+1]
            if li==i:
                b1.append(i)
            if ri==i:
                b1.append(i)
        self.b1 = b1
class HDprop:
    def __init__(self,idx,sign,pix):
        self.idx = idx
        self.n = len(idx) // 2
        self.sign = sign
        ne = 0
        for i in range(self.n):
            li,ri = idx[2*i], idx[2*i+1]
            if li==ri:
                ne += 1
        self.ne = ne
        self.pix = pix

        G = nx.MultiDiGraph()
        G.add_nodes_from(range(self.n),op=False) # add all hugenholtz vertices
        G.nodes[pix]['op'] = True
        for i in range(self.n):
            li,ri = idx[2*i],idx[2*i+1]
            G.add_edge(i,li)
            G.add_edge(i,ri)
        self.G = G
def load_diagrams(n,fname='/home/rppeng/code/EDHD/Examples/HugenDiag'):
    # n=order
    out = open(fname+f'{n}.txt','r').readlines()
    for l in out:
        if l[:4]=='Diag':
            str_ = l.split('=')[-1]
            while not str_[0].isnumeric():
                str_ = str_[1:]
            while not str_[-1].isnumeric():
                str_ = str_[:-1] 
            diags = str_.split('},{')
            for ix in range(len(diags)):
                diag = diags[ix].split(',')
                diag = tuple([int(c)-1 for c in diag])
                diags[ix] = diag
        if l[:3]=='Sym': 
            str_ = l.split('=')[-1]
            while not str_[0].isnumeric():
                str_ = str_[1:]
            while not str_[-1].isnumeric():
                str_ = str_[:-1] 
            syms = str_.split(',')
            syms = [int(c) for c in syms]
    return [HD(diags[ix],syms[ix]) for ix in range(len(diags))]
def distribute_diagrams(diags):
    hfc_map = {i:[] for i in range(diags[0].n+1)}
    while len(diags)!=0:
        diag = diags.pop()
        hfc_map[len(diag.b1)].append(diag)
    return hfc_map
def generate_property(diag):
    nhfc = len(diag.b1)
    if nhfc==0:
        return []
    idx = diag.idx
    sign = 1 if diag.sym>0 else -1
    ls = [HDprop(idx,sign,pix) for pix in diag.b1]
    if nhfc==1:
        return ls[0]
    rix = set()
    nm = nx.algorithms.isomorphism.categorical_node_match('op',False)
    for i in range(nhfc):
        for j in range(i+1,nhfc):
            if nx.is_isomorphic(ls[i].G,ls[j].G,node_match=nm):
                rix.add(j)
    lix = list(set(range(nhfc)).difference(rix))
    return [ls[i] for i in lix]
def theta(t):
    return (1.+np.sign(t))/2.
def get_propagator(mo_energies,occ_vec,t1,t2):
    norb = len(mo_energies)
    ng1,ng2 = len(t1),len(t2)
    g = np.zeros((norb,ng1,ng2,2,2),dtype=complex) #t1,t2,a1,a2
    for p in range(norb):
        for i1 in range(ng1):
            for i2 in range(ng2):
                t12 = t1[i1]-t2[i2]
                phase = np.exp(-1j*mo_energies[p]*t12)
                gpm = -1j*phase * (1-occ_vec[p])
                gmp = 1j*phase * occ_vec[p]
                g[p,i1,i2,1,0] = gpm 
                g[p,i1,i2,0,1] = gmp
                g[p,i1,i2,0,0] = theta(t12)*gpm + theta(-t12)*gmp
                g[p,i1,i2,1,1] = theta(-t12)*gpm + theta(t12)*gmp
    symm0 = np.linalg.norm(g[...,0,0]+g[...,1,1]-g[...,0,1]-g[...,1,0])
    symm1 = np.linalg.norm(g[...,0,0]+g[...,1,1].transpose(0,2,1).conj())
    symm2 = np.linalg.norm(g[...,0,1]+g[...,0,1].transpose(0,2,1).conj())
    symm3 = np.linalg.norm(g[...,1,0]+g[...,1,0].transpose(0,2,1).conj())
    if symm0 + symm1 + symm2 + symm3 > 1e-10:
        print(symm0,symm1,symm2,symm3)
    return g
def get_tn(diag,mo_energies,occ_vec,v2,to,ti,w):
    # to: open times
    # ti: integrated times
    gii = get_propagator(mo_energies,occ_vec,ti,ti)
    gio = get_propagator(mo_energies,occ_vec,ti,to)
    goi = get_propagator(mo_energies,occ_vec,to,ti)
    tn = qtn.TensorNetwork([])
    n,pix,idx = diag.n,diag.pix,diag.idx
    def select(oix,iix):
        if oix==pix:
            return 1j*goi
        elif iix==pix:
            return 1j*gio
        else:
            return 1j*gii
    for i in range(n):
        li,ri = idx[2*i],idx[2*i+1]
        if i == pix:
            oix = li if ri==i else ri
            tag = 'p' if ri==i else 'q'
            tn.add_tensor(qtn.Tensor(data=select(oix,i),
                                     inds=(f'{tag}{oix}',f't{oix}',f't{i}',f'a{oix}',f'a{i}')))
            p_,p = f'{tag}{oix}',f'{tag}{i}'
        else:
            tn.add_tensor(qtn.Tensor(data=select(li,i),
                                     inds=(f'p{li}',f't{li}',f't{i}',f'a{li}',f'a{i}')))
            tn.add_tensor(qtn.Tensor(data=select(ri,i),
                                     inds=(f'q{ri}',f't{ri}',f't{i}',f'a{ri}',f'a{i}')))
            tn.add_tensor(qtn.Tensor(data=1j*v2.copy(),
                                     inds=(f'p{li}',f'q{ri}',f'p{i}',f'q{i}')))
    tid1,tid2 = tn.ind_map[f'a{pix}']
    g1,g2 = tn.tensor_map[tid1],tn.tensor_map[tid2]
    if p in g1.inds and p_ in g2.inds:
        g1.reindex_({f'a{pix}':'a',f't{pix}':'t'})
        g2.reindex_({f'a{pix}':'a_',f't{pix}':'t_'})
    elif p in g2.inds and p_ in g1.inds:
        g2.reindex_({f'a{pix}':'a',f't{pix}':'t'})
        g1.reindex_({f'a{pix}':'a_',f't{pix}':'t_'})
    else:
        print(g1)
        print(g2)
        print(p,p_)
        exit()
    for i in range(n):
        if i!=pix:
            tn.add_tensor(qtn.Tensor(data=w.copy(),inds=(f't{i}',)))
            tn.add_tensor(qtn.Tensor(data=np.array([1.,-1.]),inds=(f'a{i}',)))
    return tn,(p,'t','a',p_,'t_','a_')
def _get_tn(diag,mo_energies=None,occ_vec=None,v2=None,v1=None,t=None,w=None):
    norb = len(mo_energies)
    ng = len(t)
    g = np.zeros((norb,ng,ng,2,2),dtype=complex)
    for p in range(norb):
        for i1 in range(ng):
            for i2 in range(ng):
                phase = np.exp(-1j*mo_energies[p]*(t[i1]-t[i2]))
                g[p,i1,i2,1,0] = -1j*phase * (1-occ_vec[p])
                g[p,i1,i2,0,1] = 1j*phase * occ_vec[p]
                if i1>i2:
                    g[p,i1,i2,0,0] = g[p,i1,i2,1,0] 
                    g[p,i1,i2,1,1] = g[p,i1,i2,0,1] 
                elif i2>i1:
                    g[p,i1,i2,0,0] = g[p,i1,i2,0,1]
                    g[p,i1,i2,1,1] = g[p,i1,i2,1,0]
                #else:
                #    g[p,i1,i2,0,0] = g[p,i1,i2,1,1] = (g[p,i1,i2,0,1]+g[p,i1,i2,1,0])/2.
    #print(np.linalg.norm(g[...,0,0]+g[...,1,1]-g[...,0,1]-g[...,1,0]))
    print(np.linalg.norm(g[...,0,0]+g[...,1,1].transpose(0,2,1).conj()))
    print(np.linalg.norm(g[...,0,1]+g[...,0,1].transpose(0,2,1).conj()))
    print(np.linalg.norm(g[...,1,0]+g[...,1,0].transpose(0,2,1).conj()))

    tn = qtn.TensorNetwork([])
    n,pix,idx = diag.n,diag.pix,diag.idx
    for i in range(n):
        li,ri = idx[2*i],idx[2*i+1]
        if li==i:
            tn.add_tensor(qtn.Tensor(data=g.copy(),
                                     inds=(f'q{ri}',f't{ri}',f't{i}',f'a{ri}',f'a{i}')))
            if i == pix:
                output_inds = f'q{ri}',f'q{i}'
                #tn.add_tensor(qtn.Tensor(data=np.array([1.,-1.]),
                #                     inds=(f'a{i}',)))
            else:
                tn.add_tensor(qtn.Tensor(data=v1.copy(),
                                     inds=(f'q{ri}',f'q{i}')))
        elif ri==i:
            tn.add_tensor(qtn.Tensor(data=g.copy(),
                                     inds=(f'p{li}',f't{li}',f't{i}',f'a{li}',f'a{i}')))
            if i == pix:
                output_inds = f'p{li}',f'p{i}'
                #tn.add_tensor(qtn.Tensor(data=np.array([1.,-1.]),
                #                     inds=(f'a{i}',)))
            else:
                tn.add_tensor(qtn.Tensor(data=-1j*v1.copy(),
                                     inds=(f'p{li}',f'p{i}')))
        else:
            tn.add_tensor(qtn.Tensor(data=g.copy(),
                                     inds=(f'p{li}',f't{li}',f't{i}',f'a{li}',f'a{i}')))
            tn.add_tensor(qtn.Tensor(data=g.copy(),
                                     inds=(f'q{ri}',f't{ri}',f't{i}',f'a{ri}',f'a{i}')))
            tn.add_tensor(qtn.Tensor(data=-1j*v2.copy(),
                                     inds=(f'p{li}',f'q{ri}',f'p{i}',f'q{i}')))
    
    output_inds = output_inds + tuple([qtn.rand_uuid() for i in range(2)])
    tid1,tid2 = tn.ind_map[f'a{pix}']
    g1,g2 = tn.tensor_map[tid1],tn.tensor_map[tid2]
    if output_inds[0] in g1.inds:
        assert output_inds[1] in g2.inds
        g1.reindex_({f'a{pix}':output_inds[2]})
        g2.reindex_({f'a{pix}':output_inds[3]})
    elif output_inds[0] in g2.inds:
        assert output_inds[1] in g1.inds
        g2.reindex_({f'a{pix}':output_inds[2]})
        g1.reindex_({f'a{pix}':output_inds[3]})
    else:
        print(g1)
        print(g2)
        print(output_inds)
        exit()
    for i in range(n):
        if i!=pix:
            tn.add_tensor(qtn.Tensor(data=w.copy(),inds=(f't{pix}',f't{i}')))
            tn.add_tensor(qtn.Tensor(data=np.array([1.,-1.]),inds=(f'a{i}',)))
    return tn,(f't{pix}',)+output_inds
