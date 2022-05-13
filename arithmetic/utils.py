import numpy as np
import quimb.tensor as qtn
from scipy.special import roots_legendre
import os,pickle,functools
def permute_1d(peps,Lx):
    Ly = peps.num_tensors
    arrays = []
    for i in range(Ly):
        tag = peps.site_tag(Lx-1,i)
        pind = peps.site_ind(Lx-1,i)
        T = peps[tag]
        if i==0 or i==Ly-1:
            ind = list(set(T.inds)-{pind})[0]
            inds = ind,pind
        else:
            ltag =  peps.site_tag(Lx-1,i-1)
            rtag =  peps.site_tag(Lx-1,i+1)
            lind = list(qtn.bonds(T,peps[ltag]))[0]
            rind = list(qtn.bonds(T,peps[rtag]))[0]
            inds = lind,rind,pind
        T.transpose_(*inds)
        arrays.append(T.data)
    return arrays
def contract_1d(arrays,tr):
    Ly = len(arrays)
    N,d = tr.shape
    out = np.einsum('rp,p->r',arrays[0],tr[0,:])
    for i in range(1,Ly-1):
        out = np.einsum('l,lrp,p->r',out,arrays[i],tr[i,:])
    data = tr[-1,:] if N==Ly else np.array([0.0,1.0])
    return np.einsum('l,lp,p->',out,arrays[-1],data)
def make_peps_with_legs(arrays):
    Lx = len(arrays)
    for i in range(Lx-1):
        row = arrays[i]
        for j in range(len(row)):
            row[j] = np.reshape(row[j],row[j].shape+(1,))
    row = arrays[-1]
    for j in range(len(row)):
        row[j] = np.einsum('...ud->...du',row[j])
    peps = qtn.PEPS(arrays,shape='lrudp')
    for i in range(peps.Lx-1):
        for j in range(peps.Ly):
            T = peps[peps.site_tag(i,j)]
            T.modify(data=T.data[...,0],inds=T.inds[:-1])
    return peps
def trace_open(peps,tr):
    N,d = tr.shape
    for j in range(peps.Ly-1): 
        T = peps[peps.site_tag(peps.Lx-1,j)]
        data = np.einsum('...p,p->...',T.data,tr[j,:])
        T.modify(data=data,inds=T.inds[:-1])
    data = tr[-1,:] if N==peps.Ly else np.array([0.0,1.0])
    T = peps[peps.site_tag(peps.Lx-1,peps.Ly-1)]
    data = np.einsum('...p,p->...',T.data,data)
    T.modify(data=data,inds=T.inds[:-1])
    return peps
def contract_from_bottom(peps,tr=None,**compress_opts):
    Lx,Ly = peps.Lx,peps.Ly
    peps = peps.contract_boundary_from(xrange=None,yrange=None,
           from_which='bottom',**compress_opts)
    arrays = permute_1d(peps,Lx)
    d = arrays[0].shape[-1]
    if tr is not None:
        return contract_1d(arrays,tr)
    return arrays
def contract(peps,from_which=None,**compress_opts):
    if from_which is None:
        return peps.contract_boundary(**compress_opts)
    else:
        peps = peps.contract_boundary_from(xrange=None,yrange=None,
               from_which=from_which,**compress_opts)
        return peps.contract()

def get_fac_ijkl(B,i,j,k,l):
    out  = B[i,j,k,l]+B[i,j,l,k]+B[i,k,j,l]+B[i,k,l,j]
    out += B[i,l,j,k]+B[i,l,k,j]+B[j,i,k,l]+B[j,i,l,k]
    out += B[j,k,i,l]+B[j,k,l,i]+B[j,l,i,k]+B[j,l,k,i]
    out += B[k,i,j,l]+B[k,i,l,j]+B[k,j,i,l]+B[k,j,l,i]
    out += B[k,l,i,j]+B[k,l,j,i]+B[l,i,j,k]+B[l,i,k,j]
    out += B[l,j,i,k]+B[l,j,k,i]+B[l,k,i,j]+B[l,k,j,i]
    return out
def get_fac_iijk(B,i,j,k):
    out  = B[i,i,j,k]+B[i,i,k,j]+B[i,j,i,k]+B[i,k,i,j]
    out += B[i,j,k,i]+B[i,k,j,i]+B[j,i,i,k]+B[k,i,i,j]
    out += B[j,i,k,i]+B[k,i,j,i]+B[j,k,i,i]+B[k,j,i,i]
    return out
def get_fac_iijj(B,i,j):
    return B[i,i,j,j]+B[i,j,i,j]+B[i,j,j,i]+B[j,i,i,j]+B[j,i,j,i]+B[j,j,i,i]
def get_fac_iiij(B,i,j):
    return B[i,i,i,j]+B[i,i,j,i]+B[i,j,i,i]+B[j,i,i,i]

def quad(a,b,n):
    x,w = roots_legendre(n)
    s,m = (b-a)/2.0,(a+b)/2.0
    w *= s
    x = np.array([xi*s+m for xi in x])
    return x,w

def delete_tn_from_disc(fname):
    try:
        os.remove(fname)
    except:
        pass
def load_tn_from_disc(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    tn = qtn.TensorNetwork([])
    for tid,ten in data['tensors'].items():
        T = qtn.Tensor(ten.data, inds=ten.inds, tags=ten.tags)
        tn.add_tensor(T,tid=tid,virtual=True)
    extra_props = dict()
    for name,prop in data['tn_info'].items():
        extra_props[name[1:]] = prop
    tn.exponent = data['exponent']
    tn = tn.view_as_(data['class'], **extra_props)
    return tn
def write_tn_to_disc(tn,fname):
    data = dict()
    data['class'] = type(tn)
    data['tensors'] = dict()
    for tid,T in tn.tensor_map.items():
        data['tensors'][tid] = T 
    data['tn_info'] = dict()
    for e in tn._EXTRA_PROPS:
        data['tn_info'][e] = getattr(tn, e)
    data['exponent'] = tn.exponent
    with open(fname, 'wb') as f:
        pickle.dump(data, f)
    return 
def compress_simplify_gauge(
    tn,
    compress_simplify_opts = {
    'output_inds':[],
    'atol':1e-15,
    'simplify_sequence_a':'ADCRS',
    'simplify_sequence_b':'RPL',
    'hyperind_resolve_mode':'tree',
    'hyperind_resolve_sort':'clustering',
    'final_resolve':True,
    'max_simplification_iterations':500,
    'converged_tol':1e-6,
    'equalize_norms':True,
    'progbar':False},
    gauge_opts={'max_iterations':500,'tol':1e-6},
    max_iter = 10,
):
    thresh = 1e-6
    tn.compress_simplify_(**compress_simplify_opts)
    for i in range(max_iter):
        nv,ne = tn.num_tensors,tn.num_indices
        tn.gauge_all_simple_(**gauge_opts)
        tn.compress_simplify_(**compress_simplify_opts)
        if ((tn.num_tensors==1) or
            (tn.num_tensors > (1.0-thresh)*nv and 
             tn.num_indices > (1.0-thresh)*ne)):
            break
    tn.gauge_all_simple_(**gauge_opts)
    return tn
