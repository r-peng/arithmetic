import numpy as np
import math
import quimb.tensor as qtn
import quimb.tensor.tensor_core as qtc
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
 
def compress(tn,Lx,Ly,linds,rinds,cutoff=0.0,max_bond=None):
    # tid[i,k] = i*n+k
    N,n = Lx,Ly
    def contr(i,k):
        tid1 = i*n
        tid2 = i*n+k
        t1 = tn._pop_tensor(tid1)
        t2 = tn._pop_tensor(tid2)
        t12 = qtc.tensor_contract(t1,t2,preserve_tensor=True)
        tn.add_tensor(t12,tid=tid1)
        return 
    for k in range(1,n):
        contr(0,k)
        for i in range(1,N):
            contr(i,k)
            tn._compress_between_tids((i-1)*n,i*n,max_bond=max_bond,cutoff=cutoff)
    output_inds = linds+rinds
    out = qtc.tensor_contract(*tn.tensors,output_inds=output_inds,
                              preserve_tensor=True)
    return out
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
