import numpy as np
import math
import quimb.tensor as qtn
import cotengra as ctg

def _Ry(x): # exp(-i*x*Y)
    U = np.eye(2)*math.cos(x)
    U -= iY*math.sin(x)
    return U
def _Rys(xs): # init
    d = len(xs) 
    U = np.zeros((d,2,2))
    for i in range(d):
        U[i,...] = _Ry(xs[i])
    return U
def _C1(op): # ccio
    U = np.zeros((2,)*4)
    U[0,0,...] = np.eye(2)
    U[1,1,...] = op.copy()
    return U
def _C2(op): # c1c1,c2c2,io
    U = np.zeros((2,)*6)
    U[0,0,0,0,...] = np.eye(2)
    U[0,0,1,1,...] = np.eye(2)
    U[1,1,0,0,...] = np.eye(2)
    U[1,1,1,1,...] = op.copy()
    return U 
iY = np.array([[0.0,1.0],[-1.0,0.0]])
NOT = np.array([[0.0,1.0],[1.0,0.0]]) 
ZERO = np.array([1.0,0.0])
H = np.array([[1.0,1.0],[1.0,-1.0]])/math.sqrt(2)
CNOT = _C1(NOT)
GHZ_inv = np.einsum('ijkl,jm->imkl',CNOT,H) # i1o1,i2o2

def _gb(R,C=None,contr_ancilla=True):
    # Rx,io: rotation on aux
    # C: controled op, default = -iY
    C = _C1(-iY) if C is None else C
    n = len(R.shape)-2
    state_inds = ['{},'.format(i+1) for i in range(n)]
    ls = []
    inds = tuple(state_inds+['ai','ci'])
    ls.append(qtn.Tensor(data=R,inds=inds,tags='R'))
    inds = tuple(state_inds+['ao','co'])
    ls.append(qtn.Tensor(data=R,inds=inds,tags='R'))
    inds = 'ci','co','i','o'
    ls.append(qtn.Tensor(data=C,inds=inds,tags='C'))
    if contr_ancilla:
        ls.append(qtn.Tensor(data=ZERO,inds=('ai',),tags='ZERO'))
        ls.append(qtn.Tensor(data=ZERO,inds=('ao',),tags='ZERO'))
        output_inds = state_inds+['i','o']
    else: 
        output_inds = state_inds+['ai','ao','i','o']
    TN = qtn.TensorNetwork(ls)
#    optimize = ctg.HyperOptimizer(max_repeats=64,parallel='ray',reconf_opts={})
    optimize='auto'
    out = TN.contract(optimize=optimize,output_inds=output_inds)
    return out.data
def _par(R,w,C=None,contr_ancilla=True):
    C = _C2(-iY) if C is None else C
    n = len(R.shape)-2
    state_inds = ['{},'.format(i+1) for i in range(n)]
    ls = []
    inds = tuple(state_inds+['a1i','c1i'])
    ls.append(qtn.Tensor(data=R,inds=inds,tags='R'))
    inds = 'a2i','c2i'
    ls.append(qtn.Tensor(data=_Ry(w),inds=inds,tags='Rw'))
    inds = 'c1i','c1o','c2i','c2o','i','o'
    ls.append(qtn.Tensor(data=C,inds=inds,tags='C'))
    inds = 'c1o','a1o','c2o','a2o'
    ls.append(qtn.Tensor(data=GHZ_inv,inds=inds,tags='GHZ_inv'))
    if contr_ancilla: 
        ls.append(qtn.Tensor(data=ZERO,inds=('a1i',),tags=ZERO))
        ls.append(qtn.Tensor(data=ZERO,inds=('a1o',),tags=ZERO))
        ls.append(qtn.Tensor(data=ZERO,inds=('a2i',),tags=ZERO))
        ls.append(qtn.Tensor(data=ZERO,inds=('a2o',),tags=ZERO))
        output_inds = state_inds+['i','o']
    else: 
        output_inds = state_inds+['a1i','a1o','a2i','a2o','i','o']
    TN = qtn.TensorNetwork(ls)
#    optimize = ctg.HyperOptimizer(max_repeats=64,parallel='ray',reconf_opts={})
    optimize='auto'
    out = TN.contract(optimize=optimize,output_inds=output_inds)
    return out.data
def _par_(R,w,C=None):
    C = _C2(-iY) if C is None else C
    n = len(R.shape)-2
    aux1 = np.einsum('...ij,i->...j',R,ZERO)
    aux2 = np.einsum('ij,i->j',_Ry(w),ZERO)
    out = np.einsum('klmnio,ln->kmio',C,np.eye(2)/math.sqrt(2))
    out = np.einsum('kmio,m->kio',out,aux2)
    out = np.einsum('kio,...k->...io',out,aux1)
    return out
def _add(R1,R2):
    n1 = len(R1.shape)-2
    n2 = len(R2.shape)-2
    state_inds1 = ['i{},'.format(i+1) for i in range(n1)]
    state_inds2 = ['j{},'.format(i+1) for i in range(n2)]
    ls = []
    inds = tuple(state_inds1+['i','j'])
    ls.append(qtn.Tensor(data=R1,inds=inds,tags='R1'))
    inds = tuple(state_inds2+['j','o'])
    ls.append(qtn.Tensor(data=R2,inds=inds,tags='R2'))
    TN = qtn.TensorNetwork(ls)
#    optimize = ctg.HyperOptimizer(max_repeats=64,parallel='ray',reconf_opts={})
    optimize='auto'
    output_inds = state_inds1+state_inds2+['i','o']
    out = TN.contract(optimize=optimize,output_inds=output_inds)
    return out.data
def _add_hidden(R1,R2):
    assert len(R1.shape)==len(R2.shape)
    n = len(R1.shape)-2
    state_inds = ['{},'.format(i+1) for i in range(n)]
    ls = []
    inds = tuple(state_inds+['i','j'])
    ls.append(qtn.Tensor(data=R1,inds=inds,tags='R1'))
    inds = tuple(state_inds+['j','o'])
    ls.append(qtn.Tensor(data=R2,inds=inds,tags='R2'))
    TN = qtn.TensorNetwork(ls)
#    optimize = ctg.HyperOptimizer(max_repeats=64,parallel='ray',reconf_opts={})
    optimize='auto'
    output_inds = state_inds+['i','o']
    out = TN.contract(optimize=optimize,output_inds=output_inds)
    return out.data

