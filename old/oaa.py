import numpy as np
import math
import quimb.tensor as qtn
import cotengra as ctg
ZERO = np.array([1.0,0.0])
def _S1(a):
    U = np.eye(2,dtype=complex)
    U[0,0] = np.exp(1j*a)
    return U
def _S2(a): #i1o1,i2o2
    U = np.zeros((2,)*4,dtype=complex)
    U[0,0,0,0] = np.exp(1j*a)
    U[0,0,1,1] = 1.0
    U[1,1,0,0] = 1.0
    U[1,1,1,1] = 1.0
    return U
def _G1(a1,a2,A):
    # Ax,io
    S1 = _S1(a1)
    S2 = _S1(a2)
    n = len(A.shape)-4
    state_inds = ['{},'.format(i+1) for i in range(n)]
    ls = []
    inds = 'ai','to'
    ls.append(qtn.Tensor(data=S1,inds=inds,tags='S1'))
    inds = tuple(state_inds+['si','to','j','i']) 
    ls.append(qtn.Tensor(data=A,inds=inds,tags='At'))
    inds = 'si','so'
    ls.append(qtn.Tensor(data=S2,inds=inds,tags='S2'))
    inds = tuple(state_inds+['so','ao','j','o']) 
    ls.append(qtn.Tensor(data=A,inds=inds,tags='A'))
    output_inds = state_inds+['ai','ao','i','o']
    TN = qtn.TensorNetwork(ls)
#    optimize = ctg.HyperOptimizer(max_repeats=64,parallel='ray',reconf_opts={})
    optimize='auto'
    out = TN.contract(optimize=optimize,output_inds=output_inds)
    return - out.data
def _G2(a1,a2,A):
    # Ax,io
    S1 = _S2(a1)
    S2 = _S2(a2)
    n = len(A.shape)-6
    state_inds = ['{},'.format(i+1) for i in range(n)]
    ls = []
    inds = 'a1i','t1o', 'a2i', 't2o'
    ls.append(qtn.Tensor(data=S1,inds=inds,tags='S1'))
    inds = tuple(state_inds+['s1i','t1o','s2i','t2o','j','i']) 
    ls.append(qtn.Tensor(data=A,inds=inds,tags='At'))
    inds = 's1i','s1o', 's2i', 's2o'
    ls.append(qtn.Tensor(data=S2,inds=inds,tags='S2'))
    inds = tuple(state_inds+['s1o','a1o','s2o','a2o','j','o']) 
    ls.append(qtn.Tensor(data=A,inds=inds,tags='A'))
    output_inds = state_inds+['a1i','a1o','a2i','a2o','i','o']
    TN = qtn.TensorNetwork(ls)
#    optimize = ctg.HyperOptimizer(max_repeats=64,parallel='ray',reconf_opts={})
    optimize='auto'
    out = TN.contract(optimize=optimize,output_inds=output_inds)
    return - out.data
def _angles(L,gamma=0.5):
    phi = []
    fac = math.sqrt(1-gamma**2)
    for j in range(L):
        tmp = math.tan(2*math.pi*(j+1)/(2*L+1))*fac
        phi.append(2*math.atan(tmp)-math.pi)
    return phi
#def _amp1(A,angles):
# doesn't work
#    L = len(angles)
#    n = len(A.shape)-4
#    state_inds = ['{},'.format(i+1) for i in range(n)]
#    ls = []
#    U = np.einsum('...ijkl,i,j->...kl',A,ZERO,ZERO)
#    o = 'o' if L==0 else '0,1,'
#    inds = tuple(state_inds+['i',o])
#    ls.append(qtn.Tensor(data=U,inds=inds,tags='A'))
#    for j in range(L):
#        G = _G1(angles[L-1-j],angles[j],A)
#        G = np.einsum('...ijkl,i,j->...kl',G,ZERO,ZERO)
#        i = '{},{},'.format(j,j+1)
#        o = 'o' if j==L-1 else '{},{},'.format(j+1,j+2)
#        inds = tuple(state_inds+[i,o])
#        ls.append(qtn.Tensor(data=G,inds=inds,tags='G{},'.format(j+1)))
#    output_inds = tuple(state_inds+['i','o']) 
#    TN = qtn.TensorNetwork(ls)
##    optimize = ctg.HyperOptimizer(max_repeats=64,parallel='ray',reconf_opts={})
#    optimize='auto'
#    out = TN.contract(optimize=optimize,output_inds=output_inds)
#    return out.data
def _normalize1(A,angles):
    L = len(angles)
    n = len(A.shape)-4
    state_inds = ['{},'.format(i+1) for i in range(n)]
    ls = []
    ao = 'ao' if L==0 else 'a0,a1,'
    o = 'o' if L==0 else '0,1,'
    inds = tuple(state_inds+['ai',ao,'i',o])
    ls.append(qtn.Tensor(data=A,inds=inds,tags='A'))
    for j in range(L):
        G = _G1(angles[L-1-j],angles[j],A)
        ai = 'a{},a{},'.format(j,j+1)
        ao = 'ao' if j==L-1 else 'a{},a{},'.format(j+1,j+2)
        i = '{},{},'.format(j,j+1)
        o = 'o' if j==L-1 else '{},{},'.format(j+1,j+2)
        inds = tuple(state_inds+[ai,ao,i,o])
        ls.append(qtn.Tensor(data=G,inds=inds,tags='G{},'.format(j)))
    output_inds = tuple(state_inds+['ai','ao','i','o']) 
    TN = qtn.TensorNetwork(ls)
#    optimize = ctg.HyperOptimizer(max_repeats=64,parallel='ray',reconf_opts={})
    optimize='auto'
    out = TN.contract(optimize=optimize,output_inds=output_inds)
    return out.data
def _normalize2(A,angles):
    L = len(angles)
    n = len(A.shape)-6
    state_inds = ['{},'.format(i+1) for i in range(n)]
    ls = []
    a1o = 'a1o' if L==0 else 'a10,a11,'
    a2o = 'a2o' if L==0 else 'a20,a21,'
    o = 'o' if L==0 else '0,1,'
    inds = tuple(state_inds+['a1i',a1o,'a2i',a2o,'i',o])
    ls.append(qtn.Tensor(data=A,inds=inds,tags='A'))
    for j in range(L):
        G = _G2(angles[L-1-j],angles[j],A)
        a1i = 'a1{},a1{},'.format(j,j+1)
        a1o = 'a1o' if j==L-1 else 'a1{},a1{},'.format(j+1,j+2)
        a2i = 'a2{},a2{},'.format(j,j+1)
        a2o = 'a2o' if j==L-1 else 'a2{},a2{},'.format(j+1,j+2)
        i = '{},{},'.format(j,j+1)
        o = 'o' if j==L-1 else '{},{},'.format(j+1,j+2)
        inds = tuple(state_inds+[a1i,a1o,a2i,a2o,i,o])
        ls.append(qtn.Tensor(data=G,inds=inds,tags='G{},'.format(j)))
    output_inds = tuple(state_inds+['a1i','a1o','a2i','a2o','i','o']) 
    TN = qtn.TensorNetwork(ls)
#    optimize = ctg.HyperOptimizer(max_repeats=64,parallel='ray',reconf_opts={})
    optimize='auto'
    out = TN.contract(optimize=optimize,output_inds=output_inds)
    return out.data
