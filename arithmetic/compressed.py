import numpy as np
import math
def tensor(f):
    out = np.zeros(f.shape+(2,))
    out[...,1] = f.copy()
    out[...,0] = np.ones_like(f)
    return out
def P(k):
    out = np.zeros((k,2,k+1))
    for i in range(k):
        out[i,0,i] = 1.0
    out[k-1,1,k] = 1.0
    return out
def binom_sum(n):
    fac = [math.factorial(k) for k in range(n+1)]
    out = np.zeros((n+1,)*3)
    for k in range(n+1):
        for i in range(k+1):
            j = k-i
            out[i,j,k] = fac[k]/(fac[i]*fac[j])
    return out
def make_ins(xs,ins):
    ls = []
    for i in range(len(xs)):
        if ins is None: 
            data = np.ones(len(xs[i]))
        else:
            data = np.zeros(len(xs[i]))
            data[ins[i]] = 1.0
        ls.append(data)
    return ls

def xpowers(x,n):
    x = tensor(x)
    out = x.copy()
    for k in range(2,n+1):
        out = np.einsum('xi,xj,ijk->xk',out,x,P(k))
    return out
def uniform(xs,coeff=None,ins=None):
    # poly(q1(x1)+...+qN(xN))
    n = len(coeff)-1
    ADD = binom_sum(n)
    ls = [xpowers(x,n) for x in xs]
    data = make_ins(xs,ins)

    out = np.einsum('xk,x->k',ls[0],data[0])
    for i in range(1,len(xs)):
        tmp = np.einsum('xk,x->k',ls[i],data[i])
        out = np.einsum('i,j,ijk->k',out,tmp,ADD)
    if coeff is None:
        return out
    else:
        return np.dot(out,coeff)
def general(xs,coeff,ins=None):
    # coeff: m*n*N tensor
    m,n,N = coeff.shape
    m -= 1
    data = make_ins(xs,ins)
    def make_c(cki):
        c = np.zeros((m+1,2))
        c[0,0] = 1
        c[:,1] = cki.copy()
        return c

    x = xpowers(xs[0],m)
    out = np.einsum('dm,mi->di',x,make_c(coeff[:,0,0]))
    for k in range(1,n):
        q = np.einsum('dm,mi->di',x,make_c(coeff[:,k,0]))
        out = np.einsum('d...,di->d...i',out,q)
    out = np.einsum('di...,d->i...',out,data[0])

    ADD = np.zeros((2,)*3)
    ADD[1,0,1] = ADD[0,1,1] = ADD[0,0,0] = 1
    for i in range(1,N):
        x = xpowers(xs[i],m)
        q = np.einsum('dm,mi->di',x,make_c(coeff[:,0,i]))
        out = np.einsum('j...,di,ijk->d...k',out,q,ADD)
        for k in range(1,n):
            q = np.einsum('dm,mi->di',x,make_c(coeff[:,k,i]))
            out = np.einsum('dj...,di,ijk->d...k',out,q,ADD)
        out = np.einsum('di...,d->i...',out,data[i])

    CP2 = np.zeros((2,)*3)
    CP2[0,0,0] = CP2[1,1,1] = 1
    for k in range(n-1):
        out = np.einsum('ij...,ijk->k...',out,CP2)
    return out[-1]
            
if __name__=='__main__':
    d = 20
    N = 50
    n = 5
    xs = [np.random.rand(d) for i in range(N)]
    ins = [np.random.randint(0,d) for i in range(N)]

    coeff = np.random.rand(n+1)
    out = contract(xs,coeff,ins=ins)
    y = sum([xs[i][ins[i]] for i in range(N)])
    out_ = 0.0
    for k in range(n+1):
        out_ += coeff[k]*y**k
    print(abs(out-out_)/out_)
