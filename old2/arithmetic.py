import numpy as np
import math, operator
from functools import reduce
np.set_printoptions(suppress=True)
def _svd(A,naxis,thresh=1e-12,bdim=20): # svd after the 1st naxis
    tdim = A.shape
    mdim1 = reduce(operator.mul,tdim[:naxis]) 
    mdim2 = reduce(operator.mul,tdim[naxis:])
    u,s,vh = np.linalg.svd(A.reshape(mdim1,mdim2),full_matrices=False)
#    print(tdim,s)
#    print('svd truncation err: ', s[bdim])
    s = s[s>thresh] if s[0]>thresh else s[:1]
    s = s[:bdim] if len(s)>bdim else s
    u = np.reshape(u[:,:len(s)],tdim[:naxis]+(len(s),))
    vh = np.reshape(vh[:len(s),:],(len(s),)+tdim[naxis:])
    return u,s,vh
def _contract(tn,ins): 
    tmp = [np.einsum('axb,x->ab',tn[i],ins[i]) for i in range(len(ins))]
    if len(tmp)==1:
        return tmp[0]
    else: 
        return np.linalg.multi_dot(tmp)
def _get_bdim(tn):
    return [tn[i].shape[-1] for i in range(len(tn))]
def _left_sweep(tn,thresh=1e-12,bdim=20):
    out = []
    vh = tn[0].copy()
    for i in range(1,len(tn)):
        A = np.einsum('axb,byd->axyd',vh,tn[i])
        u,s,vh = _svd(A,2,thresh,bdim)
        s = np.sqrt(s)
        u = np.einsum('...a,a->...a',u,s)
        vh = np.einsum('a...,a->a...',vh,s)
        out.append(u)
    out.append(vh)
    assert len(out)==len(tn)
    return out
def _right_sweep(tn,thresh=1e-12,bdim=20):
    out = []
    u = tn[-1].copy()
    for i in range(len(tn)-2,-1,-1):
        A = np.einsum('axb,byd->axyd',tn[i],u)
        u,s,vh = _svd(A,2,thresh,bdim)
        s = np.sqrt(s)
        u = np.einsum('...a,a->...a',u,s)
        vh = np.einsum('a...,a->a...',vh,s)
        out.insert(0,vh)
    out.insert(0,u)
    assert len(out)==len(tn)
    return out
def _compare_list(l1,l2):
    if len(l1)!=len(l2):
        return False
    else:
        out = True
        for i in range(len(l1)):
            if l1[i]!=l2[i]:
                out = False
                break
        return out
def _iterate(tn,thresh=1e-12,bdim=20,max_iter=50):
    old = tn.copy()
    err = 1.0
    for i in range(max_iter):
        tmp = _left_sweep(old,thresh,bdim)
        new = _right_sweep(tmp,thresh,bdim)
        if _compare_list(_get_bdim(old),_get_bdim(new)):
#            err = sum([np.linalg.norm(old[i]-new[i]) for i in range(len(tn))])
#        if err < 1e-6:
            break 
        old = new.copy()
    return old
def _multiply_const(tn,a,i=None,thresh=1e-12,bdim=20,max_iter=50):
    if i is None:
        if a > thresh:
            a = a**(1/len(tn))
            out = [x*a for x in tn]
        elif a < -thresh:
            a = (-a)**(1/len(tn))
            out = [x*a for x in tn]
            out[0] *= -1
        else:
            out = [np.zeros_like(x) for x in tn]
    else: 
        out = tn.copy()
        out[i] = out[i]*a
#    out = _iterate(out,thresh,bdim,max_iter)
    return out
def _multiply_concat(tn1,tn2,thresh=1e-12,bdim=20,max_iter=50):
    # tn1: x1...xk
    # tn2: y1...yl
    # return x1...xk,y1...yl
    out = tn1 + tn2
    out = _iterate(out,thresh,bdim,max_iter)
    return out
def _multiply(tn1,tn2,thresh=1e-12,bdim=20,max_iter=50):
    assert len(tn1)==len(tn2)
    assert tn2[0].shape[0] == 1
    vh  = np.einsum('axb,xc->axbc',tn1[0],tn2[0][0,:,:])
    out = []
    for i in range(1,len(tn1)):
        A = np.einsum('axbc,byd,cye->axyde',vh,tn1[i],tn2[i])
        u,s,vh = _svd(A,2,thresh,bdim)
        s = np.sqrt(s)
        u = np.einsum('...a,a->...a',u,s)
        vh = np.einsum('a...,a->a...',vh,s)
        out.append(u)
    b,y,d,e = vh.shape
    assert d==1
    assert e==1
    vh = vh.reshape((b,y,d))
    out.append(vh)
    assert len(out)==len(tn1)
    out = _iterate(out,thresh,bdim,max_iter)
    return out
def _const(d,a,i=None,thresh=1e-12,bdim=20,max_iter=50):
    if i is None:
        if a > thresh:
            a = a**(1/len(d))
            out = [np.ones((1,di,1))*a for di in d]
        elif a < -thresh:
            a = (-a)**(1/len(d))
            out = [np.ones((1,di,1))*a for di in d]
            out[0] *= -1
        else:
            out = [np.zeros((1,di,1))*a for di in d]
    else:
        out = [np.ones((1,d[i],1)) for i in range(len(d))]
        out[i] = out[i]*a
#    out = _iterate(out,thresh,bdim,max_iter)
    return out
def _add_const(tn,b,i=None,thresh=1e-12,bdim=20,max_iter=50):
    d = [tn[i].shape[1] for i in range(len(tn))]
    b = _const(d,b,i,thresh,bdim,max_iter)
    return _add(tn,b,thresh,bdim,max_iter)
def _add_concat(tn1,tn2,thresh=1e-12,bdim=20,max_iter=50):
    # tn1: x1...xk
    # tn2: y1...yl
    # return x1...xk,y1...yl
    dx = [tn1[i].shape[1] for i in range(len(tn1))]
    bx = _const(dx,1.0,0,thresh,bdim,max_iter)
    dy = [tn2[i].shape[1] for i in range(len(tn2))]
    by = _const(dy,1.0,0,thresh,bdim,max_iter)
    f = tn1+by
    g = bx+tn2
    return _add(f,g,thresh,bdim,max_iter)
def _add(tn1,tn2,thresh=1e-12,bdim=20,max_iter=50):
    assert len(tn1)==len(tn2)
    for i in range(len(tn1)):
        assert tn1[i].shape[1]==tn2[i].shape[1]
    if len(tn1)==1:
        return [tn1[0]+tn2[0]]
    else: 
        out = []
        a1,x,b1 = tn1[0].shape
        a2,x,b2 = tn2[0].shape
        assert a1==1
        assert a2==1
        vh = np.zeros((1,x,b1+b2))
        vh[0,:,:b1] = tn1[0][0,:,:].copy()
        vh[0,:,b1:] = tn2[0][0,:,:].copy()
        for i in range(1,len(tn1)-1):
            a1,x,b1 = tn1[i].shape
            a2,x,b2 = tn2[i].shape
            tmp = np.zeros((a1+a2,x,b1+b2))
            tmp[:a1,:,:b1] = tn1[i]
            tmp[a1:,:,b1:] = tn2[i]
            A = np.einsum('axb,byc->axyc',vh,tmp)
            u,s,vh = _svd(A,2,thresh,bdim) 
            s = np.sqrt(s)
            u = np.einsum('...a,a->...a',u,s)
            vh = np.einsum('a...,a->a...',vh,s)
            out.append(u)
        a1,x,b1 = tn1[-1].shape
        a2,x,b2 = tn2[-1].shape
        assert b1==1
        assert b2==1
        tmp = np.zeros((a1+a2,x,1))
        tmp[:a1,:,0] = tn1[-1][:,:,0].copy()
        tmp[a1:,:,0] = tn2[-1][:,:,0].copy()
        A = np.einsum('axb,byc->axyc',vh,tmp)
        u,s,vh = _svd(A,2,thresh,bdim) 
        s = np.sqrt(s)
        u = np.einsum('...a,a->...a',u,s)
        vh = np.einsum('a...,a->a...',vh,s)
        out = out+[u,vh]
        assert len(out)==len(tn1)
        out = _iterate(out,thresh,bdim,max_iter)
        return out
def _poly1(tn,a,thresh=1e-12,bdim=20,max_iter=50):
    # a = [a0,...,ap]
    powers = [None,tn.copy()]
    for i in range(2,len(a)):
        powers.append(_multiply(tn,powers[-1],thresh,bdim,max_iter))
#        print('power={},bdim={}'.format(i,_get_bdim(powers[-1])))
    assert len(powers)==len(a)
    term = _multiply_const(powers[1],a[1],None,thresh,bdim,max_iter)
    f = _add_const(term,a[0],None,thresh,bdim,max_iter)
    for i in range(2,len(a)):
        term = _multiply_const(powers[i],a[i],None,thresh,bdim,max_iter)
        f = _add(f,term,thresh,bdim,max_iter)
#        print('order={},bdim={}'.format(i,_get_bdim(f)))
    return f
def _poly2(tn,coeff,thresh=1e-12,bdim=20,max_iter=50):
    # Horner's method
    a = coeff.copy()
    a.reverse()
    f = _multiply_const(tn,a[0],None,thresh,bdim,max_iter)
    f = _add_const(f,a[1],None,thresh,bdim,max_iter)
    for i in range(2,len(a)):
        f = _multiply(f,tn,thresh,bdim,max_iter)
        f = _add_const(f,a[i],None,thresh,bdim,max_iter)
#        print('power={},bdim={}'.format(i,_get_bdim(f)))
    return f
if __name__=='__main__':
    import math
    p = 10
    a = [1.0/math.factorial(i) for i in range(p+1)]
    
    n = 10 
    d = 10
    thresh = 1e-12
    bdim = 20
    max_iter = 50
    x = np.random.rand(n,d)/n
    i = np.zeros(d)
    i[0] = 1
    ins = [i for j in range(n)]
    tn = [x[0,:].reshape(1,d,1)]
    for i in range(1,n):
        tn = _add_concat(tn,[x[i,:].reshape(1,d,1)],thresh,bdim,max_iter)
    out = _contract(tn,ins)
    print('err',abs(out[0,0]-sum(x[:,0])),_get_bdim(tn))
    #exit()
    f1 = _poly1(tn,a,thresh,bdim,max_iter)
    f2 = _poly2(tn,a,thresh,bdim,max_iter)
    out1 = _contract(f1,ins)
    out2 = _contract(f2,ins)
    print('err1',abs(out1[0,0]-np.exp(sum(x[:,0]))))
    print('err2',abs(out2[0,0]-np.exp(sum(x[:,0]))))
