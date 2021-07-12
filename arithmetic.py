import numpy as np
import math, operator
from functools import reduce
np.set_printoptions(suppress=True)
def _svd(A,naxis,thresh=1e-12,bdim=20): # svd after the 1st naxis
    tdim = A.shape
    mdim1 = reduce(operator.mul,tdim[:naxis]) 
    mdim2 = reduce(operator.mul,tdim[naxis:])
    u,s,vh = np.linalg.svd(A.reshape((mdim1,mdim2)))
#    print(s)
#    print('svd truncation err: ', s[bdim])
    s = s[s>thresh] if s[0]>thresh else s[:1]
    s = s[:bdim] if len(s)>bdim else s
    u = np.reshape(u[:,:len(s)],tdim[:naxis]+(len(s),))
    vh = np.reshape(vh[:len(s),:],(len(s),)+tdim[naxis:])
    return u,s,vh
def _swap2(i,tn,thresh=1e-12,bdim=20):
    # swap the variable at position (i,i+1)
    if len(tn[i].shape)==4:
        A = np.einsum('axcb,byd->ayxcd',tn[i],tn[i+1])
        u,s,vh = _svd(A,2,thresh,bdim)
    elif len(tn[i+1].shape)==4:
        A = np.einsum('axb,bycd->aycxd',tn[i],tn[i+1])
        u,s,vh = _svd(A,3,thresh,bdim)
    else:
        A = np.einsum('axb,byd->ayxd',tn[i],tn[i+1])
        u,s,vh = _svd(A,2,thresh,bdim)
    s = np.sqrt(s)
    u = np.einsum('...a,a->...a',u,s)
    vh = np.einsum('a...,a->a...',vh,s)
    if i+1==len(tn)-1:
        return tn[:i]+[u,vh]
    else:
        return tn[:i]+[u,vh]+tn[i+2:]
def _swapk(k,tn,thresh=1e-12,bdim=20,max_iter=50):
    # move the first k variables to the last
    out = tn.copy()
    l = len(tn)-k
    if k <= l:
        for i in range(k):
            for j in range(len(tn)-1):
                out = _swap2(j,out,thresh,bdim)
    else:
        for i in range(l):
            for j in range(len(tn)-1,0,-1):
                out = _swap2(j-1,out,thresh,bdim)
    out = _iterate(out,thresh,bdim,max_iter)
    return out
def _reverse(tn):
    return [tn[len(tn)-1-i].transpose(2,1,0) for i in range(len(tn))]
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
            err = sum([np.linalg.norm(old[i]-new[i]) for i in range(len(tn))])
        if err < 1e-6:
            break 
        old = new.copy()
    return old
def _multiply0(tn,a,thresh=1e-12,bdim=20,max_iter=50):
    a = a**(1.0/len(tn))
    out = [x*a for x in tn]
    out = _iterate(out,thresh,bdim,max_iter)
    return out
def _multiply1(tn1,tn2,thresh=1e-12,bdim=20,max_iter=50):
    # tn1: x1...xk
    # tn2: y1...yl
    # return x1...xk,y1...yl
    out = tn1 + tn2
    out = _iterate(out,thresh,bdim,max_iter)
    return out
def _multiply2(tn1,tn2,k,thresh=1e-12,bdim=20,max_iter=50):
    # tn1: x1...xk,y1...yl
    # tn2: x1...xk,z1...zm
    # return y1...yl,x1...xk,z1...zm
    f = tn1.copy()
    g = tn2.copy()
    f = _swapk(k,f,thresh,bdim)
    vh  = np.einsum('axb,xe->axbe',f[-k],g[0][0,:,:])
    xs = []
    for i in range(1,k):
        A = np.einsum('axbd,bye,dyh->axyeh',vh,f[-k+i],g[i])
        u,s,vh = _svd(A,2,thresh,bdim)
        s = np.sqrt(s)
        u = np.einsum('...a,a->...a',u,s)
        vh = np.einsum('a...,a->a...',vh,s)
        xs.append(u)
    b,y,e,h = vh.shape
    assert e==1 
    vh = vh.reshape((b,y,h))
    xs.append(vh)
    out = f[:-k]+xs+g[k:]
    assert len(out)==len(f)-k+len(g)
    out = _iterate(out,thresh,bdim,max_iter)
    return out
def _const(d,b,thresh=1e-12,bdim=20,max_iter=50):
    b = b**(1.0/len(d))
    x = np.ones((1,d[0],1))*b
    out = [x]
    for i in range(1,len(d)):
        out = _multiply1(out,[x],thresh,bdim,max_iter)
    return out
def _add0(tn,b,thresh=1e-12,bdim=20,max_iter=50):
    d = [tn[i].shape[1] for i in range(len(tn))]
    b = _const(d,b,thresh,bdim,max_iter)
    return _add2(tn,b,len(tn),thresh,bdim,max_iter)
def _add1(tn1,tn2,thresh=1e-12,bdim=20,max_iter=50):
    # tn1: x1...xk
    # tn2: y1...yl
    # return x1...xk,y1...yl
    a1,x,b1 = tn1[-1].shape
    a2,y,b2 = tn2[0].shape
    assert b1==1
    assert a2==1
    tmp1 = np.zeros((a1,x,b1+a2))
    tmp1[:,:,0] = tn1[-1][:,:,0]
#    tmp1[:,:,1] = np.ones((a1,x)) 
    tmp1[:,:,1] = np.ones((a1,x)) 
    tmp2 = np.zeros((b1+a2,y,b2))
    tmp2[1,:,:] = tn2[0][0,:,:]
    tmp2[0,:,:] = np.ones((y,b2))
    A = np.einsum('axi,iyb->axyb',tmp1,tmp2)
    u,s,vh = _svd(A,2,thresh,bdim)
    s = np.sqrt(s)
    u = np.einsum('...a,a->...a',u,s)
    vh = np.einsum('a...,a->a...',vh,s)
    out = tn1[:-1]+[u,vh]+tn2[1:]
    assert len(out)==len(tn1)+len(tn2)
    out = _iterate(out,thresh,bdim,max_iter)
    return out
def _add2(tn1,tn2,k,thresh=1e-12,bdim=20,max_iter=50):
    # tn1: x1...xk,y1...yl
    # tn2: x1...xk,z1...zm
    # return y1...yl,x1...xk,z1...zm
    f = tn1.copy()
    g = tn2.copy()
    n1 = len(tn1)
    n2 = len(tn2)

    f = _swapk(k,f,thresh,bdim,max_iter)
    xs = []
    
    af,x,bf = f[-k].shape
    ag,x,bg = g[0].shape
    assert ag==1
    if n1==k:
        assert af==1
        vh = np.ones((1,1,af+ag))
    else: 
        b,y,_ = f[-k-1].shape
        vh = np.zeros((b,y,af+ag))
        vh[:,:,:af] = f[-k-1].copy()
        vh[:,:,-1] = np.ones((b,y))
    for i in range(k):
        fi = f[n1-k+i]
        gi = g[i]
        af,x,bf = fi.shape
        ag,x,bg = gi.shape
        tmp = np.zeros((af+ag,x,bf+bg))
        tmp[:af,:,:bf] = fi
        tmp[af:,:,bf:] = gi
        A = np.einsum('axb,byc->axyc',vh,tmp)
        u,s,vh = _svd(A,2,thresh,bdim) 
        s = np.sqrt(s)
        u = np.einsum('...a,a->...a',u,s)
        vh = np.einsum('a...,a->a...',vh,s)
        xs.append(u)
    af,x,bf = f[-1].shape
    ag,x,bg = g[k-1].shape
    assert bf==1
    if n2==k:
        assert bg==1
        tmp = np.ones((bf+bg,1,1))
    else:
        _,y,c = g[k].shape
        tmp = np.zeros((bf+bg,y,c))
        tmp[0,:,:] = np.ones((y,c))
        tmp[1:,:,:] = g[k].copy()
    A = np.einsum('axb,byc->axyc',vh,tmp)
    u,s,vh = _svd(A,2,thresh,bdim) 
    s = np.sqrt(s)
    u = np.einsum('...a,a->...a',u,s)
    vh = np.einsum('a...,a->a...',vh,s)
    xs.append(u)
    yl = xs.pop(0)
    z1 = vh.copy()
    if n1==k:
        assert yl.shape[0]==1
        assert yl.shape[1]==1
        yl = yl.reshape((1,yl.shape[2]))
        xs[0] = np.einsum('ab,bxc->axc',yl,xs[0])
    else:
        f[-k-1] = yl
    if n2==k: 
        assert z1.shape[1]==1
        assert z1.shape[2]==1
        z1 = z1.reshape((z1.shape[2],1))
        xs[-1] = np.einsum('axb,bc->axc',xs[-1],z1)
    else:
        g[k] = z1
    out = f[:-k]+xs+g[k:]
    assert len(out)==len(f)-k+len(g)
    out = _iterate(out,thresh,bdim,max_iter)
    return out
def _poly1(tn,a,thresh=1e-12,bdim=20,max_iter=50):
    # a = [a0,...,ap]
    powers = [None,tn.copy()]
    for i in range(2,len(a)):
        powers.append(_multiply2(tn,powers[-1],len(tn),thresh,bdim,max_iter))
        print('power={},bdim={}'.format(i,_get_bdim(powers[-1])))
    assert len(powers)==len(a)
    term = _multiply0(powers[1],a[1],thresh,bdim,max_iter)
    f = _add0(term,a[0],thresh,bdim,max_iter)
    for i in range(2,len(a)):
        term = _multiply0(powers[i],a[i],thresh,bdim,max_iter)
        f = _add2(f,term,len(term),thresh,bdim,max_iter)
        print('order={},bdim={}'.format(i,_get_bdim(f)))
    return f
def _poly2(tn,coeff,thresh=1e-12,bdim=20,max_iter=50):
    # Horner's method
    a = coeff.copy()
    a.reverse()
    f = _multily0(tn,a[0],thresh,bdim,max_iter)
    f = _add0(f,a[1],thresh,bdim,max_iter)
    for i in range(2,len(a)):
        f = _multiply2(f,tn,len(tn),thresh,bdim,max_iter)
        f = _add0(f,a[i],thresh,bdim,max_iter)
        print('power={},bdim={}'.format(i,_get_bdim(f)))
    return f
