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
def _swapk(k,tn,thresh=1e-12,bdim=20):
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
def _cf(tn,thresh=1e-12,bdim=20):
    # control on x0
    cf1 = []
    for i in range(len(tn)):
        a,x,b = tn[i].shape
        cx = np.zeros((a,x,2,b))
        cx[:,:,1,:] = tn[i].copy()
        for j in range(cx.shape[1]):
            cx[0,j,0,0] = 1
        cf1.append(cx)
    assert len(cf1)==len(tn)
    cf2 = []
    u = cf1[-1].copy()
    for i in range(len(cf1)-2,-1,-1):
        A = np.einsum('axcb,bycd->axcyd',cf1[i],u)
        u,s,vh = _svd(A,3,thresh,bdim)
        s = np.sqrt(s)
        u = np.einsum('...a,a->...a',u,s)
        vh = np.einsum('a...,a->a...',vh,s)
        cf2.insert(0,vh)
    cf2.insert(0,u)
    assert len(cf2)==len(tn)
    return cf2
def _compose0(tn,a,op,thresh=1e-12,bdim=20,max_iter=50):
    if op=='+':
        f = _cf(tn,thresh,bdim)
        x = f[0][:,:,1,:]+f[0][:,:,0,:]*a
        out = [x]+f[1:]
    elif op=='*':
        a = a**(1.0/len(tn))
        out = [x*a for x in tn]
    else:
        print('Not implemented!')
    out = _iterate(out,thresh,bdim,max_iter)
    return out
def _compose1(tn1,tn2,op,thresh=1e-12,bdim=20,max_iter=50):
    # tn1: x1...xk
    # tn2: y1...yl
    # return x1...xk,y1...yl
    f = _reverse(tn1)
    if op=='+':
        f = _cf(f,thresh,bdim)
        g = _cf(tn2,thresh,bdim)
        A  = np.einsum('xb,ye->bxye',f[0][0,:,0,:],g[0][0,:,1,:])
        A += np.einsum('xb,ye->bxye',f[0][0,:,1,:],g[0][0,:,0,:])
    elif op=='*':
        g = tn2.copy()
        A = np.einsum('xb,ye->bxye',f[0][0,:,:],g[0][0,:,:])
    else: 
        print('Not implemented!')
    u,s,vh = _svd(A,2,thresh,bdim)
    s = np.sqrt(s)
    u = np.einsum('...a,a->...a',u,s)
    vh = np.einsum('a...,a->a...',vh,s)
    out = _reverse(f[1:])+[u,vh]+g[1:]
    assert len(out)==len(f)+len(g)
    out = _iterate(out,thresh,bdim,max_iter)
    return out
def _compose2(tn1,tn2,k,op,thresh=1e-12,bdim=20,max_iter=50):
    # tn1: x1...xk,y1...yl
    # tn2: x1...xk,z1...zm
    # return y1...yl,x1...xk,z1...zm
    if op=='+':
        f = _cf(tn1,thresh,bdim)
        g = _cf(tn2,thresh,bdim)
        f = _swapk(k,f,thresh,bdim)
        vh  = np.einsum('axb,xe->axbe',f[-k][:,:,0,:],g[0][0,:,1,:])
        vh += np.einsum('axb,xe->axbe',f[-k][:,:,1,:],g[0][0,:,0,:])
    elif op=='*': 
        f = tn1.copy()
        g = tn2.copy()
        f = _swapk(k,f,thresh,bdim)
        vh  = np.einsum('axb,xe->axbe',f[-k],g[0][0,:,:])
    else:
        print('Not implemented!')
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
def _poly1(tn,a,thresh=1e-12,bdim=20,max_iter=50):
    # a = [a0,...,ap]
    powers = [None,tn.copy()]
    for i in range(2,len(a)):
        powers.append(_compose2(tn,powers[-1],len(tn),'*',thresh,bdim,max_iter))
        print('power={},bdim={}'.format(i,_get_bdim(powers[-1])))
    assert len(powers)==len(a)
    term = _compose0(powers[1],a[1],'*',thresh,bdim,max_iter)
    f = _compose0(term,a[0],'+',thresh,bdim,max_iter)
    for i in range(2,len(a)):
        term = _compose0(powers[i],a[i],'*',thresh,bdim,max_iter)
        f = _compose2(f,term,len(term),'+',thresh,bdim,max_iter)
        print('order={},bdim={}'.format(i,_get_bdim(f)))
    return f
def _poly2(tn,coeff,thresh=1e-12,bdim=20,max_iter=50):
    # Horner's method
    a = coeff.copy()
    a.reverse()
    f = _compose0(tn,a[0],'*',thresh,bdim,max_iter)
    f = _compose0(f,a[1],'+',thresh,bdim,max_iter)
    for i in range(2,len(a)):
        f = _compose2(f,tn,len(tn),'*',thresh,bdim,max_iter)
        f = _compose0(f,a[i],'+',thresh,bdim,max_iter)
        print('power={},bdim={}'.format(i,_get_bdim(f)))
    return f
