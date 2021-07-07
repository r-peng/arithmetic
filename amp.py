import numpy as np
import math, operator
from functools import reduce
np.set_printoptions(suppress=True)
NOT = np.array([[0,1],[1,0]])
def _o(x,rep): 
    if rep == 'unit':
        return np.eye(2)*x-iY*math.sqrt(1-x**2)
    if rep == 'eye':
        return np.eye(2)*x
    if rep == 'zero':
        return np.array([[x,0],[0,0]])
    else: 
        print('not implemented!')
# utilities
def _svd(A,naxis,thresh=1e-12,bdim=20,print_s=False): # svd after the 1st naxis
    tdim = A.shape
    mdim1 = reduce(operator.mul,tdim[:naxis]) 
    mdim2 = reduce(operator.mul,tdim[naxis:])
    u,s,vh = np.linalg.svd(A.reshape((mdim1,mdim2)))
    if print_s:
        print(s)
    s = s[s>thresh]
    if len(s) > bdim:
        if print_s:
            print('svd truncation err: ', s[bdim])
        s = s[:bdim]
    u = np.reshape(u[:,:len(s)],tdim[:naxis]+(len(s),))
    vh = np.reshape(vh[:len(s),:],(len(s),)+tdim[naxis:])
    return u,s,vh
def _contract(tt,ik): # inplace
    n = len(ik)
    out = np.einsum('xa,x->a',tt[0],ik[0])
    for i in range(1,n):
        tmp = np.einsum('axb,x->ab',tt[i],ik[i])
        out = np.einsum('a,ab->b',out,tmp)
    out = np.einsum('a,apq->pq',out,tt[-1])
    return out
def _get_bdim(tt):
#    return max([tt[i].shape[-1] for i in range(len(tt))])
    return [tt[i].shape[-1] for i in range(len(tt))]
def _right_sweep(l,thresh=1e-12):
    r = []
    u = l[-1].copy()
    if len(u.shape) == 4:
        A = np.einsum('...xb,bpqc->...xpqc',l[-2],u)
        u,s,vh = _svd(A,len(A.shape)-3,thresh)
    else:
        A = np.einsum('...xb,bpq->...xpq',l[-2],u)
        u,s,vh = _svd(A,len(A.shape)-2,thresh)
    u = np.einsum('...a,a->...a',u,s)
    r.insert(0,vh)
    for i in range(len(l)-3,-1,-1):
        A = np.einsum('...xa,ayb->...xyb',l[i],u)
        u,s,vh = _svd(A,len(A.shape)-2,thresh)
        s = np.sqrt(s)
        u = np.einsum('...a,a->...a',u,s)
        vh = np.einsum('a...,a->a...',vh,s)
        r.insert(0,vh)
    return [u]+r

def _input_tt(x,rep='eye',thresh=1e-12):
    d = len(x) 
    A = np.zeros((d,2,2))
    for i in range(d):
        A[i,...] = _o(x[i],rep)
    u,s,vh = _svd(A,1,thresh)
    u = np.einsum('...a,a->...a',u,s)
    return [u,vh]

# addition
def _cx(x):
    cx = np.zeros(x.shape+(2,))
    cx[...,1] = x.copy()
    if len(x.shape)==2:
        d,a = x.shape
        for i in range(d):
            cx[i,0,0] = 1
    else:
        a,d,b = x.shape
        for i in range(d):
#            cx[:,i,:,0] = np.eye(a,b)
            cx[0,i,0,0] = 1
    return cx
def _cv(v):
    cv = np.zeros(v.shape+(2,))
    cv[...,1] = v.copy()
    cv[0,:,:,0] = np.eye(2)
    return cv
def _cy(y,thresh=1e-12):
    cy = [_cx(y[i]) for i in range(len(y)-1)]
    cy.append(_cv(y[-1]))

    l = []
    vh = cy[0].copy()
    for i in range(1,len(cy)-1):
        A = np.einsum('...xac,aybc->...xybc',vh,cy[i])
        u,s,vh = _svd(A,len(A.shape)-3,thresh)
        s = np.sqrt(s)
        u = np.einsum('...a,a->...a',u,s)
        vh = np.einsum('a...,a->a...',vh,s)
        l.append(u)
    A = np.einsum('...xac,apqc->...xpqc',vh,cy[-1])
    u,s,vh = _svd(A,len(A.shape)-3,thresh)
    u = np.einsum('...a,a->...a',u,s)
    l += [u,vh]
    assert len(l)==len(y)
    return _right_sweep(l,thresh)
    return l
def _add_input(y,x,thresh=1e-12):
    cy = _cy(y,thresh) 
    cx = _cy(x,thresh)
    vy = cy.pop()
    A = np.einsum('ji,aprj,xb,brqi->axpq',NOT,vy,cx[0],cx[1])
    u,s,vh = _svd(A,2,thresh)
    u = np.einsum('...a,a->...a',u,s)
    return cy+[u,vh]
def _add_input_all(xs,thresh=1e-12):
    y = xs[0].copy()
    for i in range(1,len(xs)):
        y = _add_input(y,xs[i],thresh)
    return y
def _add_node(y1,y2,thresh=1e-12):
    cy1 = _cy(y1,thresh) 
    cy2 = _cy(y2,thresh)
#    print('cy1',_get_bdim(cy1))
#    print('cy2',_get_bdim(cy2))
    assert len(cy1)==len(cy2)
    l = []
    vh = np.einsum('xa,xb->xab',cy1[0],cy2[0])
    for i in range(1,len(cy1)-1):
        A = np.einsum('...xab,ayc,byd->...xycd',vh,cy1[i],cy2[i])
        u,s,vh = _svd(A,len(A.shape)-3,thresh)
        s = np.sqrt(s)
        u = np.einsum('...a,a->...a',u,s)
        vh = np.einsum('a...,a->a...',vh,s)
        l.append(u)
    A = np.einsum('...xab,aprj,brqi,ji->...xpq',vh,cy1[-1],cy2[-1],NOT)
    u,s,vh = _svd(A,len(A.shape)-2,thresh)
    u = np.einsum('...a,a->...a',u,s)
    l += [u,vh]
    assert len(l)==len(cy1)
    return _right_sweep(l,thresh)
    return l
def _add_const(y,b,rep='eye',thresh=1e-12):
    cb = np.zeros((2,2,2))
    cb[...,0] = np.eye(2)
    cb[...,1] = _o(b,rep)
    cy = _cy(y,thresh) 
    vh = np.einsum('aprj,rqi,ji->apq',cy.pop(),cb,NOT)
    A = np.einsum('axb,bpq->axpq',cy.pop(),vh)
    u,s,vh = _svd(A,2,thresh)
    u = np.einsum('...a,a->...a',u,s)
    return cy+[u,vh]
# multiplication
#def _scalar_mult(y,w,thresh=1e-12):
#    if len(y) == 2:
#        return [y[0]*w,y[1]]
#    else: 
#        wy = y.copy()
#        vh = wy.pop()*w
#        A = np.einsum('axb,bpq->axpq',wy.pop(),vh)
#        u,s,vh = _svd(A,2,thresh)
#        u = np.einsum('...a,a->...a',u,s)
#        return wy+[u,vh]
def _scalar_mult(y,w,thresh=1e-12):
    n = len(y)-1
    w = w**(1/n)
    wy = [y[i]*w for i in range(n)]
    return wy+[y[-1]]
def _node_mult(y1,y2,thresh=1e-12):
    assert len(y1)==len(y2)
    l = []
    vh = np.einsum('xa,xb->xab',y1[0],y2[0])
    for i in range(1,len(y1)-1):
        A = np.einsum('...xab,ayc,byd->...xycd',vh,y1[i],y2[i])
        u,s,vh = _svd(A,len(A.shape)-3,thresh)
        s = np.sqrt(s)
        u = np.einsum('...a,a->...a',u,s)
        vh = np.einsum('a...,a->a...',vh,s)
        l.append(u)
    A = np.einsum('...xab,ap,bq->...xpq',vh,y1[-1][:,:,0],y2[-1][:,0,:])
    u,s,vh = _svd(A,len(A.shape)-2,thresh)
    u = np.einsum('...a,a->...a',u,s)
    l += [u,vh]
    assert len(l)==len(y1)
    return _right_sweep(l,thresh)
    return l
def _powers(y,p,thresh=1e-12):
    powers = [y.copy()]
    for i in range(2,p+1):
        powers.insert(0,_node_mult(powers[0],y,thresh))
    return powers # highest to 1st power
def _poly(y,a,rep='eye',thresh=1e-12):
    n = len(a)-1
    powers = _powers(y,n)
    tt = _scalar_mult(powers[0],a[0],thresh)
    for i in range(1,n):
        tmp = _scalar_mult(powers[i],a[i],thresh)
        tt = _add_node(tt,tmp,thresh)
    tt = _add_const(tt,a[-1],rep,thresh)
    return tt
if __name__=='__main__':
    d = 10
    n = 3
    rep = 'eye'
#    rep = 'zero'
    thresh = 1e-12

    xs = np.random.rand(d,n)
    ls = []
    for i in range(n):
        ls.append(_input_tt(xs[:,i],rep,thresh))
    w = np.random.rand()

    wx1 = _scalar_mult(ls[0],w)
    err = 0.0
    for i in range(d):
        i1 = np.zeros(d)
        i1[i] = 1
        out = _contract(wx1,[i1])
        err += abs(out[0,0]-w*xs[i,0]) 
    print(err, _get_bdim(wx1))
    y = _add_input_all(ls,thresh)  
    err = 0.0
    for i in range(d):
        i1 = np.zeros(d)
        i1[i] = 1
        for j in range(d):
            i2 = np.zeros(d)
            i2[j] = 1
            for k in range(d):
                i3 = np.zeros(d)
                i3[k] = 1
                out = _contract(y,[i1,i2,i3])
                err += abs(out[0,0]-(xs[i,0]+xs[j,1]+xs[k,2]))
    print(err, _get_bdim(y))
    y2 = _add_node(y,y,thresh)  
    err = 0.0
    for i in range(d):
        i1 = np.zeros(d)
        i1[i] = 1
        for j in range(d):
            i2 = np.zeros(d)
            i2[j] = 1
            for k in range(d):
                i3 = np.zeros(d)
                i3[k] = 1
                out = _contract(y2,[i1,i2,i3])
                err += abs(out[0,0]-2*(xs[i,0]+xs[j,1]+xs[k,2]))
    print(err, _get_bdim(y2))
    yw = _add_const(y,w,rep,thresh)  
    err = 0.0
    for i in range(d):
        i1 = np.zeros(d)
        i1[i] = 1
        for j in range(d):
            i2 = np.zeros(d)
            i2[j] = 1
            for k in range(d):
                i3 = np.zeros(d)
                i3[k] = 1
                out = _contract(yw,[i1,i2,i3])
                err += abs(out[0,0]-(xs[i,0]+xs[j,1]+xs[k,2]+w))
    print(err, _get_bdim(yw))
    wy = _scalar_mult(y,w,thresh)  
    err = 0.0
    for i in range(d):
        i1 = np.zeros(d)
        i1[i] = 1
        for j in range(d):
            i2 = np.zeros(d)
            i2[j] = 1
            for k in range(d):
                i3 = np.zeros(d)
                i3[k] = 1
                out = _contract(wy,[i1,i2,i3])
                err += abs(out[0,0]-w*(xs[i,0]+xs[j,1]+xs[k,2]))
    print(err, _get_bdim(wy))
    ysq = _node_mult(y,y,thresh)  
    err = 0.0
    for i in range(d):
        i1 = np.zeros(d)
        i1[i] = 1
        for j in range(d):
            i2 = np.zeros(d)
            i2[j] = 1
            for k in range(d):
                i3 = np.zeros(d)
                i3[k] = 1
                out = _contract(ysq,[i1,i2,i3])
                err += abs(out[0,0]-(xs[i,0]+xs[j,1]+xs[k,2])**2)
    print(err, _get_bdim(ysq))
