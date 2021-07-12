import numpy as np
import math, operator
from functools import reduce
# basic tensors
iY = np.array([[0,1],[-1,0]]) # iY
#H = np.array([[1,1],[1,-1]])
NOT = np.array([[0,1],[1,0]])
def _o(x): 
# for now we restrict O(x) to have only 1 dof
# but can have at least 2 dof
#    return np.eye(2)*math.cos(x)-iY*math.sin(x)
#    return np.eye(2)*x-iY*math.sqrt(1-x**2)
    return np.eye(2)*x
#    return np.array([[x,0],[0,0]])
def _svd(A,naxis,thresh=1e-12): # svd after the 1st naxis
    tdim = A.shape
    mdim1 = reduce(operator.mul,tdim[:naxis]) 
    mdim2 = reduce(operator.mul,tdim[naxis:])
    u,s,vh = np.linalg.svd(A.reshape((mdim1,mdim2)))
    s = s[s>thresh]
    u = np.reshape(u[:,:len(s)],tdim[:naxis]+(len(s),))
    vh = np.reshape(vh[:len(s),:],(len(s),)+tdim[naxis:])
    return np.einsum('...a,a->...a',u,s),vh
def _contract(y,ik): # inplace
    out = np.einsum('xa,x->a',y.pop(0),ik.pop(0))
    bdims = [out.shape[0]]
    while len(ik) > 0:
        tmp = np.einsum('axb,x->ab',y.pop(0),ik.pop(0))
        out = np.einsum('a,ab->b',out,tmp)
        bdims.append(out.shape[0])
    out = np.einsum('a,apq->pq',out,y.pop())
    return out, max(bdims)
def _xs(x,thresh=1e-12): 
    o = np.zeros((len(x),2,2))
    for i in range(len(x)):
        o[i,...] = _o(x[i])
    x,v = _svd(o,1,thresh)
    return [x,v]
def _scalar_mult(y,w): # inplace
    wy = y.copy()
    v = wy.pop()
    v = np.einsum('ai,j->aij',v[...,0],_o(w)[0,...])
    return wy+[v]
def _node_mult(y1,y2,thresh):
    assert len(y1)==len(y2)
    n = len(y1)-1
    y = []
#    A = np.einsum('ap,bq->abpq',y1[-1][:,:,0],y2[-1][:,0,:])
    A = np.einsum('a,b,pq->abpq',y1[-1][:,0,0],y2[-1][:,0,0],np.eye(2))
    for i in range(n-1,0,-1):
        A = np.einsum('axc,bxd,cd...->abx...',y1[i],y2[i],A)
        A,x = _svd(A,3,thresh)
        y.insert(0,x)
    A = np.einsum('xc,xd,cd...->x...',y1[0],y2[0],A)
    A,x = _svd(A,1,thresh)
    return [A,x] + y
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
            cx[:,i,:,0] = np.eye(a,b)
    return cx
def _cv(v):
    cv = np.zeros(v.shape+(2,))
    cv[...,1] = v.copy()
    cv[0,:,:,0] = np.eye(2)
    return cv
def _cy(y,thresh=1e-12):
    n = len(y)-1
    cy = [_cx(y[i]) for i in range(n)]
    cy.append(_cv(y[-1]))
    A = cy.pop(0)
    for i in range(1,n):
        A = np.einsum('...xac,aybc->...xybc',A,cy.pop(0))
        x,A = _svd(A,len(A.shape)-3,thresh)
        cy.append(x)
    A = np.einsum('...xac,apqc->...xpqc',A,cy.pop(0))
    x,v = _svd(A,len(A.shape)-3,thresh)
    return cy+[x,v]
def _add_input(y,x,thresh=1e-12):
    cy = _cy(y,thresh) 
    cx = _cy(x,thresh)
    vy = cy.pop()
    A = np.einsum('ji,aprj,xb,brqi->axpq',NOT,vy,cx[0],cx[1])
    cxn,v = _svd(A,2,thresh)
    return cy + [cxn,v]
def _add_node(y1,y2,thresh=1e-12):
    cy1 = _cy(y1,thresh) 
    cy2 = _cy(y2,thresh)
    assert len(cy1)==len(cy2)
    y = []
    A = np.einsum('aprj,brqi,ji->abpq',cy1.pop(),cy2.pop(),NOT)
    while len(cy1) > 1:
        A = np.einsum('axc,bxd,cd...->abx...',cy1.pop(),cy2.pop(),A)
        A,x = _svd(A,3,thresh)
        y.insert(0,x)
    A = np.einsum('xc,xd,cd...->x...',cy1.pop(),cy2.pop(),A)
    A,x = _svd(A,1,thresh)
    return [A,x] + y
# saved 
#def _add_node2(y1,y2,thresh=1e-12):
#    cy1 = _cy(y1,thresh) 
#    cy2 = _cy(y2,thresh)
#    assert len(cy1)==len(cy2)
#    y = []
#    u = np.einsum('aprj,brqi,ji->abpq',cy1.pop(),cy2.pop(),NOT)
#    A = np.einsum('axc,bxd,cd...->abx...',cy1.pop(),cy2.pop(),u)
#    u,s,vh = _svd(A,3,thresh)
#    u = np.einsum('...a,a->...a',u,s)
#    y.insert(0,vh)
#    while len(cy1) > 1:
#        A = np.einsum('axc,bxd,cd...->abx...',cy1.pop(),cy2.pop(),u)
#        u,s,vh = _svd(A,3,thresh)
#        s = np.sqrt(s)
#        u = np.einsum('...a,a->...a',u,s)
#        vh = np.einsum('a...,a->a...',vh,s)
#        y.insert(0,vh)
#    A = np.einsum('xc,xd,cd...->x...',cy1.pop(),cy2.pop(),u)
#    u,s,vh = _svd(A,1,thresh)
#    s = np.sqrt(s)
#    u = np.einsum('...a,a->...a',u,s)
#    vh = np.einsum('a...,a->a...',vh,s)
#    return [u,vh] + y
#def _node_mult2(y1,y2,thresh=1e-12):
#    assert len(y1)==len(y2)
#    y = []
#    u = np.einsum('ap,bq->abpq',y1[-1][:,:,0],y2[-1][:,0,:])
##    vh = np.einsum('a,b,pq->abpq',y1[-1][:,0,0],y2[-1][:,0,0],np.eye(2))
#    A = np.einsum('axc,bxd,cd...->abx...',y1[-2],y2[-2],u)
#    u,s,vh = _svd(A,3,thresh)
#    u = np.einsum('...a,a->...a',u,s)
#    y.insert(0,vh)
#    for i in range(len(y1)-3,0,-1):
#        A = np.einsum('axc,bxd,cd...->abx...',y1[i],y2[i],u)
#        u,s,vh = _svd(A,3,thresh)
#        s = np.sqrt(s)
#        u = np.einsum('...a,a->...a',u,s)
#        vh = np.einsum('a...,a->a...',vh,s)
#        y.insert(0,vh)
#    A = np.einsum('xc,xd,cd...->x...',y1[0],y2[0],u)
#    u,s,vh = _svd(A,1,thresh)
#    s = np.sqrt(s)
#    u = np.einsum('...a,a->...a',u,s)
#    vh = np.einsum('a...,a->a...',vh,s)
#    return [u,vh] + y
if __name__=='__main__':
    d = 10
    thresh = 1e-12

    x1 = [np.random.rand() for i in range(d)]
    x2 = [np.random.rand() for i in range(d)]
    x3 = [np.random.rand() for i in range(d)]
    o1 = _xs(x1,thresh)
    o2 = _xs(x2,thresh)
    o3 = _xs(x3,thresh)
    w = np.random.rand()

    wx1 = _scalar_mult(o1,w)
    err = 0.0
    bdims = []
    for i in range(d):
        i1 = np.zeros(d)
        i1[i] = 1
        out, max_bdim = _contract(wx1.copy(),[i1])
        err += abs(out[0,0]-w*x1[i]) 
        bdims.append(max_bdim)
    print(err, max(bdims))
    o12 = _add_input(o1,o2,thresh)
    err = 0.0
    bdims = []
    for i in range(d):
        i1 = np.zeros(d)
        i1[i] = 1
        for j in range(d):
            i2 = np.zeros(d)
            i2[j] = 1
            out, max_bdim = _contract(o12.copy(),[i1,i2])
            err += abs(out[0,0]-(x1[i]+x2[j]))
            bdims.append(max_bdim)
    print(err, max(bdims))
    o123 = _add_input(o12,o3,thresh)
    err = 0.0
    bdims = []
    for i in range(d):
        i1 = np.zeros(d)
        i1[i] = 1
        for j in range(d):
            i2 = np.zeros(d)
            i2[j] = 1
            for k in range(d):
                i3 = np.zeros(d)
                i3[k] = 1
                out, max_bdim = _contract(o123.copy(),[i1,i2,i3])
                err += abs(out[0,0]-(x1[i]+x2[j]+x3[k]))
                bdims.append(max_bdim)
    print(err, max(bdims))
    o123_2 = _add_node(o123,o123,thresh)
    err = 0.0
    bdims = []
    for i in range(d):
        i1 = np.zeros(d)
        i1[i] = 1
        for j in range(d):
            i2 = np.zeros(d)
            i2[j] = 1
            for k in range(d):
                i3 = np.zeros(d)
                i3[k] = 1
                out, max_bdim = _contract(o123_2.copy(),[i1,i2,i3])
                err += abs(out[0,0]-2*(x1[i]+x2[j]+x3[k]))
                bdims.append(max_bdim)
    print(err, max(bdims))
    o123sq = _node_mult(o123,o123,thresh)
    err = 0.0
    bdims = []
    for i in range(d):
        i1 = np.zeros(d)
        i1[i] = 1
        for j in range(d):
            i2 = np.zeros(d)
            i2[j] = 1
            for k in range(d):
                i3 = np.zeros(d)
                i3[k] = 1
                out, max_bdim = _contract(o123sq.copy(),[i1,i2,i3])
                err += abs(out[0,0]-(x1[i]+x2[j]+x3[k])**2)
                bdims.append(max_bdim)
    print(err, max(bdims))
    o123qb = _node_mult(o123,o123sq,thresh)
    err = 0.0
    bdims = []
    for i in range(d):
        i1 = np.zeros(d)
        i1[i] = 1
        for j in range(d):
            i2 = np.zeros(d)
            i2[j] = 1
            for k in range(d):
                i3 = np.zeros(d)
                i3[k] = 1
                out, max_bdim = _contract(o123qb.copy(),[i1,i2,i3])
                err += abs(out[0,0]-(x1[i]+x2[j]+x3[k])**3)
                bdims.append(max_bdim)
    print(err, max(bdims))
    o123qt = _node_mult(o123sq,o123sq,thresh)
    err = 0.0
    bdims = []
    for i in range(d):
        i1 = np.zeros(d)
        i1[i] = 1
        for j in range(d):
            i2 = np.zeros(d)
            i2[j] = 1
            for k in range(d):
                i3 = np.zeros(d)
                i3[k] = 1
                out, max_bdim = _contract(o123qt.copy(),[i1,i2,i3])
                err += abs(out[0,0]-(x1[i]+x2[j]+x3[k])**4)
                bdims.append(max_bdim)
    print(err, max(bdims))
    exit()
