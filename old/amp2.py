import numpy as np
import math, operator
from functools import reduce
# basic tensors
iY = np.array([[0,1],[-1,0]]) # iY
#H = np.array([[1,1],[1,-1]])
NOT = np.array([[0,1],[1,0]])
ZERO = np.array([1,0])
PLUS = np.array([1,1])
CiY = np.zeros((2,)*3)
CiY[0,...] = np.eye(2)
CiY[1,...] = iY
tmp = np.einsum('i,ij,i->ji',PLUS,NOT,PLUS)
def _o(x): 
# for now we restrict O(x) to have only 1 dof
# but can have at least 2 dof
#    return np.eye(2)*math.cos(x)-iY*math.sin(x)
    return np.eye(2)*x-iY*math.sqrt(1-x**2)
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
    out = np.eye(2)
    bdims = []
    while len(ik) > 0:
        tmp = np.einsum('axb,x->ab',y.pop(0),ik.pop(0))
        out = np.einsum('ab,bc->ac',out,tmp)
        bdims.append(out.shape[1])
    return out, y, max(bdims)
def _xs(x): 
    o = np.zeros((2,len(x),2))
    for i in range(len(x)):
        o[:,i,:] = _o(x[i])
    return o
def _scalar_mult(y,w): # inplace
    xn = y.pop()
    xn = np.einsum('ax,j->axj',xn[...,0],_o(w)[0,...])
    return y+[xn]
def _cx(x):
    a1,d,a2 = x.shape
    cx = np.zeros(x.shape+(2,))
    cx[...,1] = x.copy()
    for i in range(d):
        cx[:,i,:,0] = np.eye(a1,a2)
    return cx
def _cy(y,thresh=1e-12):
    n = len(y)
    cy = [_cx(y[i]) for i in range(n)]
    A = cy.pop(0)
    for i in range(1,n):
        A = np.einsum('axbc,bydc->axydc',A,cy.pop(0))
        x,A = _svd(A,2,thresh)
        cy.append(x)
    return cy+[A]
def _add_input(y,x,thresh=1e-12):
    cy = _cy(y,thresh) 
    cx = _cx(x)
    cxn = cy.pop()
    A = np.einsum('ji,pxrj,ryqi->pxyq',tmp,cxn,cx)
    cxn,cx = _svd(A,2,thresh)
    return cy + [cxn,cx]
def _add_node(y1,y2,thresh=1e-12):
    cy1 = _cy(y1,thresh) 
    cy2 = _cy(y2,thresh)
    assert len(cy1)==len(cy2)
    y = []
    A = np.einsum('axpj,bxqi,ji->abxpq',cy1.pop(),cy2.pop(),tmp)
    while len(cy1) > 0:
        A = np.einsum('axc,bxd,cdy...->abxy...',cy1.pop(),cy2.pop(),A)
        A,x = _svd(A,3,thresh)
        y.insert(0,x)
    
#def _xs(x,thresh=1e-10): #xcio
#    o = np.zeros((len(x),2,2,2))
#    for i in range(len(x)):
#        o[i,0,...] = np.eye(2)
#        o[i,1,...] = _x(x[i])
#    x,v = _svd(o,2,thresh)
#    return [x,v]
if __name__=='__main__':
    d = 10
    thresh = 1e-12

    x1 = [np.random.rand() for i in range(d)]
    x2 = [np.random.rand() for i in range(d)]
    x3 = [np.random.rand() for i in range(d)]
    o1 = _xs(x1)
    o2 = _xs(x2)
    o3 = _xs(x3)
    w = np.random.rand()

    wx1 = _scalar_mult([o1],w)
    err = 0.0
    bdims = []
    for i in range(d):
        i1 = np.zeros(d)
        i1[i] = 1
        out, _, max_bdim = _contract(wx1.copy(),[i1])
        err += abs(out[0,0]-w*x1[i]) 
        bdims.append(max_bdim)
    print(err, max(bdims))
    o12 = _add_input([o1],o2,thresh)
    err = 0.0
    bdims = []
    for i in range(d):
        i1 = np.zeros(d)
        i1[i] = 1
        for j in range(d):
            i2 = np.zeros(d)
            i2[j] = 1
            out, _, max_bdim = _contract(o12.copy(),[i1,i2])
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
                out, _, max_bdim = _contract(o123.copy(),[i1,i2,i3])
                err += abs(out[0,0]-(x1[i]+x2[j]+x3[k]))
                bdims.append(max_bdim)
    print(err, max(bdims))
