import numpy as np
import math, operator
from functools import reduce
# basic tensors
X = np.array([[0,1],[1,0]],dtype=complex)
ZERO = np.array([1,0],dtype=complex)
CiX = np.zeros((2,)*3,dtype=complex)
CiX[0,...] = np.eye(2,dtype=complex)
CiX[1,...] = 1j*X
def _S(a):
    return np.array([[np.exp(1j*a),0],[0,1]],dtype=complex)
def _R(x): # Rio=[exp(-iX*x)]io
    return np.eye(2,dtype=complex)*math.cos(x)-1j*X*math.sin(x)
# utilities
def _angles(L,gamma=0.5):
    phi = []
    fac = math.sqrt(1-gamma**2)
    for j in range(L):
        tmp = math.tan(2*math.pi*(j+1)/(2*L+1))*fac
        phi.append(2*math.atan(tmp)-math.pi)
    return phi
def _svd(A,naxis,thresh=1e-10): # svd after the 1st naxis
    tdim = A.shape
    mdim1 = reduce(operator.mul,tdim[:naxis]) 
    mdim2 = reduce(operator.mul,tdim[naxis:])
    u,s,vh = np.linalg.svd(A.reshape((mdim1,mdim2)))
    s = s[s>thresh]
    u = np.reshape(u[:,:len(s)],tdim[:naxis]+(len(s),))
    vh = np.reshape(vh[:len(s),:],(len(s),)+tdim[naxis:])
    return np.einsum('...a,a->...a',u,s),vh
def _compound(y1,y2,thresh=1e-10):
    assert len(y1) == len(y2)
    n = len(y1) - 1
    xk = []
    xn = np.einsum('xa,xb->xab',y1[0],y2[0])
    for i in range(1,n):
        A = np.einsum('...xab,ayc,byd->...xycd',xn,y1[i],y2[i])
        xi,xn = _svd(A,len(A.shape)-3,thresh)
        xk.append(xi)
    return xk,xn
def _contract(y,ik):
    n = len(ik)
    out = np.einsum('xa,x->a',y[0],ik[0])
    for k in range(1,n):
        tmp = np.einsum('axb,x->ab',y[k])
        out = np.einsum('a,ab->b',out,tmp)
    return np.einsum('a,apq->pq',out,y[-1])
# RUS
def _Rs(x,thresh=1e-10): 
    R = np.zeros((len(x),2,2),dtype=complex)
    for i in range(len(x)):
        R[i,...] = _R(x[i])
    x,v = _svd(R,1,thresh)
    return [x,v]
def _add_input(xk,x,thresh=1e-10):
    v = xk.pop()
    A = np.einsum('aij,xb,bjk->axik',v,x[0],x[1])
    xn,v = _svd(A,2,thresh)
    return xk + [xn,v]
def _add_hidden(y1,y2,thresh=1e-10):
    xk,xn = _compound(y1,y2,thresh)
    A = np.einsum('...ab,aij,bjk->...ik',xn,y1[-1],y2[-1])
    xn,v = _svd(A,len(A.shape)-2,thresh)
    return xk + [xn,v]
def _oaa(gb,angles,thresh=1e-10):
    gbt = [x.conj() for x in gb]
    gbt[-1] = gbt[-1].transpose(0,2,1,4,3)
    L = len(angles)
    xkG,xnG = _compound(gbt,gb,thresh)
    def _G(j):
        svt = np.einsum('ij,ajkpq->aikpq',_S(angles[L-1-j]),gbt[-1])
        sv = np.einsum('ij,ajkpq->aikpq',_S(angles[j]),gb[-1])
        A = np.einsum('...ab,aijpr,bjkrq->...ikpq',xnG,svt,sv)
        xn,v = _svd(A,len(A.shape)-4,thresh)
        return [xn,v]
    AL = gb.copy() 
    AL[-1] = np.einsum('aijpq,i->ajpq',AL[-1],ZERO)
    for j in range(L):
        G = xkG + _G(j)
        xkL,xnL = _compound(AL,G,thresh)
        A = np.einsum('...ab,aipr,bijrq->...jpq',xnL,AL[-1],G[-1])
        xnL,vL = _svd(A,len(A.shape)-3,thresh)
        AL = xkL + [xnL,vL]
    AL[-1] = np.einsum('ajpq,j->apq',AL[-1],ZERO)
    return AL
def _scalar_mult(R,w,angles,thresh=1e-10):
    Rw = _R(w)
    gb = R.copy()
    gb[-1] = np.einsum('...ij,jpq,jk->...ikpq',gb[-1],CiX,Rw)
    return _oaa(gb,angles,thresh)
def _node_mult(y1,y2,angles,thresh=1e-10):
    xk,xn = _compound(y1,y2,thresh)
    A = np.einsum('...ab,aij,bjk,jpq->...ikpq',xn,y1[-1],y2[-1],CiX)
    xn,v = _svd(A,len(A.shape)-4,thresh)
    return _oaa(xk+[xn,v],angles,thresh)
if __name__=='__main__':
    d = 10
    L = 10 
    epsilon = 0.01
    thresh = 1e-12
    angles = _angles(L)
    
