import numpy as np
import math, operator
from functools import reduce
X = np.array([[0,1],[1,0]],dtype=complex)
ZERO = np.array([1,0],dtype=complex)
CiX = np.zeros((2,)*3,dtype=complex)
CiX[0,...] = np.eye(2,dtype=complex)
CiX[1,...] = 1j*X
def _angles(L,gamma=0.5):
    phi = []
    fac = math.sqrt(1-gamma**2)
    for j in range(L):
        tmp = math.tan(2*math.pi*(j+1)/(2*L+1))*fac
        phi.append(2*math.atan(tmp)-math.pi)
    return phi
def _svd(T,naxis,thresh=1e-10): # svd after the 1st naxis
    tdim = T.shape
    mdim1 = reduce(operator.mul,tdim[:naxis]) 
    mdim2 = reduce(operator.mul,tdim[naxis:])
    u,s,vh = np.linalg.svd(T.reshape((mdim1,mdim2)))
    s = s[s>thresh]
    u = np.reshape(u[:,:len(s)],tdim[:naxis]+(len(s),))
    vh = np.reshape(vh[:len(s),:],(len(s),)+tdim[naxis:])
    return np.einsum('...a,a->...a',u,s),vh
def _S(a):
    return np.array([[np.exp(1j*a),0],[0,1]],dtype=complex)
def _R(x): # Rio=[exp(-iX*x)]io
    return np.eye(2,dtype=complex)*math.cos(x)-1j*X*math.sin(x)
def _Rs(x,thresh=1e-10): 
    R = np.zeros((len(x),2,2),dtype=complex)
    for i in range(len(x)):
        R[i,...] = _R(x[i])
    x,v = _svd(R,1,thresh)
    return [x,v]
def _add_input(Rk,R,thresh=1e-10):
    v = Rk.pop()
    A = np.einsum('aij,xb,bjk->axik',v,R[0],R[1])
    x,v = _svd(A,2,thresh)
    return Rk + [x,v]
def _compound(y1,y2,thresh=1e-10):
    assert len(y1) == len(y2)
    n = len(y1) - 1
    ls = []
    xi = np.einsum('xa,xb->xab',y1[0],y2[0])
    for i in range(1,n):
        A = np.einsum('...xab,ayc,byd->...xycd',xi,y1[i],y2[i])
        xim,xi = _svd(A,len(A.shape)-3,thresh)
        ls.append(xim)
    return ls,xi
def _add_hidden(y1,y2,thresh=1e-10):
    ls,xn = _compound(y1,y2,thresh)
    A = np.einsum('...ab,aij,bjk->...ik',xn,y1[-1],y2[-1])
    xn,v = _svd(A,len(A.shape)-2,thresh)
    return ls + [xn,v]
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
        A = np.einsum('...ab,akpr,bkjrq->...jpq',xnL,AL[-1],G[-1])
        xnL,vL = _svd(A,len(A.shape)-3,thresh)
        AL = xkL + [xnL,vL]
    AL[-1] = np.einsum('ajpq,j->apq',AL[-1],ZERO)
    return AL
def _scalar_mult(R,w,angles,thresh=1e-10):
    Rw = _R(w)
    gb = R.copy()
    gb[-1] = np.einsum('aij,jpq,jk->aikpq',gb[-1],CiX,Rw)
    return _oaa(gb,angles,thresh)
def _node_mult(y1,y2,angles,thresh=1e-10):
    xk,xn = _compound(y1,y2,thresh)
    A = np.einsum('...ab,aij,bjk,jpq->...ikpq',xn,y1[-1],y2[-1],CiX)
    xn,v = _svd(A,len(A.shape)-4,thresh)
    return _oaa(xk+[xn,v],angles,thresh)
def _sq(Rs,angles,thresh=1e-10):
    # gb
    n = len(Rs)
    R = Rs.pop() # start with Rn
    u = np.einsum('ymb,ybn->ymnb',R,R)
    for i in range(n-1):
        R = Rs.pop() # Rn-1
        T = np.einsum('xim,ymnb,xnj->xijyb',R,u,R)
        u,s,vh = _svd(T,3,thresh)
        Rs.insert(0,np.einsum('a,ayb->ayb',s,vh))
    Rs.insert(0,u)
    Rs[-1] = np.einsum('ayk,kpq->aypq',Rs[-1],CiX)
    # oaa
    def _SA(angle,RLs,hc=False,thresh=1e-10):
        S = _S(angle)
        AL = RLs.pop(0)
        A = Rs[0].transpose(0,2,1,3).conj() if hc else Rs[0].copy() 
        vh = np.einsum('ixa,ij->jxa',AL,S)
        vh = np.einsum('ixa,xijb->jxab',vh,A)
        for i in range(1,n):
            AL = RLs.pop(0)
            A = Rs[i].conj() if hc else Rs[i].copy()
            if i == n-1:
                A = A.transpose(0,1,3,2) if hc else A
                T = np.einsum('ixab,aypr,byrq->ixypq',vh,AL,A)
            else: 
                T = np.einsum('ixab,ayc,byd->ixycd',vh,AL,A)
            u,s,vh = _svd(T,2,thresh)
            RLs.append(np.einsum('ixa,a->ixa',u,s))
        RLs.append(vh)
        return RLs
    L = len(angles)
    RLs = [Rs[i].copy() for i in range(len(Rs))]
    RLs[0] = np.einsum('xija,i->jxa',RLs[0],ZERO)
    for j in range(L):
        RLs = _SA(angles[L-1-j],RLs,hc=True)
        RLs = _SA(angles[j],RLs,hc=False)
    RLs[0] = np.einsum('ixa,i->xa',RLs[0],ZERO)
    return RLs
     
