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
def _S(a):
    return np.array([[np.exp(1j*a),0],[0,1]],dtype=complex)
def _R(x): # Rio=[exp(-iX*x)]io
    R  = np.eye(2,dtype=complex)*math.cos(x)
    R -= 1j*X*math.sin(x)
    return R
def _Rs(x): # Rxio=[exp(-iX*x)]io
    R = np.zeros((len(x),2,2),dtype=complex)
    for i in range(len(x)):
        R[i,...] = _R(x[i])
    return R
def _wx(R,w,angles):
    Rw = _R(w)
    # gb
    gb = np.einsum('xik,kj,kpq->xijpq',R,Rw,CiX)
    # oaa
    def _SA(angle,A,AL):
        S = _S(angle)
        AL = np.einsum('xipq,ij->xjpq',AL,S)
        return np.einsum('xipr,xijrq->xjpq',AL,A)
    L = len(angles)
    AL = np.einsum('xijpq,i->xjpq',gb,ZERO)
    for j in range(L):
        AL = _SA(angles[L-1-j],gb.transpose(0,2,1,4,3).conj(),AL)
        AL = _SA(angles[j], gb, AL)
    return np.einsum('xipq,i->xpq',AL,ZERO)
def _svd(T,naxis,thresh=1e-10): # svd after the 1st naxis
    tdim = T.shape
    mdim1 = reduce(operator.mul,tdim[:naxis]) 
    mdim2 = reduce(operator.mul,tdim[naxis:])
    u,s,vh = np.linalg.svd(T.reshape((mdim1,mdim2)))
    s = s[s>thresh]
    u = np.reshape(u[:,:len(s)],tdim[:naxis]+(len(s),))
    vh = np.reshape(vh[:len(s),:],(len(s),)+tdim[naxis:])
    return u, s, vh
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
def _mult_y(R1s,R2s,angles,thresh_1e-10):
    n = len(R1s)
     
