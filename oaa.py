import numpy as np
import math
import quimb.tensor as qtn
import cotengra as ctg

def _S(a):
    U = np.eye(2,dtype=complex)
    U[0,0] = np.exp(1j*a)
    return U
def _angles(L,gamma=0.5):
    phi = []
    fac = math.sqrt(1-gamma**2)
    for j in range(L):
        tmp = math.tan(2*math.pi*(j+1)/(2*L+1))*fac
        phi.append(2*math.atan(tmp)-math.pi)
    return phi
#def _oaa(A):
    # A_x1/...n,io
