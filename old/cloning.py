import numpy as np
import math

def _get_qbit(a=None,b=None,normalize=True):
    a = np.random.rand() + 1j*np.random.rand() if a is None else a
    b = np.random.rand() + 1j*np.random.rand() if b is None else b
    if normalize: 
        tmp = a*np.conj(a) + b*np.conj(b)
        a /= math.sqrt(tmp.real)
        b /= math.sqrt(tmp.real)
        tmp = a*np.conj(a) + b*np.conj(b)
    x = np.array([a,b],dtype=complex)
    x_ = np.array([-np.conj(b),np.conj(a)],dtype=complex)
    return x, x_

QID = np.zeros()

BH = np.zeros((2,)*4,dtype=complex) # o1,2,3/i
BH[0,0,1,0] =   math.sqrt(2.0/3.0)
BH[1,0,0,0] = - math.sqrt(1.0/6.0)
BH[0,1,0,0] = - math.sqrt(1.0/6.0)
BH[1,1,0,1] = - math.sqrt(2.0/3.0)
BH[1,0,1,1] =   math.sqrt(1.0/6.0)
BH[0,1,1,1] =   math.sqrt(1.0/6.0)
#BH[1,0,1,1] =   math.sqrt(1.0/12.0)
#BH[0,1,1,1] =   math.sqrt(1.0/12.0)
normalize = True
#normalize = False
psi, psi_ = _get_qbit(normalize=normalize)
out1 = np.einsum('abcd,d->abc',BH,psi)
out2  = np.einsum('a,b,c->abc',psi,psi,psi_)*math.sqrt(2.0/3.0)
out2 -= np.einsum('a,b,c->abc',psi,psi_,psi)*math.sqrt(1.0/6.0)
out2 -= np.einsum('a,b,c->abc',psi_,psi,psi)*math.sqrt(1.0/6.0)
print('err: ', np.linalg.norm(out1-out2))
