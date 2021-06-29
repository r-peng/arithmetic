import numpy as np
import math
#import quimb.tensor as qtn
#import cotengra as ctg
np.set_printoptions(precision=3,suppress=True)
X = np.array([[0,1],[1,0]],dtype=complex)
ZERO = np.array([1,0],dtype=complex)
CiX = np.zeros((2,)*4,dtype=complex)
CiX[0,0,...] = np.eye(2,dtype=complex)
CiX[1,1,...] = 1j*X
def _R(x): # exp(-iX*x)
    ls = []
    for i in range(len(x)):
        U  = np.eye(2,dtype=complex)*math.cos(x[i])
        U -= 1j*X*math.sin(x[i])
        ls.append(U)
    return np.array(ls,dtype=complex)
# SVD of GB(1)
d = 3
thresh = 1e-6
x1 = [np.random.rand() for i in range(d)]
x2 = [np.random.rand() for i in range(d)]
R1 = _R(x1)
R2 = _R(x2)
##print(R1.shape, R2.shape)
M = np.einsum('iak,ilc->iackl',R1,R1)
N = np.einsum('jkb,jbl->jbkl',R2,R2) 
A = np.einsum('iackl,jbkl->iacjb',M,N)
dimM = len(x1)*2*2 
dimN = len(x2)*2
A = A.reshape((dimM,dimN))
Q,R = np.linalg.qr(A)
print(R)
print(np.dot(A[:,0].conj(),A[:,1]))
err = 0
for i in range(len(x2)):
    for j in range(len(x2)):
        err += np.dot(A[:,2*i].conj(),A[:,2*j+1])
print(err)
#us = []
#es = []
#def proj(u,a):
#    return 
#for k in range(dimN):
#    uk = A[:,k].copy()
#    for j in range(len(us)):
#        uk -= np.dot(us[j].conj(),A[:,k])/np.dot(us[j].conj(),us[j])*us[j]
#    ek = uk/np.linalg.norm(uk)
#    us.append(uk)
#    es.append(ek)
#    print('u{}'.format(k),uk)
#    print('e{}'.format(k),ek)
#R = np.zeros((dimN,dimN))
#for k in range(dimN):
#    for j in range(k+1):
#        print(j,k,np.dot(es[j].conj(),A[:,k]))
