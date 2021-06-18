import numpy as np
import math
X = np.zeros((2,2),dtype=complex)
X[0,1] = X[1,0] = 1.0
Y = np.zeros((2,2),dtype=complex)
Y[0,1] = -1j
Y[1,0] = 1j
Z = np.zeros((2,2),dtype=complex)
Z[0,0] = 1.0
Z[1,1] = -1.0
ONE = np.zeros(2)
ONE[1] = 1.0
ZERO = np.zeros(2)
ZERO[0] = 1.0

#def _U1(phi):
#    U = np.eye(2,dtype=complex)*math.cos(phi)
#    U -= 1j*X*math.sin(phi)
#    return U
#def _COp1(Op): #cico
#    U = np.zeros((4,4),dtype=complex)
#    U[0,0] = U[1,1] = 1.0
#    U[-2:,-2:] = Op.copy()
#    U = np.reshape(U,(2,)*4,order='C')
#    return U
def _U1(a,b):
    U = np.eye(2,dtype=complex)*a
    U += 1j*X*b
    return U
def _COp1(Op): # ccio
    U = np.zeros((2,)*4,dtype=complex)
    U[0,0,...] = np.eye(2,dtype=complex)
    U[1,1,...] = Op.copy()
    return U
def _gb1(U):
    c = np.dot(U,ZERO)
    CiX = _COp1(1j*X)
    U = np.einsum('j,k,jkxy->xy',c,c.conj(),CiX)
    return U
def _gb(U,d):
    for i in range(d):
        U = _gb1(U)
    return U
#CNOT = _COp1(X)
#U0 = np.einsum('ijkl,i,k->jl',CNOT,ZERO,ZERO)
#U1 = np.einsum('ijkl,i,k->jl',CNOT,ONE,ONE)
#print(np.linalg.norm(U1-X))
#print(np.linalg.norm(U0-np.eye(2)))
#exit()

#phi = np.random.rand()
#U1 = _U1(phi)
#CiX = _COp1(-1j*X)
#R = np.einsum('i,ij,jxky,kl,l->xy',ZERO,U1,CiX,U1.T.conj(),ZERO)
#print('cos', math.cos(phi)**2)
#print('sin', math.sin(phi)**2)
#print(R)
#a = np.random.rand()
#b = np.random.rand()
#U1 = _U1(a,b)
#c = np.dot(U1,ZERO)
#CiX = _COp1(-1j*X)
#R = np.einsum('j,k,jkxy->xy',c,c.conj(),CiX)
#print('a,b',a,b)
#print('a^2,b^2',a**2,b**2)
#print(R)
a = np.random.rand()
b = np.random.rand()
print('a,b',a,b)
U1 = _U1(a,b)
R = _gb1(U1)
print('a^2,b^2',a**2,b**2)
print(R)
d = 3
R = _gb(U1,d)
print('a^(2^d),b^(2^d)',a**(2**d),b**(2**d))
print(R)
