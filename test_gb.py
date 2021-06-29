import numpy as np
import math, statistics
import gb
d = 3
epsilon = 0.01
L = 10 

ZERO = np.array([1,0],dtype=complex)
X = np.array([[0,1],[1,0]],dtype=complex)
CiX = np.zeros((2,)*3,dtype=complex)
CiX[0,...] = np.eye(2,dtype=complex)
CiX[1,...] = 1j*X
angles = gb._angles(L)

#x = [np.random.rand()*epsilon for i in range(d)]
#R = gb._Rs(x)
#w = np.random.rand()*epsilon
#R = gb._wx(R,w,angles)
#err = 0
#angs = []
#phases = []
#for i in range(len(x)):
#    tan = 1j*R[i,1,0]/R[i,0,0]
#    angle = math.atan(tan.real)
#    err += tan.imag
#    print(i, w*x[i], angle, abs((w*x[i]-angle)/(w*x[i])))
#    angs.append(abs((w*x[i]-angle)/(w*x[i])))
#    U = gb._R(angle)
#    tmp = np.divide(R[i,...],U).flatten()
#    err += statistics.pstdev(list(tmp.real))
#    err += statistics.pstdev(list(tmp.imag))
#    phases.append(tmp[0])
#phases = np.array(phases,dtype=complex)
#print(err)
#print(statistics.mean(list(phases.real)),statistics.pstdev(list(phases.real)))
#print(statistics.mean(list(phases.imag)),statistics.pstdev(list(phases.imag)))
#print(statistics.mean(angs))
#exit()
##print(R1.shape, R2.shape)
x1 = [np.random.rand()*epsilon for i in range(d)]
x2 = [np.random.rand()*epsilon for i in range(d)]
x3 = [np.random.rand()*epsilon for i in range(d)]
R1 = gb._Rs(x1)
R2 = gb._Rs(x2)
R3 = gb._Rs(x3)
thresh = 1e-6
print('########### check SVD #########################')
ls1 = gb._sq([R1,R2,R3],[])
ls2 = gb._sq([R1,R2,R3],angles)
err1 = 0
err2 = 0
err = 0.0
phases = []
angs = [] 
for i in range(d):
    i1 = np.zeros(d,dtype=complex)
    i1[i] = 1.0
    R1_1 = np.einsum('xa,x->a',ls1[0],i1)
    R1_2 = np.einsum('xa,x->a',ls2[0],i1)
    for j in range(d):
        i2 = np.zeros(d,dtype=complex)
        i2[j] = 1.0
        R2_1 = np.einsum('axb,x->ab',ls1[1],i2)
        R2_2 = np.einsum('axb,x->ab',ls2[1],i2)
        for k in range(d):
            i3 = np.zeros(d,dtype=complex)
            i3[k] = 1.0
            R3_1 = np.einsum('axpq,x->apq',ls1[2],i3)
            R3_2 = np.einsum('axpq,x->apq',ls2[2],i3)

            R1 = np.einsum('a,ab,bpq->pq',R1_1,R2_1,R3_1)
            tan1 = 1j*R1[1,0]/R1[0,0]
            angle1 = math.atan(tan1.real)
            err1 += tan1.imag
            R2 = np.einsum('a,ab,bpq->pq',R1_2,R2_2,R3_2)
            tan2 = 1j*R2[1,0]/R2[0,0]
            angle2 = math.atan(tan2.real)
            err2 += tan2.imag
            exact = (x1[i]+x2[j]+x3[k])**2
#            print(exact, angle1, abs((exact-angle1)/exact))
#            print(exact, angle2, abs((exact-angle2)/exact))
            angs.append(abs((exact-angle2)/exact))
#            err += abs((angle2-angle1)/angle1)
            U = gb._R(angle2)
            tmp = np.divide(R2,U).flatten()
#            print(tmp)
            err += statistics.pstdev(list(tmp.real))
            err += statistics.pstdev(list(tmp.imag))
            phases.append(tmp[0])
phases = np.array(phases,dtype=complex)
print(err, err1, err2)
print(statistics.mean(list(phases.real)),statistics.pstdev(list(phases.real)))
print(statistics.mean(list(phases.imag)),statistics.pstdev(list(phases.imag)))
print(statistics.mean(angs))
exit()
print('############# SVD1 ###########')
A = np.einsum('ymk,ykn->ymnk',R3,R3) 
A = np.einsum('xim,ymnk,xnj->xijyk',R2,A,R2)
tdim = A.shape
mdim = tdim[0]*tdim[1]*tdim[2], tdim[3]*tdim[4]
A = np.reshape(A,mdim)
A,s,M3 = np.linalg.svd(A)
s_ = []
for i in range(len(s)):
    if s[i]>thresh:
        s_.append(math.sqrt(s[i]))
print(s_)
D23 = len(s_)
A = np.einsum('id,d->id',A[:,:D23],np.array(s_))
M3 = np.einsum('di,d->di',M3[:D23,:],np.array(s_))
A = np.reshape(A,(len(x2),2,2,D23))
M3 = np.reshape(M3,(D23,)+tdim[3:])

A = np.einsum('xim,ymnk,xnj->xijyk',R1,A,R1)
dim1 = len(x1)*2*2 
dim2 = len(x2)*D23
A = np.reshape(A,(dim1,dim2))
A,s,M2 = np.linalg.svd(A)
s_ = []
for i in range(len(s)):
    if s[i]>thresh:
        s_.append(math.sqrt(s[i]))
print(s_)
D12 = len(s_)
A = np.einsum('id,d->id',A[:,:D12],np.array(s_))
M2 = np.einsum('di,d->di',M2[:D12,:],np.array(s_))
A = np.reshape(A,(len(x1),2,2,D12))
M2 = np.reshape(M2,(D12,len(x2),D23))
print('########### check SVD #########################')
R = np.einsum('i,j,xija,ayb,bzc,cpq->xyzpq',ZERO,ZERO,A,M2,M3,CiX)
err = 0
for i in range(len(x1)):
    for j in range(len(x2)):
        for k in range(len(x3)):
            tan = 1j*R[i,j,k,1,0]/R[i,j,k,0,0]
            angle = math.atan(tan.real)
            err += tan.imag
            exact = (x1[i]+x2[j]+x3[k])**2
            print(exact, angle, abs((exact-angle)/exact))
print(err)
print('############# SVD2 ###########')
A = np.einsum('ymk,ykn->ymnk',R3,R3) 
A = np.einsum('xim,ymnk,xnj->xijyk',R2,A,R2)
dim1 = len(x2)*2*2 
dim2 = len(x3)*2
A = np.reshape(A,(dim1,dim2))
A,s,M3 = np.linalg.svd(A)
s_ = s[s>thresh]
print(s_)
D23 = len(s_)
M3 = np.einsum('di,d->di',M3[:D23,:],np.array(s_))
A = np.reshape(A[:,:D23],(len(x2),2,2,D23))
M3 = np.reshape(M3,(D23,len(x3),2))

A = np.einsum('xim,ymnk,xnj->xijyk',R1,A,R1)
dim1 = len(x1)*2*2 
dim2 = len(x2)*D23
A = np.reshape(A,(dim1,dim2))
A,s,M2 = np.linalg.svd(A)
s_ = []
for i in range(len(s)):
    if s[i]>thresh:
        s_.append(s[i])
print(s_)
D12 = len(s_)
M2 = np.einsum('di,d->di',M2[:D12,:],np.array(s_))
A = np.reshape(A[:,:D12],(len(x1),2,2,D12))
M2 = np.reshape(M2,(D12,len(x2),D23))
print('########### check SVD #########################')
R = np.einsum('i,j,xija,ayb,bzc,cpq->xyzpq',ZERO,ZERO,A,M2,M3,CiX)
err = 0
for i in range(len(x1)):
    for j in range(len(x2)):
        for k in range(len(x3)):
            tan = 1j*R[i,j,k,1,0]/R[i,j,k,0,0]
            angle = math.atan(tan.real)
            err += tan.imag
            exact = (x1[i]+x2[j]+x3[k])**2
            print(exact, angle, abs((exact-angle)/exact))
print(err)
exit()
print('############# SVD2 ###########')
T = np.einsum('iak,jkb->ijab',R1,R2)
B = np.einsum('jdk,ikc->jidc',R2,R1)
A = np.einsum('ijab,jidc->iacjdb',T,B)
dim1 = len(x1)*2*2
dim2 = len(x2)*2*2
A = np.reshape(A,(dim1,dim2))
u,s,vh = np.linalg.svd(A)
s_ = []
for i in range(len(s)):
    if s[i]>thresh:
        s_.append(s[i])
print(s_)
