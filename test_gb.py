import numpy as np
import math, statistics
import gb
d = 10
epsilon = 0.01
L = 10 
thresh = 1e-12

ZERO = np.array([1,0],dtype=complex)
X = np.array([[0,1],[1,0]],dtype=complex)
CiX = np.zeros((2,)*3,dtype=complex)
CiX[0,...] = np.eye(2,dtype=complex)
CiX[1,...] = 1j*X
angles = gb._angles(L)

#print('########### check decomp ##############')
#x = [np.random.rand() for i in range(d)]
#R = gb._Rs(x,thresh)
#R_ = np.einsum('xa,aij->xij',R[0],R[1])
#err = 0.0
#for i in range(d):
#    R = R_[i,...]
#    tan = 1j*R[1,0]/R[0,0]
#    angle = math.atan(tan.real)
#    err += abs(x[i]-angle)
#    U = gb._R(angle)
#    tmp = np.divide(R,U).flatten()
#    err += abs(1.0-tmp[0])
#    err += statistics.pstdev(list(tmp.real))
#    err += statistics.pstdev(list(tmp.imag))
#print(err)
#exit()
#print('########### check addition ##############')
#x1 = [np.random.rand()*epsilon for i in range(d)]
#x2 = [np.random.rand()*epsilon for i in range(d)]
#R1 = gb._Rs(x1,thresh)
#R2 = gb._Rs(x2,thresh)
#R12 = gb._add_input(R1,R2,thresh)
#print(R12[0].shape, R12[1].shape, R12[2].shape)
#R12_ = np.einsum('xa,ayb,bij->xyij',R12[0],R12[1],R12[2])
#err = 0.0
#for i in range(d):
#    for j in range(d):
#        R = R12_[i,j,...]
#        tan = 1j*R[1,0]/R[0,0]
#        angle = math.atan(tan.real)
#        err += abs(tan.imag)
#        err += abs(x1[i]+x2[j]-angle)
#        U = gb._R(angle)
#        tmp = np.divide(R,U).flatten()
#        err += abs(1.0-tmp[0])
#        err += statistics.pstdev(list(tmp.real))
#        err += statistics.pstdev(list(tmp.imag))
#print(err)
#R12 = gb._add_hidden(R12,R12,thresh)
#print(R12[0].shape, R12[1].shape, R12[2].shape)
#R12_ = np.einsum('xa,ayb,bij->xyij',R12[0],R12[1],R12[2])
#err = 0.0
#for i in range(d):
#    for j in range(d):
#        R = R12_[i,j,...]
#        tan = 1j*R[1,0]/R[0,0]
#        angle = math.atan(tan.real)
#        err += abs(tan.imag)
#        err += abs((x1[i]+x2[j])*2-angle)
#        U = gb._R(angle)
#        tmp = np.divide(R,U).flatten()
#        err += abs(1.0-tmp[0])
#        err += statistics.pstdev(list(tmp.real))
#        err += statistics.pstdev(list(tmp.imag))
#print(err)
#R12 = gb._add_hidden(R12,R12,thresh)
#print(R12[0].shape, R12[1].shape, R12[2].shape)
#R12_ = np.einsum('xa,ayb,bij->xyij',R12[0],R12[1],R12[2])
#err = 0.0
#for i in range(d):
#    for j in range(d):
#        R = R12_[i,j,...]
#        tan = 1j*R[1,0]/R[0,0]
#        angle = math.atan(tan.real)
#        err += abs(tan.imag)
#        err += abs((x1[i]+x2[j])*4-angle)
#        U = gb._R(angle)
#        tmp = np.divide(R,U).flatten()
#        err += abs(1.0-tmp[0])
#        err += statistics.pstdev(list(tmp.real))
#        err += statistics.pstdev(list(tmp.imag))
#print(err)
#exit()
print('################ check multiplication ################')
def _num_oaa(x):
    A = np.array([[x,math.sqrt(1-x**2)],
               [math.sqrt(1-x**2),-x]])
    AL = np.array(A,dtype=complex)
    for j in range(L):
        Sl = np.array([np.exp(1j*angles[j]),1],dtype=complex)
        Sl = np.diag(Sl)
        j_ = L-1-j
        Sr = np.array([np.exp(1j*angles[j_]),1],dtype=complex)
        Sr = np.diag(Sr)
        G = np.linalg.multi_dot([A,Sl,A,Sr])
        AL = np.dot(G,AL)
    return AL[0,0]
x1 = [np.random.rand()*epsilon for i in range(d)]
x2 = [np.random.rand()*epsilon for i in range(d)]
R1 = gb._Rs(x1,thresh)
R2 = gb._Rs(x2,thresh)
R12 = gb._add_input(R1,R2,thresh)
w = np.random.rand()*epsilon
R12w = gb._scalar_mult(R12,w,[],thresh)
R12wL = gb._scalar_mult(R12,w,angles,thresh)
R12sq = gb._node_mult(R12,R12,[],thresh)
R12sqL = gb._node_mult(R12,R12,angles,thresh)
print(R12wL[0].shape,R12wL[1].shape,R12wL[2].shape)
print(R12sqL[0].shape,R12sqL[1].shape,R12sqL[2].shape)
def gb_err(y,yL,f):
    err = 0
    angs = []
    phases = []
    for i in range(d):
        i1 = np.zeros(d,dtype=complex)
        i1[i] = 1.0
        R1 = np.einsum('xa,x->a',y[0],i1)
        R1L = np.einsum('xa,x->a',yL[0],i1)
        for j in range(d):
            i2 = np.zeros(d,dtype=complex)
            i2[j] = 1.0
            R2 = np.einsum('axb,x->ab',y[1],i2)
            R2L = np.einsum('axb,x->ab',yL[1],i2)
    
            R = np.einsum('a,ab,bpq->pq',R1,R2,y[-1])
            RL = np.einsum('a,ab,bpq->pq',R1L,R2L,yL[-1])
            tan = 1j*R[1,0]/R[0,0]
            tanL = 1j*RL[1,0]/RL[0,0]
            angle = math.atan(tan.real)
            angleL = math.atan(tanL.real)
            err += abs(angle-angleL)
            err += abs(tan.imag)
            err += abs(tanL.imag)
            exact = f(x1[i],x2[j]) 
            angs.append(abs((exact-angle)/exact))
            U = gb._R(angle)
            tmp = np.divide(R,U).flatten()
            tmpL = np.divide(RL,U).flatten()
            err += tmp[0].imag
    #        err += statistics.pstdev(list(tmp.real))
    #        err += statistics.pstdev(list(tmp.imag))
    #        err += statistics.pstdev(list(tmpL.real))
    #        err += statistics.pstdev(list(tmpL.imag))
            amp = _num_oaa(tmp[0].real)
            err += abs(tmpL[0]-amp)
            phases.append(tmpL[0])
    phases = np.array(phases,dtype=complex)
    print(err)
    print(statistics.mean(list(phases.real)),statistics.pstdev(list(phases.real)))
    print(statistics.mean(list(phases.imag)),statistics.pstdev(list(phases.imag)))
    print(statistics.mean(angs))
def f1(x1,x2):
    return w*(x1+x2)
def f2(x1,x2):
    return (x1+x2)**2
gb_err(R12w,R12wL,f1)
gb_err(R12sq,R12sqL,f2)
exit()
print('########### check SVD #########################')
##print(R1.shape, R2.shape)
x1 = [np.random.rand()*epsilon for i in range(d)]
x2 = [np.random.rand()*epsilon for i in range(d)]
x3 = [np.random.rand()*epsilon for i in range(d)]
R1 = gb._Rs(x1)
R2 = gb._Rs(x2)
R3 = gb._Rs(x3)
thresh = 1e-6
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
