import numpy as np
import math
import rus
# test full/contracted rus
ZERO = np.array([1.0,0.0])
x = np.random.rand()
R = rus._Ry(x)
R1 = rus._gb(R)
R2 = rus._gb(R,contr_ancilla=False)
R2 = np.einsum('...ijkl,i,j->...kl',R2,ZERO,ZERO)
print(np.linalg.norm(R1-R2))
w = np.random.rand()
R1 = rus._par_(R,w)
R2 = rus._par(R,w,contr_ancilla=False)
R2 = np.einsum('...ijklmn,i,j,k,l->...mn',R2,ZERO,ZERO,ZERO,ZERO)
print(np.linalg.norm(R1-R2))
# test rotation angle
epsilon = 0.1
print('epsilon = ',epsilon)
#test = 'gb'
test = 'par'
#test = 'add'
if test == 'gb':
    # arctan(tan(x)**(2**d))=x**(2**d)+2**(d)/3*O(x**(2**d+2))
    xs = [np.random.rand()*epsilon for i in range(2)]
    R = rus._Rys(xs)
    for d in range(1,5):
        print('########## d={} ###############'.format(d))
        R = rus._gb(R)
        # check gb
        for i in range(len(xs)):
            exact = xs[i]**2
            appr = math.atan(R[i,1,0]/R[i,0,0])
            print('rel err', abs(exact-appr)/exact)
if test == 'par': 
# arctan(tan(w)tan(x))=wx+O(epsilon**4) 
    xs = [np.random.rand()*epsilon for i in range(2)]
    w = np.random.rand()*epsilon
    R = rus._Rys(xs)
    R = rus._par(R,w) 
    for i in range(len(xs)):
        exact = xs[i]*w
        appr = math.atan(R[i,1,0]/R[i,0,0])
        print('rel err', abs(exact-appr)/exact)
if test == 'add': 
    xs = [np.random.rand()*epsilon for i in range(2)]
    w = np.random.rand()*epsilon
    R = rus._Rys(xs)
    R1 = rus._add(R,R)
    for i in range(len(xs)):
        for j in range(len(xs)):
            exact = xs[i]+xs[j]
            appr = math.atan(R1[i,j,1,0]/R1[i,j,0,0])
            print('rel err', abs(exact-appr)/exact)
    R2 = rus._add_hidden(R,R)
    for i in range(len(xs)):
        exact = xs[i]*2.0
        appr = math.atan(R2[i,1,0]/R2[i,0,0])
        print('rel err', abs(exact-appr)/exact)
