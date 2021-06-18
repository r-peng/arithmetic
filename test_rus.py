import numpy as np
import math
import rus

epsilon = 0.1
print('epsilon = ',epsilon)
test = 'gb'
#test = 'par'
#test = 'add'
if test == 'gb':
    # arctan(tan(x)**(2**d))=x**(2**d)+2**(d)/3*O(x**(2**d+2))
    xs = [np.random.rand()*epsilon for i in range(2)]
    R = rus._Rys(xs)
    print('########## d=1 ###############')
    R = rus._gb(R)
    # check gb
    for i in range(len(xs)):
        exact = xs[i]**2
        appr = math.atan(R[i,1,0]/R[i,0,0])
        print('rel err', abs(exact-appr)/exact)
    print('########## d=2 ###############')
    R = rus._gb(R)
    for i in range(len(xs)):
        exact = xs[i]**(2**2)
        appr = math.atan(R[i,1,0]/R[i,0,0])
        print('rel err', abs(exact-appr)/exact)
    print('########## d=3 ###############')
    R = rus._gb(R)
    for i in range(len(xs)):
        exact = xs[i]**(2**3)
        appr = math.atan(R[i,1,0]/R[i,0,0])
        print('rel err', abs(exact-appr)/exact)
    print('########## d=4 ###############')
    R = rus._gb(R)
    for i in range(len(xs)):
        exact = xs[i]**(2**4)
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
