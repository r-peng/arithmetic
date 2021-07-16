import numpy as np
import arithmetic as amt
np.set_printoptions(precision=6,suppress=True)
d = 10
thresh = 1e-12
bdim = 10
max_iter = 50

print('###### _compose0 #####')
x = np.random.rand(d)
i = np.zeros(d)
i[0] = 1

a = np.random.rand()
f = [x.reshape(1,d,1)]
f = amt._add0(f,a,thresh,bdim,max_iter)
out = amt._contract(f,[i])
print('+',abs(out[0,0]-(a+x[0])))
f = [x.reshape(1,d,1)]
f = amt._multiply0(f,a,thresh,bdim,max_iter)
out = amt._contract(f,[i])
print('*',abs(out[0,0]-a*x[0]))
print('######### _compose1 ###########')
x1 = np.random.rand(d)
x2 = np.random.rand(d)
i1 = np.zeros(d)
i1[0] = 1
i2 = np.zeros(d)
i2[1] = 1

f = [x1.reshape(1,d,1)]
g = [x2.reshape(1,d,1)]
h = amt._add1(f,g,thresh,bdim,max_iter)
out = amt._contract(h,[i1,i2])
print('+',abs(out[0,0]-(x1[0]+x2[1])),amt._get_bdim(h))
f = [x1.reshape(1,d,1)]
g = [x2.reshape(1,d,1)]
h = amt._multiply1(f,g,thresh,bdim,max_iter)
out = amt._contract(h,[i1,i2])
print('*',abs(out[0,0]-x1[0]*x2[1]),amt._get_bdim(h))
print('######### _compose2 ###########')
x1 = np.random.rand(d)
x2 = np.random.rand(d)
x3 = np.random.rand(d)
i1 = np.zeros(d)
i1[0] = 1
i2 = np.zeros(d)
i2[1] = 1
i3 = np.zeros(d)
i3[2] = 1

f1 = [x1.reshape(1,d,1)]
f2 = [x2.reshape(1,d,1)]
f3 = [x3.reshape(1,d,1)]
f = amt._add1(f1,f2,thresh,bdim,max_iter)
g = amt._add1(f1,f3,thresh,bdim,max_iter)
h = amt._add2(f,g,1,thresh,bdim,max_iter)
out = amt._contract(h,[i2,i1,i3])
print('+',abs(out[0,0]-(2*x1[0]+x2[1]+x3[2])),amt._get_bdim(h))
f1 = [x1.reshape(1,d,1)]
f2 = [x2.reshape(1,d,1)]
f3 = [x3.reshape(1,d,1)]
f = amt._add1(f1,f2,thresh,bdim,max_iter)
g = amt._add1(f1,f3,thresh,bdim,max_iter)
h = amt._multiply2(f,g,1,thresh,bdim,max_iter)
out = amt._contract(h,[i2,i1,i3])
print('*',abs(out[0,0]-(x1[0]+x2[1])*(x1[0]+x3[2])),amt._get_bdim(h))
h2 = amt._add2(h,h,3,thresh,bdim,max_iter)
out = amt._contract(h2,[i2,i1,i3])
print('*',abs(out[0,0]-2*(x1[0]+x2[1])*(x1[0]+x3[2])),amt._get_bdim(h2))
hsq = amt._multiply2(h,h,3,thresh,bdim,max_iter)
out = amt._contract(hsq,[i2,i1,i3])
print('*',abs(out[0,0]-((x1[0]+x2[1])*(x1[0]+x3[2]))**2),amt._get_bdim(hsq))
print('########## exp(x1+...+xn) #############')
import math
p = 10
a = [1.0/math.factorial(i) for i in range(p+1)]

n = 10 
d = 10
thresh = 1e-12
bdim = 20
max_iter = 50
x = np.random.rand(n,d)/n
i = np.zeros(d)
i[0] = 1
ins = [i for j in range(n)]
tn = [x[0,:].reshape(1,d,1)]
for i in range(1,n):
    tn = amt._add1(tn,[x[i,:].reshape(1,d,1)],thresh,bdim,max_iter)
out = amt._contract(tn,ins)
print('err',abs(out[0,0]-sum(x[:,0])),amt._get_bdim(tn))
#exit()
f1 = amt._poly1(tn,a,thresh,bdim,max_iter)
f2 = amt._poly2(tn,a,thresh,bdim,max_iter)
out1 = amt._contract(f1,ins)
out2 = amt._contract(f2,ins)
print('err1',abs(out1[0,0]-np.exp(sum(x[:,0]))))
print('err2',abs(out2[0,0]-np.exp(sum(x[:,0]))))
