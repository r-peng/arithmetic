import numpy as np
import arithmetic as amt
import math, scipy
from scipy import integrate
np.set_printoptions(precision=6,suppress=True)
thresh = 1e-6
bdim = 20
iterate = True
#iterate = False
print('########## exp(x1+...+xn) #############')
n = 3  # number of variable
xmin = -1.0/n
xmax = 1.0/n
p = 10

def _integrate(ngrid):
    print('########### ngrid={} ###########'.format(ngrid))
    dx = (xmax-xmin)/ngrid
    x = np.arange(xmin,xmax,dx)
    coeff = [1]
    for i in range(2,p+1):
        coeff.append(coeff[-1]*i)
    f = [x.reshape(1,len(x),1)]
    for i in range(1,n):
        f = amt._compose1(f,[x.reshape(1,len(x),1)],'+',thresh,bdim,iterate)
    powers = [f]
    for i in range(2,p+1):
        powers.append(amt._compose2(powers[-1],f,n,'*',thresh,bdim,iterate))
        print('power={}, bdim={}'.format(i,amt._get_bdim(powers[-1])))
    assert len(powers)==len(coeff)
    f = amt._compose0(powers[0],1.0,'+',thresh,bdim,iterate)
    for i in range(1,len(powers)):
        tmp = amt._compose0(powers[i],1.0/coeff[i],'*',thresh,bdim,iterate)
        f = amt._compose2(f,tmp,n,'+',thresh,bdim,iterate)
        print('order={}, bdim={}'.format(i+1,amt._get_bdim(f)))
#    i = np.zeros(len(x))
#    i[-1] = 1.0
#    exact = np.exp(sum([x[-1] for i in range(n)]))
#    out = amt._contract(f,[i for j in range(n)])[0,0]
    out = amt._contract(f,[np.ones(len(x)) for j in range(n)])[0,0]
    return out*(dx**n)
def func(*args):
    return np.exp(sum(args))
out2, err = scipy.integrate.nquad(func,[[xmin,xmax] for i in range(n)])
for ngrid in [10,20,50,100,1000,2000,5000,10000]:
#for ngrid in [10,20,50]:
    out1 = _integrate(ngrid)
    print('err',abs(out1-out2))
exit()
print('########## cos(x1+...+xn) #############')
n = 12  # number of variable
xmin = -1.0/n
xmax = 1.0/n
ngrid = 20
x = np.arange(xmin,xmax,(xmax-xmin)/ngrid)

p = 4
coeff = []
for i in range(p+1):
    coeff.append((-1)**i/math.factorial(2*i))
f = [x.reshape(1,len(x),1)]
for i in range(1,n):
    f = amt._compose1(f,[x.reshape(1,len(x),1)],'+',thresh,bdim,iterate)
powers = [None,amt._compose2(f,f,n,'*',thresh,bdim,iterate)]
for i in range(2,p+1):
    tmp = amt._compose2(powers[-1],f,n,'*',thresh,bdim,iterate)
    powers.append(amt._compose2(tmp,f,n,'*',thresh,bdim,iterate))
    print('power={}, bdim={}'.format(i*2,amt._get_bdim(powers[-1])))
assert len(powers)==len(coeff)
tmp = amt._compose0(powers[1],coeff[1],'*',thresh,bdim,iterate)
f = amt._compose0(tmp,coeff[0],'+',thresh,bdim,iterate)
for i in range(2,len(powers)):
    tmp = amt._compose0(powers[i],coeff[i],'*',thresh,bdim,iterate)
    f = amt._compose2(f,tmp,n,'+',thresh,bdim,iterate)
    print('order={}, bdim={}'.format(2*i,amt._get_bdim(f)))
i = np.zeros(len(x))
i[-1] = 1.0
out1 = amt._contract(f,[i for j in range(n)])[0,0]
exact = np.cos(sum([x[-1] for i in range(n)]))
print('err',abs(out1-exact))
def func(*args):
    return np.exp(sum(args))
exit()
#out1 = amt._contract(exp,[np.ones(len(xs)) for i in range(n)])[0,0]*(dx**n)
exit()
#out1 = amt._contract(exp,[np.ones(len(xs)) for i in range(n)])[0,0]*(dx**n)
out2, err = scipy.integrate.nquad(func,[[xmin,xmax] for i in range(n)])
print(out1,out2)
