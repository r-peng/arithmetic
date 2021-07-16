import numpy as np
import math, scipy
from scipy import integrate
import arithmetic, example
np.set_printoptions(precision=6,suppress=True)
thresh = 1e-12
bdim = 20
max_iter = 10

xmin = -1.0
xmax = 1.0
p = 10
def _integrate(n,ng):
    print('################### n={},ngrid={} #############'.format(n,ng))
    mu = [0 for i in range(n)]
    sigma = np.random.rand(n)*10
    #sigma = np.ones(n)
    tmp = np.random.rand(n,n)
    tmp -= tmp.T
    tmp = scipy.linalg.expm(tmp)
    sigma = np.linalg.multi_dot([tmp,np.diag(sigma),tmp.T])
    sigma_inv = np.linalg.inv(sigma)

    dx = (xmax-xmin)/ng
    xs = [np.arange(xmin,xmax,dx) for i in range(n)]
    tn2 = example._gauss2(mu,sigma,xs,p,thresh,bdim,max_iter)

    ins = []
    x = []
    for i in range(n):
        idx = np.random.randint(low=0,high=len(xs[i]))
        tmp = np.zeros_like(xs[i])
        tmp[idx] = 1.0
        ins.append(tmp)
        x.append(xs[i][idx])

    out2 = arithmetic._contract(tn2,ins)
    def f(*args):
        x = np.array(args).flatten()-mu
        exponent = -0.5*np.einsum('i,ij,j->',x,sigma_inv,x)
        return np.exp(exponent)/math.sqrt(np.linalg.det(sigma)*(2*math.pi)**n)
    out = f(x) 
    print('single point err={},bdim={}'.format(abs(out-out2),arithmetic._get_bdim(tn2)))
    print('integration by contraction')
    out2 = arithmetic._contract(tn2,[np.ones_like(xs[i]) for i in range(n)])*(dx**n)
    print('integration in scipy')
    out, __ = scipy.integrate.nquad(f,[(xmin,xmax) for i in range(n)])
    print('integration err={}'.format(abs(out-out2)))

for ng in [10,20,50,100,500,1000]:
    _integrate(3,ng)
exit()
for n in [3,5,10,20]:
    _integrate(n,10)
