import numpy as np
import arithmetic.nn_peps as nn
import arithmetic.compress as cp
import h5py
import itertools
np.set_printoptions(linewidth=200)

N = 5
d = 4
xmin = -1.0
xmax = 1.0
dx = (xmax-xmin)/d
xs = np.arange(xmin,xmax,dx)
tr = np.ones(d)/d
coeff = [0.5,0.25,0,-1.0/48]

mode = 'mps'
def integrate(Ws,Bs,max_bond=None,cutoff=1e-14):
    print('building peps')
    hs = nn.get_layer1(xs,Ws[0],Bs[0],coeff)
    for i in range(1,len(Ws)):
        hs = nn.get_layer(hs,Ws[i],Bs[i],coeff)
    out = []
    print('integrating peps')
    for i in range(len(hs)):
        peps = nn.get_peps(hs[i],N,tr)
        out.append(cp.contract_boundary(peps,from_which='top',
                   max_bond=max_bond,mode=mode,cutoff=cutoff))
    return out

l = 3 # number hidden layers
print('N={},d={},l={}'.format(N,d,l))
# weight and bias
Ws = [np.random.rand(N,N) for i in range(l)] 
Bs = [np.random.rand(N) for i in range(l)] 

# check numerical integration for small case
if N==5 and d==4:
    xls = list(itertools.product(xs,repeat=N))
    out = np.zeros(N)
    for x in xls:
        hs = list(x).copy()
        for i in range(l):
            hs = np.einsum('ij,j->i',Ws[i],np.array(hs))+Bs[i]
#            hs = [1.0/(1.0+np.exp(-h)) for h in hs]
            hs = [np.dot(np.array([h**i for i in range(len(coeff))]),coeff) for h in list(hs)]
        out += np.array(hs)
    out/=d**N
    print(out)

print(integrate(Ws,Bs))
exit()

max_bonds = list(range(2,8))
for max_bond in max_bonds:
    out.append(integrate(Ws,Bs,max_bond=max_bond))
    print('max_bond=',max_bond)
    print(out[-1])
