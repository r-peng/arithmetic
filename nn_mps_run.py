import numpy as np
import arithmetic.nn_mps as nn
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

form = 'left'
cutoff_mode = 'rel'
def integrate(Ws,Bs,max_bond=None,cutoff=1e-14):
    hs = []
    out = []
    print('layer=',0)
    hs.append(nn.get_layer1(xs,Ws[0],Bs[0],coeff,form=form,max_bond=max_bond,cutoff=cutoff,cutoff_mode=cutoff_mode))
    for i in range(1,len(Ws)):
        print('layer=',i)
        hs.append(nn.get_layer(hs[-1],Ws[i],Bs[i],coeff,form=form,max_bond=max_bond,cutoff=cutoff,cutoff_mode=cutoff_mode))
    out = [nn.integrate(hs[-1][i].copy(),tr) for i in range(len(hs[-1]))]
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
