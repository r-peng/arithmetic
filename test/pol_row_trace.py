import numpy as np
import scipy.linalg
import quimb.tensor as qtn
from arithmetic.pol import (
    get_sum,
    trace_pol_compress_row,
)
import itertools
np.set_printoptions(precision=8,suppress=True)

N = 10
tag,new_tag = 'x','a'
ng = 4
n = 10
cutoff = 1e-15
max_bond = 500 
xs = dict()
for k in range(1,n+1):
    xs[k] = {i:np.random.rand(ng) for i in range(1,N+1)}
xs = {k:xs[1] for k in range(1,N+1)}
tnx = dict()
tr = {i:np.ones(ng)/ng for i in range(1,N+1)}
for k in range(1,n+1):
    tnx[k] = get_sum(xs[k],tag,iprint=1,cutoff=cutoff)
sign,data = trace_pol_compress_row(tnx,tag,tr,iprint=1,
                              cutoff=cutoff,max_bond=max_bond)
exit()
print(f'numerical integration...')
ls = list(itertools.product(range(ng),repeat=N))
out = 0.
for idxs in ls:
    prod = 1.
    for k in range(1,n+1):
        prod *= sum([xs[k][i][idxs[i-1]] for i in range(1,N+1)])
    out += prod
out = np.log10(out) - N*np.log10(ng)
print('check trace prod=',abs(out-data)/abs(out))
