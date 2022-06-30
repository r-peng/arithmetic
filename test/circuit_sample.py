import numpy as np
import quimb.tensor as qtn
from arithmetic.circuit import build,contract_norm,get_phase_map,evaluate
import itertools

nl = 4
theta_map = {i:np.random.rand() for i in range(1,nl+1)}
N = (2**nl-1)*2
ng = 4
xs,ws = np.polynomial.legendre.leggauss(ng)

idxs = [np.random.randint(low=0,high=ng) for i in range(N)]
phase_map = get_phase_map(nl*2,theta_map)
out = evaluate([xs[idx] for idx in idxs],phase_map)

tr = {i:np.zeros(ng) for i in range(1,N+1)}
for i in range(N):
    tr[i+1][idxs[i]] = 1.
tn = build({i:xs for i in range(1,N+1)},theta_map)
print(tn)
for i in range(1,N+1):
    tn.add_tensor(qtn.Tensor(data=tr[i],inds=(f'x{i}',)))
out1 = tn.contract()
print('out2=',out)
print('out1=',out1)
exit()
ls = itertools.product(range(2),repeat=2*nl)
out1_ = 0.
for idxs in ls:
    #print(idxs,out1.data[idxs])
    if sum(idxs)>0:
        out1_ += out1.data[idxs]
print(out1_)
