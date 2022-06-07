import numpy as np
import scipy.linalg
import quimb.tensor as qtn
from arithmetic.pol import (
    get_sum,
    trace_pol_compress_row,
    trace_pol_compress_col,
)
import itertools
from arithmetic.utils import worker_execution
np.set_printoptions(precision=8,suppress=True)
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
print(f'RANK={RANK},SIZE={SIZE}')
if RANK==0:
    N = 6
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
    if SIZE==1:
        print('check row...')
        sign,data = trace_pol_compress_row(tnx,tag,tr,iprint=1,
                                      cutoff=cutoff,max_bond=max_bond)
    else:
        print('check col...')
        sign,data = trace_pol_compress_col(tnx,tag,new_tag,tr,iprint=1,
                                      cutoff=cutoff,max_bond=max_bond)
    print(f'numerical integration...')
    ls = list(itertools.product(range(ng),repeat=N))
    idxs = ls[0]
    prod = 0.
    for k in range(1,n+1):
        prod += np.log10(sum([xs[k][i][idxs[i-1]] for i in range(1,N+1)]))
    out = prod
    for idxs in ls[1:]:
        prod = 0.
        for k in range(1,n+1):
            prod += np.log10(sum([xs[k][i][idxs[i-1]] for i in range(1,N+1)]))
        out += np.log10(1.+10.**(prod-out))
    out -= N*np.log10(ng)
    print('check trace prod=',abs(out-data)/abs(out))
    for complete_rank in range(1,SIZE):
        COMM.send('finished',dest=complete_rank)
else:
    worker_execution()
