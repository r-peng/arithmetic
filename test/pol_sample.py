import numpy as np
import scipy.linalg
import quimb.tensor as qtn
from arithmetic.pol import (
    get_sum_scalar,
    trace_pol_compress_row,
    trace_pol_compress_col,
)
from arithmetic.utils import worker_execution
np.set_printoptions(precision=8,suppress=True)
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
print(f'RANK={RANK},SIZE={SIZE}')
if RANK==0:
    N = 10
    tag,new_tag = 'x','a'
    ng = 4
    n = 4
    cutoff = 1e-15
    max_bond = 500 
    tnx = dict()
    sum_ = dict()
    idxs = [np.random.randint(low=0,high=ng) for i in range(N)]
    for k in range(1,n+1):
        xs = {i:np.random.rand(ng) for i in range(1,N+1)}
        
        sum_[k] = sum(np.array([xs[i][idxs[i-1]] for i in range(1,N+1)]))
        tr = {i:np.zeros(ng) for i in range(1,N+1)}
        for i in range(1,N+1):
            tr[i][idxs[i-1]] = 1. 
        
        tnx[k] = get_sum_scalar(xs,tag)
        tny = tnx[k].copy()
        for i in range(1,N+1):
            tny.add_tensor(qtn.Tensor(data=tr[i],inds=(f'{tag}{i}',)))
        data = tny.contract()
        print(f'k={k},check get_field={abs(sum_[k]-data)/sum_[k]}')
    prod = sum([np.log10(sum_[k]) for k in range(1,n+1)])
    print('check row...')
    sign,data = trace_pol_compress_row(tnx,tag,tr,iprint=1,
                                  cutoff=cutoff,max_bond=max_bond)
    print('check trace prod=',abs(prod-data)/abs(prod))
    print('check col...')
    sign,data = trace_pol_compress_col(tnx,tag,new_tag,tr,iprint=1,
                                  cutoff=cutoff,max_bond=max_bond)
    print('check trace prod=',abs(prod-data)/abs(prod))
    for complete_rank in range(1,SIZE):
        COMM.send('finished',dest=complete_rank)
else:
    worker_execution()
