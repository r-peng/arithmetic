import numpy as np
import scipy.linalg
import cotengra as ctg
import quimb.tensor as qtn
from arithmetic.pol import (
    get_peps,
    insert_projectors,
    compress_row,
    compress_col,
)
import itertools
from arithmetic.utils import parallelized_looped_function,worker_execution
np.set_printoptions(precision=8,suppress=True)
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
print(f'RANK={RANK},SIZE={SIZE}')
def fxn(idxs,xs):
    n,N = len(xs),len(xs[1])
    prod = 0.
    for k in range(1,n+1):
        prod += np.log10(sum([xs[k][i][idxs[i-1]] for i in range(1,N+1)]))
    return prod
if RANK==0:
    N = 6
    tag,new_tag = 'x','a'
    ng = 4
    n = 10
    cutoff = 1e-15
    max_bond = 500 
    xs = dict()
    qs = np.zeros((N,n,ng))
    row1 = {i:np.random.rand(ng) for i in range(1,N+1)}
    for k in range(1,n+1):
        #xs[k] = {i:np.random.rand(ng) for i in range(1,N+1)}
        xs[k] = row1 
        for i in range(1,N+1):
            qs[i-1,k-1,:] = xs[k][i]
    #print(f'numerical integration...')
    #iterate_over = list(itertools.product(range(ng),repeat=N))
    #args = [xs]
    #kwargs = dict()
    #ls = parallelized_looped_function(fxn,iterate_over,args,kwargs) 
    #out = ls[0]
    #for prod in ls[1:]:
    #    out += np.log10(1.+10.**(prod-out))
    #out -= N*np.log10(ng)

    tn = get_peps(qs,np.ones(ng)/ng)
    tn1 = insert_projectors(tn.copy())
    tn1.full_simplify_(seq='CR')
    print(tn1.outer_inds())
    opt = ctg.ReusableHyperOptimizer(
        minimize='flops',
        reconf_opts={},
        slicing_reconf_opts={'target_size':2**26},
        parallel='ray',
        progbar=True,
        directory=f'./hyper_path/')
    tree = tn1.contraction_tree(opt)
    out,exp1 = tree.contract(tn1.arrays,progbar=True,strip_exponent=True)
    exp1 += tn1.exponent + np.log10(out)
    print('exp1=',exp1)
 
    sign,data_row = compress_row(tn,max_bond=max_bond,cutoff=cutoff) 
    print('check trace prod=',data_row)
    sign,data_col = compress_col(tn,max_bond=max_bond,cutoff=cutoff) 
    print('check trace prod=',data_col)
    for complete_rank in range(1,SIZE):
        COMM.send('finished',dest=complete_rank)
else:
    worker_execution()
