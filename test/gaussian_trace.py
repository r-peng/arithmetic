import numpy as np
import scipy.linalg
import quimb.tensor as qtn
from arithmetic.gaussian import (
    get_field,
    get_quadratic,
    trace_pol_compress_row,
    parse,
)
from arithmetic.utils import parallelized_looped_function,worker_execution
import itertools
np.set_printoptions(precision=8,suppress=True)
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
print(f'RANK={RANK},SIZE={SIZE}')
def fxn(idxs,A,xs,ws):
    N = len(idxs)
    vec = np.array([xs[idxs[i]] for i in range(N)])
    wts = np.array([ws[idxs[i]] for i in range(N)])
    return - np.einsum('ij,i,j->',A,vec,vec) + sum([np.log(w) for w in wts])
if RANK==0:
    N = 6
    tag,new_tag = 'x','a'
    
    D = np.random.rand(N)
    K = np.random.rand(N,N)
    K -= K.T
    U = scipy.linalg.expm(K)
    A = np.linalg.multi_dot([U,np.diag(D),U.T])
    
    ng = 4
    xs,ws = np.polynomial.legendre.leggauss(ng)
    tr = {i:ws for i in range(1,N+1)}
    
    n = 20
    cutoff = 1e-10
    max_bond = 500 
    tny = get_field(xs,tag,N,iprint=2,cutoff=cutoff)
    tnA = get_quadratic(tny,A,tag,iprint=2,cutoff=cutoff)
    data,_ = trace_pol_compress_row(tnA,tag,tr,n,'./tmpdir/',iprint=2,
                                        cutoff=cutoff,max_bond=max_bond)
    sign = [(-1)**i for i in range(1,n+1)]
    out = parse(data,sign,log10_a0=N*np.log10(sum(ws)))*np.log(10.)
    print('out=',out)
    print('numerical integration...')
    iterate_over = list(itertools.product(range(ng),repeat=N))
    args = [A,xs,ws]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    e0 = ls[0]
    for e1 in ls[1:]:
        e0 += np.log(1.+np.exp(e1-e0)) 
    print('check trace exp=',abs(e0-out)/abs(e0))
    for complete_rank in range(1,SIZE):
        COMM.send('finished',dest=complete_rank)
else:
    worker_execution()
