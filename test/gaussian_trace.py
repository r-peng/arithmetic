import numpy as np
import scipy.linalg
import quimb.tensor as qtn
from arithmetic.gaussian import (
    get_field,
    get_quadratic,
    trace_pol_compress_row,
    trace_pol_compress_col,
    parse,
)
from arithmetic.utils import worker_execution
import itertools
np.set_printoptions(precision=8,suppress=True)
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
print(f'RANK={RANK},SIZE={SIZE}')
if RANK==0:
    N = 8
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
    exp_cut = np.log10(cutoff)
    tny = get_field(xs,tag,N,iprint=2,cutoff=cutoff)
    tnA = get_quadratic(tny,A,tag,iprint=2,cutoff=cutoff)
    if SIZE==1:
        print('check row...')
        data,_ = trace_pol_compress_row(tnA,tag,tr,n,iprint=2,
                                      exp_cut=exp_cut,cutoff=cutoff,max_bond=max_bond)
    else:
        print('check col...')
        data = trace_pol_compress_col(tnA,tag,new_tag,tr,n,iprint=1,
                                      cutoff=cutoff,max_bond=max_bond)
    sign = [(-1)**i for i in range(1,n+1)]
    out = parse(data,sign,log10_a0=N*np.log10(sum(ws)))*np.log(10.)
    print('out=',out)
    print('numerical integration...')
    ls = list(itertools.product(range(ng),repeat=N))
    idxs = ls[0]
    vec = np.array([xs[idxs[i]] for i in range(N)])
    wts = np.array([ws[idxs[i]] for i in range(N)])
    e0 = - np.einsum('ij,i,j->',A,vec,vec) + sum([np.log(w) for w in wts])
    for idxs in ls[1:]:
        vec = np.array([xs[idxs[i]] for i in range(N)])
        wts = np.array([ws[idxs[i]] for i in range(N)])
        e1 = - np.einsum('ij,i,j->',A,vec,vec) + sum([np.log(w) for w in wts])
        e0 += np.log(1.+np.exp(e1-e0)) 
    print('check trace exp=',abs(e0-out)/abs(e0))
    for complete_rank in range(1,SIZE):
        COMM.send('finished',dest=complete_rank)
else:
    worker_execution()
