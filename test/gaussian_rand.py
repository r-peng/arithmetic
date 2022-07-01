import numpy as np
import cotengra as ctg
from arithmetic.gaussian import (
    get_A,
    get_hypergraph,
    resolve,
    contract,
)
import itertools,h5py
from arithmetic.utils import parallelized_looped_function,worker_execution
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
    N = 16
    hw = N
    cutoff = 1e-5
    ng = 4
    xs,ws = np.polynomial.legendre.leggauss(ng)
    
    #A,D = get_A(N,hw)
    #print(A)
    A = np.random.rand(N,N)
    A += A.T
    A -= 1.
    #f = h5py.File('A.hdf5','w')
    #f.create_dataset('A',data=A)
    #f.close()
    #exit()

    #f = h5py.File('A.hdf5','r')
    #A = f['A'][:]
    #f.close()

    tn1 = get_hypergraph(A,xs,ws,simplify=True,cutoff=cutoff)
    tn2 = resolve(tn1.copy(),N,remove_lower=True)

    tn1.full_simplify_(seq='CR')
    opt = ctg.ReusableHyperOptimizer(
        minimize='flops',
        reconf_opts={},
        slicing_reconf_opts={'target_size':2**26},
        max_repeats=64,
        parallel='ray',
        progbar=True,
        directory=f'./hyper_path/')
    #tree = tn.contraction_tree(opt,output_inds='')
    tree = tn1.contraction_tree(opt)
    out,exp1 = tree.contract(tn1.arrays,progbar=True,strip_exponent=True)
    exp1 += tn1.exponent + np.log10(out)
    exp1 *= np.log(10.)
    print(f'ng={ng},exp1={exp1}')
    #exit()

    print('num_tensors=',tn2.num_tensors)
    exp2 = contract(tn2,final=3,total=tn2.num_tensors*2,max_bond=128) 
    exp2 *= np.log(10.)
    print('exp1=',exp1)
    print('exp2=',exp2)
    exit()
    
    print('numerical integration...')
    iterate_over = list(itertools.product(range(ng),repeat=N))
    args = [A,xs,ws]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    e0 = ls[0]
    for e1 in ls[1:]:
        e0 += np.log(1.+np.exp(e1-e0)) 
    print('check trace exp=',abs(e0-exp)/abs(e0))
    for complete_rank in range(1,SIZE):
        COMM.send('finished',dest=complete_rank)
else:
    worker_execution()
