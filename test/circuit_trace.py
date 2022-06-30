import numpy as np
import quimb.tensor as qtn
from arithmetic.circuit import build,contract_norm,get_phase_map,evaluate
from arithmetic.utils import parallelized_looped_function,worker_execution
import itertools,time
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
print(f'RANK={RANK},SIZE={SIZE}')
def fxn(idxs,xs,ws,phase_map):
    term = evaluate([xs[idx] for idx in idxs],phase_map,probability=True)
    return np.log10(term) + sum([np.log10(w) for w in [ws[idx] for idx in idxs]])
if RANK==0:
    nl = 3
    theta_map = {i:np.random.rand() for i in range(1,nl+1)}
    N = (2**nl-1)*2
    for ng in [4,8,12,16,20,50]:
        xs,ws = np.polynomial.legendre.leggauss(ng)
        xs *= np.sqrt(3) 
        print(f'ng={ng},sum(w)={sum(ws)},int={sum(np.multiply([x**2 for x in xs],ws))}')
    
    ng = 2
    xs,ws = np.polynomial.legendre.leggauss(ng)
    xs *= np.sqrt(3) 
    tn = build({i:xs for i in range(1,N+1)},theta_map)
    exp1 = contract_norm({i:ws for i in range(1,N+1)},tn)
   
    t0 = time.time() 
    phase_map = get_phase_map(nl*2,theta_map)
    print('time for getting phase_map',time.time()-t0)

    iterate_over = list(itertools.product(range(ng),repeat=N))
    args = [xs,ws,phase_map]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    out = ls[0]
    for term in ls[1:]:
        out += np.log10(1.+10.**(term-out))
    print('exp1=',exp1)
    print('exp2=',out)
    for complete_rank in range(1,SIZE):
        COMM.send('finished',dest=complete_rank)
else:
    worker_execution()
