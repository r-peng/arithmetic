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
np.set_printoptions(precision=8,suppress=True)
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
print(f'RANK={RANK},SIZE={SIZE}')
if RANK==0:
    N = 10
    tag,new_tag = 'x','a'
    
    D = np.random.rand(N)
    K = np.random.rand(N,N)
    K -= K.T
    U = scipy.linalg.expm(K)
    A = np.linalg.multi_dot([U,np.diag(D),U.T])
    
    ng = 4
    xs,ws = np.polynomial.legendre.leggauss(ng)
    
    idxs = [np.random.randint(low=0,high=ng) for i in range(N)]
    vec = np.array([xs[idxs[i]] for i in range(N)])
    A_abs = np.einsum('ij,i,j->',A,vec,vec)
    exp = np.exp(-A_abs)
    
    tr = {i:np.zeros(ng) for i in range(1,N+1)}
    for i in range(1,N+1):
        tr[i][idxs[i-1]] = 1. 
    
    n = 20
    expi = 0.
    data = [np.log10(A_abs)]
    sign = [-1]
    for i in range(2,n+1):
        expi -= np.log10(i)
        data.append(expi+i*data[0])
        sign.append((-1)**i)
        print(f'i={i},data={data[-1]}')
    log_pol = parse(data,sign)*np.log(10.)
    print(f'A={A_abs},log_pol={log_pol}')
    #exit()
    
    cutoff = 1e-15
    max_bond = 500 
    tny = get_field(xs,tag,N,iprint=2,cutoff=cutoff)
    tnA = get_quadratic(tny,A,tag,iprint=2,cutoff=cutoff)
    if SIZE==1:
        print('check row...')
        data,_ = trace_pol_compress_row(tnA,tag,tr,n,'./tmpdir/',iprint=2,
                                        cutoff=cutoff,max_bond=max_bond)
    else:
        print('check col...')
        data = trace_pol_compress_col(tnA,tag,new_tag,tr,n,iprint=1,
                                      cutoff=cutoff,max_bond=max_bond)
    sign = [(-1)**i for i in range(1,n+1)]
    out = parse(data,sign)*np.log(10.)
    
    for i in range(1,N+1):
        tny.add_tensor(qtn.Tensor(data=tr[i],inds=(f'{tag}{i}',)))
    data = tny.contract(output_inds=('p','k')).data
    print('check get_field[0]=',np.linalg.norm(np.ones(N)-data[:,0])/np.sqrt(N))
    print('check get_field[1]=',np.linalg.norm(vec-data[:,1])/np.linalg.norm(vec))
    
    for i in range(1,N+1):
        tnA.add_tensor(qtn.Tensor(data=tr[i],inds=(f'{tag}{i}',)))
    data = tnA.contract().data
    print('check get_exponent[0]=',abs(1.-data[0]))
    print('check get_exponent[1]=',abs(A_abs-data[1])/A_abs)
    
    print('check trace pol=',abs(log_pol-out)/abs(log_pol))
    print('check trace exp=',abs(-A_abs-out)/A_abs)
    for complete_rank in range(1,SIZE):
        COMM.send('finished',dest=complete_rank)
else:
    worker_execution()
