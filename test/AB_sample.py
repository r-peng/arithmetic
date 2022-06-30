import numpy as np
import scipy.linalg
import quimb.tensor as qtn
from arithmetic.gaussian import (
    get_field,
    get_quadratic,
    get_quartic,
    add_exponent,
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
    N = 9
    tag,new_tag = 'x','a'
    
    D = np.random.rand(N)
    K = np.random.rand(N,N)
    K -= K.T
    U = scipy.linalg.expm(K)
    A = np.linalg.multi_dot([U,np.diag(D),U.T])
    D = np.random.rand(N)
    K = np.random.rand(N,N)
    K -= K.T
    U = scipy.linalg.expm(K)
    B = np.einsum('s,si,sj,sk,sl->ijkl',D,U,U,U,U) 
    
    ng = 16
    xs,ws = np.polynomial.legendre.leggauss(ng)
    
    idxs = [np.random.randint(low=0,high=ng) for i in range(N)]
    vec = np.array([xs[idxs[i]] for i in range(N)])
    A_abs = np.einsum('ij,i,j->',A,vec,vec)
    B_abs = np.einsum('ijkl,i,j,k,l->',B,vec,vec,vec,vec)
    n = 30
    expi = 0.
    data = [np.log10(A_abs+B_abs)]
    sign = [-1]
    for i in range(2,n+1):
        expi -= np.log10(i)
        data.append(expi+i*data[0])
        sign.append((-1)**i)
        print(f'i={i},data={data[-1]}')
    log_pol = parse(data,sign)*np.log(10.)
    print(f'AB={A_abs+B_abs},log_pol={log_pol}')
    
    tr = {i:np.zeros(ng) for i in range(1,N+1)}
    for i in range(1,N+1):
        tr[i][idxs[i-1]] = 1. 
    
    cutoff = 1e-15
    max_bond = 500 
    tny = get_field(xs,tag,N,iprint=2,cutoff=cutoff)
    tnA = get_quadratic(tny,A,tag,iprint=2,cutoff=cutoff)
    tnB = get_quartic(tny,B,tag,iprint=2,cutoff=cutoff)
    tnAB = add_exponent(tnA,tnB,tag,iprint=2,cutoff=cutoff)   
 
    for i in range(1,N+1):
        tny.add_tensor(qtn.Tensor(data=tr[i],inds=(f'{tag}{i}',)))
    data = tny.contract(output_inds=('p','k')).data
    print('check get_field[0]=',np.linalg.norm(np.ones(N)-data[:,0])/np.sqrt(N))
    print('check get_field[1]=',np.linalg.norm(vec-data[:,1])/np.linalg.norm(vec))
    
    for i in range(1,N+1):
        tnA.add_tensor(qtn.Tensor(data=tr[i],inds=(f'{tag}{i}',)))
    data = tnA.contract().data
    print('check get_exponent[0]=',abs(1.-data[0]))
    print('check get_exponent[1]=',abs(A_abs-data[1])/abs(A_abs))
    
    for i in range(1,N+1):
        tnB.add_tensor(qtn.Tensor(data=tr[i],inds=(f'{tag}{i}',)))
    data = tnB.contract().data
    print('check get_exponent[0]=',abs(1.-data[0]))
    print('check get_exponent[1]=',abs(B_abs-data[1])/abs(B_abs))

    for i in range(1,N+1):
        tnAB.add_tensor(qtn.Tensor(data=tr[i],inds=(f'{tag}{i}',)))
    data = tnAB.contract().data
    AB = A_abs+B_abs
    print('check get_exponent[0]=',abs(1.-data[0]))
    print('check get_exponent[1]=',abs(AB-data[1])/abs(AB))
    for complete_rank in range(1,SIZE):
        COMM.send('finished',dest=complete_rank)
else:
    worker_execution()
