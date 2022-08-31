import numpy as np
import cotengra as ctg
from arithmetic.pair_potential import morse,get_hypergraph,resolve
from arithmetic.utils import parallelized_looped_function,worker_execution
import itertools
np.set_printoptions(precision=8,suppress=True)
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
print(f'RANK={RANK},SIZE={SIZE}')
def fxn(idxs,x,y,z,wx,wy,wz,beta,**v_params):
    xix,yix,zix = idxs
    N = len(xix)
    out = 0.
    for i in range(N):
        xi,yi,zi = xix[i],yix[i],zix[i]
        ri = np.array([x[xi],y[yi],z[zi]])
        for j in range(i+1,N):
            xj,yj,zj = xix[j],yix[j],zix[j]
            rj = np.array([x[xj],y[yj],z[zj]])
            out += -beta*morse(ri,rj,**v_params)
    fac = 0.
    for i in range(N):
        xi,yi,zi = xix[i],yix[i],zix[i]
        fac += sum(np.log([wx[xi],wy[yi],wz[zi]]))
    return out + fac
if RANK==0:
    N = 10
    ng = 4
    beta = 1.
    v_params = {'De':1.,'a':1.,'re':1.}
    xs,ws = np.polynomial.legendre.leggauss(ng)
    tn = get_hypergraph(xs,xs,xs,ws,ws,ws,N,beta=beta,v_params=v_params)
    #tn = resolve(tn,N)
    #print(tn.outer_inds())
    #exit()
    print(f'N={N},ng={ng},num_tensors={tn.num_tensors}')
    for run in range(20):
        opt = ctg.ReusableHyperOptimizer(
            minimize='flops',
            reconf_opts={},
            slicing_reconf_opts={'target_size':2**26},
            parallel='ray',
            progbar=True,
            directory=f'./hyper_path/N{N}_run{run}/')
        tree = tn.contraction_tree(opt,output_inds='')
        #tree = tn.contraction_tree(opt)
        flop = np.log10(float(tree.total_flops()))
        peak_size = np.log2(float(tree.peak_size()))
        max_size = np.log2(float(tree.max_size()))
        print(f'run={run},flop={flop},max_size={max_size},peak_size={peak_size}')
    exit()
    out,exp = tree.contract(tn.arrays,progbar=True,strip_exponent=True)
    exp += tn.exponent + np.log10(out)
    exp *= np.log(10.)
    print('exp=',exp)
    
    print('numerical integration...')
    ls_x = list(itertools.product(range(ng),repeat=N))
    ls_y = list(itertools.product(range(ng),repeat=N))
    ls_z = list(itertools.product(range(ng),repeat=N))
    iterate_over = [(x,y,z) for x in ls_x for y in ls_y for z in ls_z]

    args = [xs,xs,xs,ws,ws,ws,beta]
    kwargs = v_params
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    e0 = ls[0]
    for e1 in ls[1:]:
        e0 += np.log(1.+np.exp(e1-e0)) 
    print('check trace exp=',abs(e0-exp)/abs(e0))
    for complete_rank in range(1,SIZE):
        COMM.send('finished',dest=complete_rank)
else:
    worker_execution()
