import numpy as np
import cotengra as ctg
from arithmetic.pair_potential import morse,get_hypergraph2D,resolve2D,contract
from arithmetic.utils import parallelized_looped_function,worker_execution
import itertools
np.set_printoptions(precision=8,suppress=True)
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
print(f'RANK={RANK},SIZE={SIZE}')
def fxn(idxs,x,y,wx,wy,beta,**v_params):
    xix,yix = idxs
    N = len(xix)
    out = 0.
    for i in range(N):
        xi,yi = xix[i],yix[i]
        ri = np.array([x[xi],y[yi]])
        for j in range(i+1,N):
            xj,yj = xix[j],yix[j]
            rj = np.array([x[xj],y[yj]])
            out += -beta*morse(ri,rj,**v_params)
    wtsx = np.array([wx[xix[i]] for i in range(N)])
    wtsy = np.array([wy[yix[i]] for i in range(N)])
    return out + sum([np.log(w) for w in wtsx]) + sum([np.log(w) for w in wtsy])
if RANK==0:
    N = 8
    ng = 4
    beta = 1.
    v_params = {'De':1.,'a':1.,'re':1.}
    xs,ws = np.polynomial.legendre.leggauss(ng)
    tn = get_hypergraph2D(xs,xs,ws,ws,N,beta=beta,v_params=v_params)
    peps = resolve2D(tn.copy(),N)
    #print(tn.outer_inds())
    #exit()
    print(f'N={N},ng={ng},num_tensors={tn.num_tensors}')
    for run in range(1):
        opt = ctg.ReusableHyperOptimizer(
            minimize='flops',
            reconf_opts={},
            slicing_reconf_opts={'target_size':2**26},
            parallel='ray',
            progbar=True,
            directory=f'./hyper_path/N{N}_run{run}/')
        #tree = tn.contraction_tree(opt,output_inds='')
        tree = tn.contraction_tree(opt)
        flop = np.log10(float(tree.total_flops()))
        peak_size = np.log2(float(tree.peak_size()))
        max_size = np.log2(float(tree.max_size()))
        print(f'run={run},flop={flop},max_size={max_size},peak_size={peak_size}')
    #exit()
    out,exp1 = tree.contract(tn.arrays,progbar=True,strip_exponent=True)
    exp1 += tn.exponent + np.log10(out)
    exp1 *= np.log(10.)
    print('exp1=',exp1)
    exp2 = contract(peps,['x','y'],max_bond=100,progbar=True)    
    exp2 *= np.log(10.)
    print('exp2=',exp2)
    exit()
    
    print('numerical integration...')
    ls_x = list(itertools.product(range(ng),repeat=N))
    ls_y = list(itertools.product(range(ng),repeat=N))
    iterate_over = [(x,y) for x in ls_x for y in ls_y]

    args = [xs,xs,ws,ws,beta]
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
