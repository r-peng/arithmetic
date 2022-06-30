import numpy as np
import quimb.tensor as qtn
from .utils import scale,write_tn_to_disc,load_tn_from_disc
from .gaussian import parse,trace_field,compress_row_1step
import time
def disentangle(ref_tn,tag,utag,max_bond,steps=100,tol=1e-5,
                cutoff=1e-10,rand_scale=0.1):
    tn = ref_tn.copy()
    nf = tn.num_tensors 
    # compress to max_bond
    for j in range(1,nf):
        #tn.compress_between(f'{tag}{j}',f'{tag}{j+1}',absorb='right',
        #                    cutoff=cutoff)
        tn.canonize_between(f'{tag}{j}',f'{tag}{j+1}',absorb='right')
    for j in range(nf-1,0,-1):
        tn.compress_between(f'{tag}{j}',f'{tag}{j+1}',absorb='left',
                            cutoff=cutoff,max_bond=max_bond)
    for j in range(1,nf+1):
        tn[f'{tag}{j}'].reindex_({f'{tag}{j}':f'{tag}{j}_'})
    # add unitaries
    ng = tn[f'{tag}1'].shape[tn[f'{tag}1'].inds.index(f'{tag}1_')]
    idxs = {j:qtn.rand_uuid() for j in range(1,nf)}
    for j in range(1,nf):
        data = np.random.rand(*(ng,)*4)*rand_scale
        idx1 = f'{tag}{j}_' if j==1 else idxs[j-1]
        idx4 = f'{tag}{j+1}' if j==nf-1 else idxs[j]
        inds = idx1,f'{tag}{j+1}_',f'{tag}{j}',idx4
        U = qtn.Tensor(data=data,inds=inds,tags=f'{utag}{j},{j+1}',
                       left_inds=inds[:2])
        U.unitize_(left_inds=U.left_inds,method='exp')
        tn.add_tensor(U)
    
    def norm_fn(tn):
        for j in range(1,nf):
            U = tn[f'{utag}{j},{j+1}']
            U.unitize_(left_inds=U.left_inds,method='exp')
        return tn
    BB = ref_tn.copy() 
    BB.add_tensor_network(BB.copy(),check_collisions=True)
    for j in range(1,nf-1):
        BB.contract_tags((f'{tag}{j}',f'{tag}{j+1}'),which='any',inplace=True)
    BB = BB.contract()
    print('BB=',BB)
    def loss_fn(tn):
        AA = qtn.TensorNetwork([])
        for j in range(1,nf+1):
            AA.add_tensor(tn[f'{tag}{j}'])
        AA.add_tensor_network(AA.copy(),check_collisions=True)
        for j in range(1,nf-1):
            AA.contract_tags((f'{tag}{j}',f'{tag}{j+1}'),which='any',inplace=True)
        AA = AA.contract()

        AB = tn.copy()
        AB.add_tensor_network(ref_tn.copy(),check_collisions=True)
        for j in range(1,nf-1):
            AB.contract_tags((f'{tag}{j}',f'{tag}{j+1}',f'{utag}{j},{j+1}'),
                             which='any',inplace=True)
        AB = AB.contract()
        return (AA+BB-2*AB)/BB
    print('initial loss=',loss_fn(tn))
    #exit()

    opt = qtn.TNOptimizer(tn,loss_fn,norm_fn=norm_fn)
    tn = opt.optimize(steps,tol=tol)
    return tn
