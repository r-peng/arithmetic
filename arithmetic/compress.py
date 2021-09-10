import numpy as np
import math
import quimb.tensor as qtn
import quimb.tensor.tensor_core as qtc
import quimb.tensor.tensor_2d as qt2d
def get_peps(tn,N,n):
    arrays = []
    for k in range(n):
        row = []
        for i in range(N):
            array = tn.tensor_map[i*n+k].data.copy()
            array = array.reshape(array.shape+(1,))
            row.append(array)
        arrays.append(row)
    return qtn.PEPS(arrays,shape='dulrp')
def contract_boundary(peps,from_which=None,**kwargs):
    for i in range(peps.Lx):
        for j in range(peps.Ly):
            tid = peps._get_tids_from_tags(tags=peps.site_tag(i,j),which='any')
            tid = tuple(tid)[0]
            t = peps._pop_tensor(tid)
            tmp = qtn.Tensor(data=np.array([1.0]),inds=[peps.site_ind(i,j)])
            t = qtc.tensor_contract(t,tmp,preserve_tensor=True)
            peps.add_tensor(t,tid=tid)
#    print(peps)
    if from_which is None:
        return peps.contract_boundary(**kwargs)
    else:
        peps = peps.contract_boundary_from(xrange=None,yrange=None,
               from_which=from_which,**kwargs)
        return peps.contract()
def greedy_compress_optimizer(max_bond,**kwargs):
    import cotengra as ctg
    name = 'tmp'
    ssa_func = ctg.path_greedy.trial_greedy_compressed
    space = {
        'coeff_size_compressed': {'type': 'FLOAT', 'min': 0.5, 'max': 2.0},
        'coeff_size': {'type': 'FLOAT', 'min': 0.0, 'max': 1.0},
        'coeff_size_inputs': {'type': 'FLOAT', 'min': -1.0, 'max': 1.0},
        'score_size_inputs': {
            'type': 'STRING',
            'options': ['min', 'max', 'mean', 'sum', 'diff']},
        'coeff_subgraph': {'type': 'FLOAT', 'min': -1.0, 'max': 1.0},
        'score_subgraph': {
            'type': 'STRING',
            'options': ['min', 'max', 'mean', 'sum', 'diff']},
        'coeff_centrality': {'type': 'FLOAT', 'min': -10.0, 'max': 10.0},
        'centrality_combine': {
            'type': 'STRING',
            'options': ['min', 'max', 'mean']},
        'score_centrality': {
            'type': 'STRING',
            'options': ['min', 'max', 'mean', 'diff']},
        'temperature': {'type': 'FLOAT', 'min': -0.1, 'max': 1.0},
    }
    constants = {'chi': max_bond}
    ctg.hyper.register_hyper_function(name=name,ssa_func=ssa_func,space=space,constants=constants)
    return ctg.HyperOptimizer(methods=[name],**kwargs)
def from_top(tn,Lx,Ly,linds=[],rinds=[],cutoff=0.0,max_bond=None):
    # tid[i,k] = i*n+k
    N,n = Lx,Ly
    def contr(i,k):
        tid1,tid2 = i*n,i*n+k
        t1 = tn._pop_tensor(tid1)
        t2 = tn._pop_tensor(tid2)
        t12 = qtc.tensor_contract(t1,t2,preserve_tensor=True)
        tn.add_tensor(t12,tid=tid1)
        return 
    for k in range(1,n):
        contr(0,k)
        for i in range(1,N):
            contr(i,k)
            tn._compress_between_tids((i-1)*n,i*n,max_bond=max_bond,cutoff=cutoff)
    output_inds = linds+rinds
    out = qtc.tensor_contract(*tn.tensors,output_inds=output_inds,
                              preserve_tensor=True)
    return out
def svd(tn,Lx,Ly,from_which='left',**kwargs):
    N,n = Lx,Ly
    def _svd(i,k):
        tid1,tid2 = i*n,i*n+k
        t1 = tn._pop_tensor(tid1)
        t2 = tn._pop_tensor(tid2)
        t12 = qtc.tensor_contract(t1,t2,preserve_tensor=True)
        i1 = 'i{},0,'.format(i+1) if k==1 else 'k{},{},'.format(i+1,k-1)
        right_inds = i1,'i{},{},'.format(i+1,k)
        bond_ind = 'k{},{},'.format(i+1,k)
        tl,tr = qtc.tensor_split(t12,left_inds=None,get='tensors',
                                 right_inds=right_inds,bond_ind=bond_ind,**kwargs)
        tn.add_tensor(tl,tid=tid1)
        tid = (i+1)*n+k
        t = tn._pop_tensor(tid)
        t = qtc.tensor_contract(tr,t,preserve_tensor=True)
        tn.add_tensor(t,tid=tid)
    if from_which=='left':
        for i in range(N-1): 
            for k in range(1,n):
                _svd(i,k)
    elif from_which=='top':
        for k in range(1,n):
            for i in range(N-1): 
                _svd(i,k)
    tid1 = (N-1)*n
    t12 = tn._pop_tensor(tid1)
    for k in range(1,n):
        tid2 = tid1+k
        t2 = tn._pop_tensor(tid2)
        t12 = qtc.tensor_contract(t12,t2,preserve_tensor=True)
    tn.add_tensor(t12,tid=tid1)
    return tn
