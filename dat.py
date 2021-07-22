import numpy as np
import quimb.tensor as qtn
import quimb.tensor.tensor_core as qtc
import hyper as hp
import cotengra as ctg
def chebyshev(x,typ,p):
    p0 = 1.0
    p1 = x
    if typ=='u' or 'U':
        p1 *= 2.0
    ls = [p0,p1]
    for i in range(2,p+1):
        ls.append(2.0*x*ls[-1]-ls[-2])
    return ls

dt = 0.1
n = 5
p = 4
xs = [np.arange(-1.0,1.0,dt) for i in range(n)]
tn = hp.train(xs)
ts = hp.chebyshev(tn,'t',p)
tn = ts[-1].copy()
print('before simplify',tn.num_tensors)
tn = hp.simplify(tn)
print('after simplify',tn.num_tensors)
ins = [np.random.randint(0,len(xs[i])) for i in range(n)]
out = hp.contract(tn.copy(),ins)
true = sum([xs[i][ins[i]] for i in range(n)])
true_ts = chebyshev(true,'t',p)
print('check output', true_ts[-1]-out)

output_inds = ['x{},'.format(i) for i in range(n)]
initial_layout = 'spectral'
initial_layout = 'kamada_kawai' 
iterations = 0
fig = tn.draw(output_inds=output_inds,initial_layout=initial_layout,
              iterations=iterations,
              show_inds=True,show_tags=False,return_fig=True)
fig.savefig(initial_layout+'p{},n{}.pdf'.format(p,n))
exit()
layout='ring'

methods = ['greedy','kahypar']
#methods = ['kahypar-tmp']
reconf_opts = {'inplace':True}




print(tn.outer_inds())
#for key in tn.tensor_map.keys():
#    print(key,tn.tensor_map[key])
opt = ctg.HyperOptimizer(methods=methods,reconf_opts=reconf_opts)
info = tn.contract(output_inds=tn._outer_inds,optimize=opt,get='path-info')
labels = info.input_subscripts.split(',') 
tree = ctg.ContractionTree.from_info(info)
print(info)
#for key in tree.children.keys():
#    print(key,tree.children[key])
#    hp.contract_from_tree(tn,tree)
#    tree2 = tree.subtree_reconfigure_forest()
#    print(tree2.children)

output = info.output_subscript
colors = [np.random.rand(3) for i in range(n)]
highlight = dict(zip(output,colors)) 
fig = ctg.plot_tree(tree,return_fig=True,
                    layout=layout,
                    highlight=highlight,
                    plot_leaf_labels=labels)
fig.savefig(layout+'p{},n{}.pdf'.format(p,n))
exit()


true_us = chebyshev(true,'u',p)
for j in range(p+1):
    out_t = hp.contract(ts[j],ins) 
    out_u = hp.contract(us[j],ins) 
    print('order={}, err={},{}'.format(j,true_ts[j]-out_t,true_us[j]-out_u))
