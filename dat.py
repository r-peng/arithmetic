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

d = 3
n = 3
p = 3 
ins = [np.random.randint(0,d) for i in range(n)]

layout='ring'

xs = [np.random.rand(d) for i in range(n)]
f = hp.train(xs)
ft = hp.chebyshev(f,'t',p)
#exit()
fu = hp.chebyshev(f,'u',p)

#for j in range(len(ts)):
#    print('######################### order={} ##########################'.format(j))
#    tn_ = ts[j].copy()
##    print(tn_.outer_inds())
#    tn_ = tn_.full_simplify(seq='ADCRS',output_inds=tn_._outer_inds)
##    print(tn_)
#    for i in range(n):
#        tn_._outer_inds.add('x{},'.format(i))
#        tn_._inner_inds.discard('x{},'.format(i))
#    print(tn_)
#    print(tn_.outer_inds())
#    opt = ctg.HyperOptimizer()
#    info = tn_.contract(output_inds=tn_._outer_inds,get='path-info')
#    print(info)
#    tree = ctg.ContractionTree.from_info(info)
#
#    output = info.output_subscript
#    output = output[:-1] if j<2 else output[1:]
#    colors = [np.random.rand(3) for i in range(n)]
#    highlight = dict(zip(output,colors)) 
#    fig = ctg.plot_tree(tree,return_fig=True,
#                        layout=layout,
#                        highlight=highlight,
#                        plot_leaf_labels=True)
#    fig.savefig(layout+'{}.pdf'.format(j))
#exit()

out = hp.contract(f,ins) 
true = sum([xs[i][ins[i]] for i in range(n)])
print('sum err', true-out[1])

true_ts = chebyshev(true,'t',p)
true_us = chebyshev(true,'u',p)
for j in range(2,p+1):
    out_t = hp.contract(ft[j],ins) 
    out_u = hp.contract(fu[j],ins) 
    print('order={}, err={},{}'.format(j,true_ts[j]-out_t[1],true_us[j]-out_u[1]))
