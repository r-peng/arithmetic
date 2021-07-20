import numpy as np
import quimb.tensor as qtn
import quimb.tensor.tensor_core as qtc
import hyper as hp
import cotengra as ctg

def _chebyshev(x,typ,p):
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

show_tags=False
show_inds=True
iterations=0
initial_layout='kamada_kawai'
#initial_layout='circular'
span_opts = {}

layout='ring'
#layout='tent'
plot_leaf_labels=False
#plot_leaf_labels=True

xs = [np.random.rand(d) for i in range(n)]
tn = hp._train(xs,'+')
ts = hp._chebyshev(tn,'t',p)
for j in range(len(ts)):
    print('order=',j)
    tn_ = ts[j].copy()
    print(tn_.outer_inds())
    tn_ = tn_.full_simplify(seq='ADCRS',output_inds=tn_._outer_inds)
#    print(tn_)
    print(tn_.outer_inds())
#    print(tn_._inner_inds)
    for i in range(n):
        tn_._outer_inds.add('x{},'.format(i))
        tn_._inner_inds.discard('x{},'.format(i))
    print(tn_.outer_inds())
#    print(tn_._inner_inds)
#    fig = tn_.draw(return_fig=True,
#                   show_tags=False,
#                   show_inds=show_inds,
#                   iterations=iterations,
#                   initial_layout=initial_layout)
#    fig.savefig(initial_layout+'{}.pdf'.format(j))
#    print(tn_)
#    tids = tn_._get_tids_from_inds(tn_.outer_inds(),which='any')
#    fig = tn_._draw_tree_span_tids(tids=tids,
#                                   return_fig=True,
#                                   show_tags=False,
#                                   show_inds=show_inds,
#                                   iterations=iterations,
#                                   initial_layout=initial_layout,
#                             )
#    fig.savefig('tree{}.pdf'.format(j))
    opt = ctg.HyperOptimizer()
    info = tn_.contract(output_inds=tn_._outer_inds,get='path-info')
    print(info)
    tree = ctg.ContractionTree.from_info(info)
    fig = ctg.plot_tree(tree,return_fig=True,
                        layout=layout,
                        plot_leaf_labels=plot_leaf_labels)
    fig.savefig(layout+'{}.pdf'.format(j))
exit()
us = hp._chebyshev(tn,'u',p)

tn_ = tn.copy()
for i in range(n):
    data = np.zeros(d)
    data[ins[i]] = 1.0
    tn_.add_tensor(qtn.Tensor(data=data,inds=['x{},'.format(i)],tags='i'))
tn_ = tn_.full_simplify(seq='ADCRS',output_inds=tn_._outer_inds)
opt = ctg.HyperOptimizer()
out = tn_.contract(output_inds=tn_._outer_inds,optimize=opt)
true = sum([xs[i][ins[i]] for i in range(n)])
print('sum err', true-out.data[1])

true_ts = _chebyshev(true,'t',p)
for j in range(len(ts)):
    tn_ = ts[j].copy()
    for i in range(n):
        data = np.zeros(d)
        data[ins[i]] = 1.0
        tn_.add_tensor(qtn.Tensor(data=data,inds=['x{},'.format(i)],tags='i'))
    tn_ = tn_.full_simplify(seq='ADCRS',output_inds=tn_._outer_inds)
    opt = ctg.HyperOptimizer()
    out = tn_.contract(output_inds=tn_._outer_inds,optimize=opt)
    true = sum([xs[i][ins[i]] for i in range(n)])
    print('order={}, err={}'.format(j,true_ts[j]-out.data[1]))
exit()
true_us = _chebyshev(true,'u',p)
