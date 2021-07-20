import numpy as np
import quimb.tensor as qtn
import quimb.tensor.tensor_core as qtc
# basic tensors
def _tensor(f):
    out = np.zeros(f.shape+(2,))
    out[...,1] = f.copy()
    out[...,0] = np.ones_like(f)
    return out
ADD = np.zeros((2,2,2)) # i1,i2,o
ADD[0,0,0] = ADD[0,1,1] = ADD[1,0,1] = 1.0

def _compose(tn1,tn2,op):
    tn = tn1.copy()
    tn.add_tensor_network(tn2)

    x1 = list(tn1._outer_inds._d.keys())
    o1 = x1.pop()
    x2 = list(tn2._outer_inds._d.keys())
    o2 = x2.pop()

    for x in x1:
        if x in x2:
            tn._outer_inds.add(x)
            tn._inner_inds.discard(x)

    o = qtc.rand_uuid()
    if op=='*':
        tn.reindex(index_map={o1:o},inplace=True)
        tn.reindex(index_map={o2:o},inplace=True)
        tn._outer_inds.discard(o1)
        tn._outer_inds.discard(o2)
        tn._outer_inds.add(o)
    elif op=='+':
        inds = o1,o2,o 
        tn.add_tensor(qtn.Tensor(data=ADD,inds=inds,tags='+'))
    return tn
def _train(xs,op='+'):
    inds = 'x0,',qtc.rand_uuid()
    xi = qtn.Tensor(data=_tensor(xs[0]),inds=inds,tags={'x'})
    tn = qtn.TensorNetwork([xi])
    for i in range(1,len(xs)):
        inds = 'x{},'.format(i),qtc.rand_uuid()
        xi = qtn.Tensor(data=_tensor(xs[i]),inds=inds,tags={'x'})
        tni = qtn.TensorNetwork([xi])
        tn = _compose(tn,tni,op)
    return tn
def _const_fxn(ds,a):
    xs = [np.ones(di) for di in ds]
    xs[0] *= a
    return _train(xs,op='*')
def _scalar_mult(tn,a):
    data = np.array([1.0,a])
    inds = [qtc.rand_uuid()]
    a = qtn.TensorNetwork([qtn.Tensor(data=data,inds=inds,tags='c')])
    return _compose(tn,a,'*')
def _chebyshev(tn,typ,p):
    # tn: quimb list
    # typ: 't' or 'u'
    # p: highest degree
    n = len(tn._outer_inds)-1

    _,output,size_dict = tn.get_inputs_output_size_dict()
    ds = [size_dict[key] for key in output[:-1]]
    p0 = _const_fxn(ds,1.0)

    p1 = tn.copy()
    if typ=='u'or'U':
        p1 = _scalar_mult(p1,2.0)

    ls = [p0,p1]
    for i in range(2,p+1):
        tmp1 = _compose(tn,ls[-1],'*')
        tmp1 = _scalar_mult(tmp1,2.0)
        tmp2 = _scalar_mult(ls[-2],-1.0)
        ls.append(_compose(tmp1,tmp2,'+'))
    return ls 
def _simplify(tn,seq='ADCRS',atol=1e-15,equalize_norm=True):
    tn = tn.full_simplify(seq=seq,output_inds=tn._outer_inds,atol=atol,
                          equalize_norms=equalize_norm)
    for i in range(n):
        tn._outer_inds.add('x{},'.format(i))
        tn._inner_inds.discard('x{},'.format(i))
    o = tn
def _out(tn,ins,seq='ADCRS',atol=1e-15,equalize_norm=True): 
    _,output,size_dict = tn.get_inputs_output_size_dict()
    ds = [size_dict[key] for key in output[:-1]]
    for i in range(n):
        data = np.zeros(ds[i])
        data[ins[i]] = 1.0
        tn.add_tensor(qtn.Tensor(data=data,inds=['x{},'.format(i)],tags='i'))
    tn = tn.full_simplify(seq=seq,output_inds=tn._outer_inds,atol=atol,
                          equalize_norms=equalize_norm)
    opt = ctg.HyperOptimizer()
    out = tn.contract(output_inds=tn._outer_inds,optimize=opt)
    return out.data[1]
if __name__=='__main__':
    import cotengra as ctg

    d = 3
    n = 3

    xs = [np.random.rand(d) for i in range(n)]
    tn = _train(xs,'+')
#    print(tn)
#    print('outer',tn._outer_inds)
#    print('inner',tn._inner_inds)
    opt = ctg.HyperOptimizer()
    out = tn.contract(output_inds=tn._outer_inds,optimize=opt)
    ins = [np.random.randint(0,d) for i in range(n)]
    true = sum([xs[i][ins[i]] for i in range(n)])
    print('check train: ', true-out.data[...,1][tuple(ins)])
    print('check train: ', 1.0-out.data[...,0][tuple(ins)])

    ds = [d for i in range(n)]
    a = np.random.rand()
    tn = _const_fxn(ds,a)
#    print(tn)
#    print('outer',tn._outer_inds)
#    print('inner',tn._inner_inds)
    opt = ctg.HyperOptimizer()
    out = tn.contract(output_inds=tn._outer_inds,optimize=opt)
    ins = [np.random.randint(0,d) for i in range(n)]
    true = a 
    print('check const: ', true-out.data[...,1][tuple(ins)])
    print('check const: ', 1.0-out.data[...,0][tuple(ins)])

    xs = [np.random.rand(d) for i in range(n)]
    a = np.random.rand()
    tn = _train(xs,'+')
    tn = _scalar_mult(tn,a)
#    print(tn)
#    print('outer',tn._outer_inds)
#    print('inner',tn._inner_inds)
    opt = ctg.HyperOptimizer()
    out = tn.contract(output_inds=tn._outer_inds,optimize=opt)
    ins = [np.random.randint(0,d) for i in range(n)]
    true = a*sum([xs[i][ins[i]] for i in range(n)])
    print('check scalar_mult: ', true-out.data[...,1][tuple(ins)])
    print('check scalar_mult: ', 1.0-out.data[...,0][tuple(ins)])

    x1s = [np.random.rand(d) for i in range(n)]
    tn1 = _train(x1s)
#    print(tn1)
#    print('outer1',tn1._outer_inds)
#    print('inner1',tn1._inner_inds)
    x2s = [np.random.rand(d) for i in range(n)]
    tn2 = _train(x2s)
#    print(tn2)
#    print('outer2',tn2._outer_inds)
#    print('inner2',tn2._inner_inds)
    tn = _compose(tn1,tn2,'*')
#    print(tn)
#    print('outer',tn._outer_inds)
#    print('inner',tn._inner_inds)
    opt = ctg.HyperOptimizer()
    out = tn.contract(output_inds=tn._outer_inds,optimize=opt)
    ins = [np.random.randint(0,d) for i in range(n)]
    true  = sum([x1s[i][ins[i]] for i in range(n)])
    true *= sum([x2s[i][ins[i]] for i in range(n)])
    print('check mult: ', true-out.data[...,1][tuple(ins)])
    print('check mult: ', 1.0-out.data[...,0][tuple(ins)])


