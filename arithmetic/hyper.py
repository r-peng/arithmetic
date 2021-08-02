import numpy as np
import quimb.tensor as qtn
import quimb.tensor.tensor_core as qtc
# basic tensors
def tensor(f):
    out = np.zeros(f.shape+(2,))
    out[...,1] = f.copy()
    out[...,0] = np.ones_like(f)
    return out
ADD = np.zeros((2,2,2)) # i1,i2,o
ADD[0,0,0] = ADD[0,1,1] = ADD[1,0,1] = 1.0

def compose(tn1,tn2,op,contract=False):
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
        tn.add_tensor(qtn.Tensor(data=ADD,inds=inds))
        if contract:
            tn.contract_ind(o2)
    else:
        raise NotImplementedError

    return tn
def compose_scalar(tn,a,op,contract=False):
    inds = [qtc.rand_uuid()]
    a = qtn.TensorNetwork([qtn.Tensor(data=np.array([1.0,a]),inds=inds)])
    return compose(tn,a,op,contract)
def train(xs,op='+',coeff=None,contract=False):
    inds = 'x0,',qtc.rand_uuid()
    tn = qtn.TensorNetwork([qtn.Tensor(data=tensor(xs[0]),inds=inds)])
    if coeff is not None:
        tn = compose_scalar(tn,coeff[0],'*') 
    for i in range(1,len(xs)):
        inds = 'x{},'.format(i),qtc.rand_uuid()
        tni = qtn.TensorNetwork([qtn.Tensor(data=tensor(xs[i]),inds=inds)])
        if coeff is not None:
            tni = compose_scalar(tni,coeff[i],'*') 
        tn = compose(tn,tni,op,contract)
    return tn
def const_fxn(ds,a):
    xs = [np.ones(di) for di in ds]
    xs[0] *= a
    return train(xs,op='*')
def simplify(tn,**kwargs):
    tmp = tn._outer_inds.copy()
    o = tmp.popright()
    tn.add_tensor(qtn.Tensor(data=np.array([0.0,1.0]),inds=[o]))
    tn = tn.full_simplify(output_inds=tmp,**kwargs)
    tn.make_tids_consecutive()
    tn._outer_inds = tmp
    return tn
def contract(tn,ins=None,simplify=True,simplify_kwargs={},contract_kwargs={},optimize_kwargs={}): 
    import cotengra as ctg
    _,output,size_dict = tn.get_inputs_output_size_dict()
    ds = [size_dict[key] for key in output[:-1]]
    for i in range(len(ds)):
        if ins is None:
            data = np.ones(ds[i])
        else:
            data = np.zeros(ds[i])
            data[ins[i]] = 1.0
        tn.add_tensor(qtn.Tensor(data=data,inds=['x{},'.format(i)]))
    o = tn._outer_inds.popright() 
    tn.add_tensor(qtn.Tensor(data=np.array([0.0,1.0]),inds=[o]))
    tn.contract_ind(o)
#    print(tn)
    tn.make_tids_consecutive()
#    print(tn)
#    for tid in tn.tensor_map.keys():
#        print(tid,tn.tensor_map[tid])
    if simplify: 
        tn = tn.full_simplify(output_inds=[],**simplify_kwargs)
    opt = ctg.HyperOptimizer(**optimize_kwargs)
    out = tn.contract(output_inds=[],optimize=opt,**contract_kwargs)
    return out
def fit_mps(tn,bdim=10,**kwargs):
    xs = list(tn._outer_inds._d.keys())
    _,output,size_dict = tn.get_inputs_output_size_dict()
    ds = [size_dict[key] for key in output]

    inds = xs[0],qtc.rand_uuid()
    xi = qtn.Tensor(data=np.ones((ds[0],bdim)),inds=inds) 
    mps = qtn.TensorNetwork([xi])
    for i in range(1,len(xs)-1):
        inds = xs[i],inds[-1],qtc.rand_uuid()
        xi = qtn.Tensor(data=np.ones((ds[i],bdim,bdim)),inds=inds) 
        mps.add_tensor(xi)
    inds = xs[-1],inds[-1]
    xi = qtn.Tensor(data=np.ones((ds[i],bdim)),inds=inds) 
    mps.add_tensor(xi)
    return qtc.tensor_network_fit_autodiff(mps,tn,**kwargs)
def plot(tn,name,**kwargs):
    fig = tn.draw(return_fig=True,**kwargs)
    fig.savefig(name)

def chebyshev(tn,typ,p):
    # tn: quimb list
    # typ: 't' or 'u'
    # p: highest degree
    inds = [qtc.rand_uuid()]
    p0 = qtn.TensorNetwork([qtn.Tensor(data=np.ones(2),inds=inds)])

    p1 = tn.copy()
    if typ=='u'or'U':
        p1 = compose_scalar(p1,2.0,'*')

    ls = [p0,p1]
    for i in range(2,p+1):
        tmp1 = compose(tn,ls[-1],'*')
        tmp1 = compose_scalar(tmp1,2.0,'*')
        tmp2 = compose_scalar(ls[-2],-1.0,'*')
        ls.append(compose(tmp1,tmp2,'+'))
    return ls
def horner(tn,coeff,contract=False):
    # coeff = an,...,a0
    inds = [qtc.rand_uuid()]
    b = qtn.TensorNetwork([qtn.Tensor(data=np.array([1.0,coeff[0]]),inds=inds)])
    for a in coeff[1:]:
        b = compose(tn,b,'*')
        b = compose_scalar(b,a,'+',contract)
    return b
if __name__=='__main__':
    d = 20
    n = 50
    p = 5
    xs = [np.random.rand(d) for i in range(n)]
    ins = [np.random.randint(0,len(xs[i])) for i in range(n)]

    tn = train(xs,contract=True)
    x = sum([xs[i][ins[i]] for i in range(n)])

    coeff = np.random.rand(p+1)
    h = horner(tn,coeff,contract=True)
    out = contract(h,ins=ins)
    def _horner(x,coeff):
        b = coeff[0]
        for a in coeff[1:]:
            b = b*x+a
        return b
    print('check horner', _horner(x,coeff)-out)

    typ = 't'
    ts = chebyshev(tn,typ,p)
    out = contract(ts[-1],ins=ins)
    def _chebyshev(x,typ,p):
        p0 = 1.0
        p1 = x
        if typ=='u' or 'U':
            p1 *= 2.0
        ls = [p0,p1]
        for i in range(2,p+1):
            ls.append(2.0*x*ls[-1]-ls[-2])
        return ls
    print('check chebyshev', _chebyshev(x,typ,p)[-1]-out)

