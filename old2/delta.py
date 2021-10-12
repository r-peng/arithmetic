import numpy as np
import quimb.tensor as qtn
import quimb.tensor.tensor_core as qtc
import cotengra as ctg
# basic tensors
def _delta(d):
    out = np.zeros((d,)*3)
    for i in range(d):
        out[i,i,i] = 1.0
    return out
def _tensor(f):
    out = np.zeros(f.shape+(2,))
    out[...,1] = f.copy()
    out[...,0] = np.ones_like(f)
    return out
ADD = np.zeros((2,2,2)) # i1,i2,o
ADD[0,0,0] = ADD[0,1,1] = ADD[1,0,1] = 1.0
CP2 = _delta(2)

def _compose(tn1,tn2,xs1,xs2,o1,o2,op,common): 
    if op=='+':
        OP = ADD
    elif op=='*':
        OP = CP2
    else:
        raise(NotImplementedError)
    tn = tn1.copy()
    tn.add_tensor_network(tn2)

    o = qtc.rand_uuid()
    tn.add_tensor(qtn.Tensor(data=OP,inds=(o1,o2,o),tags=op))
    if common:
        xs = []
        assert len(xs1)==len(xs2)
        for i in range(len(xs1)):
            tid1 = list(tn1._get_tids_from_inds(xs1[i])._d.keys())[0]
            d1 = tn1.tensor_map[tid1].data.shape[0]
            tid2 = list(tn2._get_tids_from_inds(xs2[i])._d.keys())[0]
            d2 = tn2.tensor_map[tid2].data.shape[0]
            assert d1==d2
            xs.append(qtc.rand_uuid())
            inds = xs1[i],xs2[i],xs[-1]
            tn.add_tensor(qtn.Tensor(data=_delta(d1),inds=inds,tags='d'))
    else:
        xs = xs1+xs2
    return tn,xs,o
def _train(fs,op='+'):
    data = _tensor(fs[0])
    inds = qtc.rand_uuid(),qtc.rand_uuid() 
    tn = qtn.TensorNetwork([qtn.Tensor(data=data,inds=inds,tags='x')])
    xs,o = [inds[0]], inds[-1]
    for i in range(1,len(fs)):
        data = _tensor(fs[i])
        inds = qtc.rand_uuid(),qtc.rand_uuid() 
        tni = qtn.TensorNetwork([qtn.Tensor(data=data,inds=inds,tags='x')])
        tn,xs,o = _compose(tn,tni,xs,[inds[0]],o,inds[-1],op,False)
    return tn,xs,o
def _const_fxn(ds,a):
    xs = [np.ones(di) for di in ds]
    xs[0] *= a
    return _train(xs,'*')
def _scalar_mult(tn,a):
    data = np.array([1.0,a])
    inds = [qtc.rand_uuid()]
    a = qtn.TensorNetwork([qtn.Tensor(data=data,inds=inds,tags='c')])
    return _compose(tn,a,'*')
def _chebyshev(tn,x,o,typ,p):
    # tn: quimb list
    # typ: 't' or 'u'
    # p: highest degree
    n = len(tn._outer_inds)-1

    tids = tn._get_tids_from_tags(tags='x',which='any')
    ds = [tn[i].data.shape[0] for i in tids]
    p0,xs0,o0 = _const_fxn(ds,1.0)

    const = 1.0 if typ='t' or 'T' else 2.0
    p1,xs1,o1 =_mult_const(p1,2.0)

    ls = [p0,p1]
    for i in range(2,p+1):
        tn1,tn2 = ls[-1].copy(deep=True),ls[-2].copy(deep=True) 
class tensor_fxn():
    def __init__(tn,xs,o):
        self.tn = tn
        self.xs = xs
        self.o = o
    def compose(self,f,op,common):
        self.tn,self.xs,self.o = compose(self.tn,f.tn,self.xs,f.xs,self.o,f.o,
                                         op,common)
if __name__=='__main__':
    d = 3
    n = 3

    fs = [np.random.rand(d) for i in range(n)]
    tn,xs,o = _train(fs,'+')
    opt = ctg.HyperOptimizer()
    out = tn.contract(output_inds=xs+[o],optimize=opt)
    ins = [np.random.randint(0,d) for i in range(n)]
    true = sum([fs[i][ins[i]] for i in range(n)])
    print('check train: ', true-out.data[...,1][tuple(ins)])
    print('check train: ', 1.0-out.data[...,0][tuple(ins)])

    ds = [d for i in range(n)]
    a = np.random.rand()
    tn,xs,o = _const_fxn(ds,a)
    opt = ctg.HyperOptimizer()
    out = tn.contract(output_inds=xs+[o],optimize=opt)
    ins = [np.random.randint(0,d) for i in range(n)]
    true = a 
    print('check const: ', true-out.data[...,1][tuple(ins)])
    print('check const: ', 1.0-out.data[...,0][tuple(ins)])

    f1s = [np.random.rand(d) for i in range(n)]
    tn1,xs1,o1 = _train(f1s,'+')
    f2s = [np.random.rand(d) for i in range(n)]
    tn2,xs2,o2 = _train(f2s,'+')
    tn,xs,o = _compose(tn1,tn2,xs1,xs2,o1,o2,'*',True)
    opt = ctg.HyperOptimizer()
    out = tn.contract(output_inds=xs+[o],optimize=opt)
    ins = [np.random.randint(0,d) for i in range(n)]
    true  = sum([f1s[i][ins[i]] for i in range(n)])
    true *= sum([f2s[i][ins[i]] for i in range(n)])
    print('check mult: ', true-out.data[...,1][tuple(ins)])
    print('check mult: ', 1.0-out.data[...,0][tuple(ins)])


