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
MUL = np.zeros((2,2,2)) # i1,i2,o
MUL[0,0,0] = MUL[1,1,1] = 1.0

def compose(f1,f2,op):
    tn = f1.tn.copy()
    tn.add_tensor_network(f2.tn)

    if op=='*':
        tn.reindex(index_map={f2.o:f1.o},inplace=True)
        o = f1.o
    elif op=='+':
        o = qtc.rand_uuid()
        inds = f1.o,f2.o,o 
        tn.add_tensor(qtn.Tensor(data=ADD,inds=inds))

    x = f1.x.copy()
    for xi in f2.x.keys():
        if xi not in f1.x.keys():
            x.update({xi:f2.x[xi]})
    return fxn(tn,x,o)
def train(xs,op='+'):
    inds = 'x0,',qtc.rand_uuid()
    tn = qtn.TensorNetwork([qtn.Tensor(data=tensor(xs[0]),inds=inds)])
    f = fxn(tn,{inds[0]:len(xs[0])},inds[-1])
    for i in range(1,len(xs)):
        inds = 'x{},'.format(i),qtc.rand_uuid()
        tni = qtn.TensorNetwork([qtn.Tensor(data=tensor(xs[i]),inds=inds)])
        fi = fxn(tni,{inds[0]:len(xs[i])},inds[-1])
        f = compose(f,fi,op)
    return f
def const_fxn(ds,a):
    xs = [np.ones(di) for di in ds]
    xs[0] *= a
    return train(xs,op='*')
def scalar_mult(f,a):
    inds = [qtc.rand_uuid()]
    a = qtn.TensorNetwork([qtn.Tensor(data=np.array([1.0,a]),inds=inds)])
    return compose(f,fxn(a,dict(),inds[-1]),'*')
def chebyshev(f,typ,p):
    # tn: quimb list
    # typ: 't' or 'u'
    # p: highest degree
    keys = ['x{},'.format(i) for i in range(len(f.x.keys()))]
    ds = [f.x[key] for key in keys]
    p0 = const_fxn(ds,1.0)

    p1 = fxn(f.tn.copy(),f.x.copy(),f.o)
    if typ=='u'or'U':
        p1 = scalar_mult(p1,2.0)

    ls = [p0,p1]
    for i in range(2,p+1):
        tmp1 = compose(f,ls[-1],'*')
        tmp1 = scalar_mult(tmp1,2.0)
        tmp2 = scalar_mult(ls[-2],-1.0)
        ls.append(compose(tmp1,tmp2,'+'))
    for p in ls:
        print(p.tn)
    return ls 
def simplify(f,seq='ADCRS',atol=1e-15,equalize_norm=True):
    output_inds = ['x{},'.format(i) for i in range(len(f.x.keys()))]+[f.o]
    f.tn = f.tn.full_simplify(seq=seq,output_inds=output_inds,
                              atol=atol,equalize_norms=equalize_norm)
def contract(f,ins,seq='ADCRS',atol=1e-12,equalize_norm=True): 
    import cotengra as ctg
    keys = ['x{},'.format(i) for i in range(len(f.x.keys()))]
    ds = [f.x[key] for key in keys]
    for i in range(len(ds)):
        data = np.zeros(ds[i])
        data[ins[i]] = 1.0
        f.tn.add_tensor(qtn.Tensor(data=data,inds=['x{},'.format(i)],tags='i'))
    tn = f.tn.full_simplify(seq=seq,output_inds=[f.o],atol=atol,
                            equalize_norms=equalize_norm)
    opt = ctg.HyperOptimizer()
    out = tn.contract(output_inds=[f.o],optimize=opt)
    return out.data
class fxn():
    def __init__(self,tn,x,o):
        self.tn = tn
        self.o = o
        self.x = x 
if __name__=='__main__':
    d = 3
    n = 3

    xs = [np.random.rand(d) for i in range(n)]
    f = train(xs)
    ins = [np.random.randint(0,d) for i in range(n)]
    out = contract(f,ins)
    true = sum([xs[i][ins[i]] for i in range(n)])
    print('check train: ', true-out[1])
    print('check train: ', 1.0 -out[0])

    ds = [d for i in range(n)]
    a = np.random.rand()
    f = const_fxn(ds,a)
    ins = [np.random.randint(0,d) for i in range(n)]
    out = contract(f,ins)
    true = a 
    print('check const: ', true-out[1])
    print('check const: ', 1.0 -out[0])

    xs = [np.random.rand(d) for i in range(n)]
    a = np.random.rand()
    f = train(xs)
    f = scalar_mult(f,a)
    ins = [np.random.randint(0,d) for i in range(n)]
    out = contract(f,ins)
    true = a*sum([xs[i][ins[i]] for i in range(n)])
    print('check scalar_mult: ', true-out[1])
    print('check scalar_mult: ', 1.0 -out[0])

    x1s = [np.random.rand(d) for i in range(n)]
    f1 = train(x1s)
    x2s = [np.random.rand(d) for i in range(n)]
    f2 = train(x2s)
    f = compose(f1,f2,'*')
    ins = [np.random.randint(0,d) for i in range(n)]
    out = contract(f,ins)
    true  = sum([x1s[i][ins[i]] for i in range(n)])
    true *= sum([x2s[i][ins[i]] for i in range(n)])
    print('check mult: ', true-out[1])
    print('check mult: ', 1.0 -out[0])


