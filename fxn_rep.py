import numpy as np
import quimb.tensor as qtn
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
def _const_fxn(ds,a,label=''):
    xs = [_tensor(np.ones(di)) for di in ds]
    xs[0][:,1] *= a
    tn = qtn.TensorNetwork([])
    o = ''.join(['{},'.format(i) for i in range(n)])
    for i in range(len(xs)):
        i_ = '{},'.format(i)
        inds = 'x'+i_, label+o
        tn.add(qtn.Tensor(data=_tensor(xs[i]),inds=inds,tags={label,inds[0]}))
    return qtn.TensorNetwork(ls) 
ADD = np.zeros((2,2,2)) # i1,i2,o
ADD[0,0,0] = ADD[0,1,1] = ADD[1,0,1] = 1.0
#MUL = _delta(2) # use hyperedge
#DEL = _delta(d)
# x1+...+xn
def _train(xs,label=''):
    ls = []
    for i in range(len(xs)):
        i_ = '{},'.format(i)
        inds = 'x'+i_, label+i_
        ls.append(qtn.Tensor(data=_tensor(xs[i]),inds=inds,tags={label,inds[0]}))
    inds = ['0,']
    for i in range(1,len(xs)):
        i_ = '{},'.format(i)
        inds = inds[-1],i_,inds[-1]+i_
        ls.append(qtn.Tensor(data=ADD,inds=[label+i for i in inds],tags={label,'+'}))
    return ls
def _chebyshev(n,tn,typ,p,label=''):
    # n: number of variables
    # tn: quimb list
    # typ: 't' or 'u'
    # p: highest degree
    ds = [tn[i].data.shape[0] for i in range(n)]
    p0 = _const(ds,1,label)
    p1 = tn.copy()
    if typ=='u'or'U':
        data = np.array([1.0,2.0])
        inds = [p1[-1].inds[-1]]
        tags = {label,'2'}
        p1.append(qtn.Tensor(data=data,inds=inds,tags=tags))
    
if __name__=='__main__':
    d = 3
    n = 3
    xs = [np.random.rand(d) for i in range(n)]
    tn = _train(xs)
    output_inds = [tn[i].inds[0] for i in range(n)]+[tn[-1].inds[-1]]
    print(output_inds)
    opt = ctg.HyperOptimizer()
    tn = qtn.TensorNetwork(tn)
    print(tn)
    out = tn.contract(output_inds=output_inds,optimize=opt)
    ins = [np.random.randint(0,d) for i in range(n)]
    true = sum([xs[i][ins[i]] for i in range(n)])
    print('check train: ', true-out.data[...,1][tuple(ins)])
    print('check train: ', 1.0-out.data[...,0][tuple(ins)])
     
