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
def _const_fxn(ds,a,label=''):
    xs = [_tensor(np.ones(di)) for di in ds]
    xs[0][:,1] *= a
    tn = qtn.TensorNetwork([])
    o = qtc.rand_uuid()
    for i in range(len(xs)):
        i_ = '{},'.format(i)
        inds = 'x'+i_,o
        xi = qtn.Tensor(data=_tensor(xs[i]),inds=inds,tags={label,'x',i_})
        tn.add_tensor(xi)
        tn._outer_inds.add(inds[0])
    tn._outer_inds.add(o)
    return tn
ADD = np.zeros((2,2,2)) # i1,i2,o
ADD[0,0,0] = ADD[0,1,1] = ADD[1,0,1] = 1.0
#MUL = _delta(2) # use hyperedge
#DEL = _delta(d)
# x1+...+xn
def _train(xs,label=''):
    tn = qtn.TensorNetwork([])
    indx = 'x0,',qtc.rand_uuid()
    xi = qtn.Tensor(data=_tensor(xs[0]),inds=indx,tags={label,'x','0,'})
    indp = (indx[-1],)
    tn.add_tensor(xi,tid=0,virtual=False)
    tn._outer_inds.add(indx[0])
    tn._inner_inds.add(indx[-1])
    for i in range(1,len(xs)):
        i_ = '{},'.format(i)
        indx = 'x'+i_,qtc.rand_uuid()
        xi = qtn.Tensor(data=_tensor(xs[i]),inds=indx,tags={label,'x',i_})
        indp = indp[-1],indx[-1],qtc.rand_uuid()
        ai = qtn.Tensor(data=ADD,inds=indp,tags={label,'+',i_})
        tn.add_tensor(xi)
        tn.add_tensor(ai)
        tn._outer_inds.add(indx[0])
        tn._inner_inds.add(indx[-1])
        if i==len(xs)-1:
            tn._outer_inds.add(indp[-1])
        else:
            tn._inner_inds.add(indp[-1])
    return tn
def _mult_const(tn,a): #inplace
    n = len(tn._outer_inds)-1
    keys = set(['{},'.format(i) for i in range(n)]+['x','+'])
    labels = set(tn.tag_map.keys())-keys

    data = np.array([1.0,a])
    inds = [tn._outer_inds[-1]]
    tags = labels+{'const'}
    tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags=tags))
    return tn
def _chebyshev(tn,typ,p,label=''):
    # tn: quimb list
    # typ: 't' or 'u'
    # p: highest degree
    n = len(tn._outer_inds)-1
    tids = tn._get_tids_from_tags(tags='x',which='any')
    ds = [tn[i].data.shape[0] for i in tids]
    p0 = _const(ds,1,label)
    p1 = tn.copy(deep=True)
#    if typ=='u'or'U':
#        ls = [p0,p1]
#    for i in range(2,p+1):
#        tn1,tn2 = ls[-1],ls[-2] 
if __name__=='__main__':
    d = 3
    n = 3
    xs = [np.random.rand(d) for i in range(n)]
    tn = _train(xs)
    opt = ctg.HyperOptimizer()
    out = tn.contract(output_inds=tn._outer_inds,optimize=opt)
    ins = [np.random.randint(0,d) for i in range(n)]
    true = sum([xs[i][ins[i]] for i in range(n)])
    print('check train: ', true-out.data[...,1][tuple(ins)])
    print('check train: ', 1.0-out.data[...,0][tuple(ins)])
    print(tn.tag_map)
    print(tn._outer_inds)
    print(tn._get_tids_from_tags(tags='x',which='any'))
