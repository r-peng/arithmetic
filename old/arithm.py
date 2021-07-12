import numpy as np
import quimb.tensor as qtn
import cotengra as ctg
optimize = ctg.ReusableHyperOptimizer(max_repeats=64,parallel='ray',reconf_opts={},directory='ctg_path_chache')
ADD = np.zeros((2,)*3,dtype=complex) #i1/2,o
ADD[0,1,1] = ADD[1,0,1] = ADD[0,0,0] = 1.0
MULT = np.zeros((2,)*3,dtype=complex) #i1/2,o
MULT[1,1,1] = MULT[0,0,0] = 1.0
def scalar_mult(w):
    out = np.zeros((2,)*2,dtype=complex)
    out[1,1] = w
    out[0,0] = 1.0
    return out
def state(b):
    out = np.zeros(2,dtype=complex)
    out[1] = b
def weighted_sum(pars, states):
    # pars: w1/.../n,b
    # states: x1/.../n
    # x' = w1x1+..wnxn+b
    # p(x) = a1x^1+...+anx^n+a0
    ls = []
    n = len(pars) - 1
    # sates
    for i in range(n):
        i_ = '{},'.format(i+1)
        ls.append(qtn.Tensor(data=states[i],inds=('x'+i_,),tags='x'+i_))
    # weights
    for i in range(n):
        i_ = '{},'.format(i+1)
        inds = 'x'+i_,'xw'+i_ 
        ls.append(qtn.Tensor(data=scalar_mult(pars[i]),inds=inds,tags='w'+i_))
    # add
    for i in range(2,n+1):
        i_, im_ = '{},'.format(i), '{},'.format(i-1)
        i1 = 'xw1,' if i == 2 else 's'+im_
        inds = i1,'xw'+i_,'s'+i_
        ls.append(qtn.Tensor(data=ADD,inds=inds,tags='+'))
    inds = 's{},'.format(n),'b','o'
    ls.append(qtn.Tensor(data=state(pars[-1]),inds=inds,tags='b'))

    TN = qtn.TensorNetwork(ls)
#    optimize = ctg.HyperOptimizer(max_repeats=64,parallel='ray',reconf_opts={},directory='ctg_path_chache')
    output_inds = ['o']
    out = TN.contract(optimize=optimize,output_inds=output_inds)
    info = TN.contract(get='path-info',optimize=optimize,output_inds=output_inds)
#    width = info.largest_intermediate
#    cost = info.opt_cost/2
#    width = math.log10(width)
#    cost = math.log10(cost)
    return out, info
def power(state,n): # x^n
    for i in range(2,n+1):
        i_, im_ = '{},'.format(i), '{},'.format(i-1)
        i1 = im_ if i == 2 else 'p'+im_
        inds = i1,i_,'p'+i_
        ls.append(qtn.Tensor(data=MULT,inds=inds,tags='*'))

    TN = qtn.TensorNetwork(ls)
#    optimize = ctg.HyperOptimizer(max_repeats=64,parallel='ray',reconf_opts={},directory='ctg_path_chache')
    output_inds = ['p{}'.format(n)]
    out = TN.contract(optimize=optimize,output_inds=output_inds)
    info = TN.contract(get='path-info',optimize=optimize,output_inds=output_inds)
    return out, info
def layer(W,b,states,a,n):
    # W: w11/...w1(ni),...,w(no)1/...w(no)(ni)
    # b: b1.../(no)
    # states: x1/...(ni)
    # a: a1...an,a0 
    no, ni = W.shape
    assert ni==len(states) and no==len(b)
    outs = []
    cost = 0
    # linear
    for i in range(no):
        out, info = weighted_sum(list(W[i,:])+[b[i]],states)
        outs.append(out)
        cost += info.opt_cost
    # activation
    for i in range(no):
        state = outs[i].copy()
        ps = [state]
        for p in range(2,n+1):
            out, info = power(state, p)
            ps.append(out)
            cost += info.opt_cost
        out, info = weighted_sum(a,ps)
        outs[i] = out
        cost += info.opt_cost
    return outs, cost
def kernel(Ws,bs,inputs,depth,a,n):

