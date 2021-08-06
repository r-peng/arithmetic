import numpy as np
import math
import quimb.tensor as qtn
import quimb.tensor.tensor_core as qtc
#import cotengra as ctg
def tensor(f):
    out = np.zeros(f.shape+(2,))
    out[...,1] = f.copy()
    out[...,0] = np.ones_like(f)
    return out
def P(k):
    out = np.zeros((k,2,k+1))
    for i in range(k):
        out[i,0,i] = 1.0
    out[k-1,1,k] = 1.0
    return out
def binom_sum(n):
    fac = [math.factorial(k) for k in range(n+1)]
    out = np.zeros((n+1,)*3)
    for k in range(n+1):
        for i in range(k+1):
            j = k-i
            out[i,j,k] = fac[k]/(fac[i]*fac[j])
    return out
def make_ins(xs,ins):
    ls = []
    for i in range(len(xs)):
        if ins is None: 
            data = np.ones(len(xs[i]))/len(xs[i])
        else:
            data = np.zeros(len(xs[i]))
            data[ins[i]] = 1.0
        ls.append(data)
    return ls

def xpowers(x,n):
    x = tensor(x)
    out = x.copy()
    for k in range(2,n+1):
        out = np.einsum('xi,xj,ijk->xk',out,x,P(k))
    return out
def uniform(xs,coeff=None,ins=None):
    # poly(q1(x1)+...+qN(xN))
    n = len(coeff)-1
    ADD = binom_sum(n)
    ls = [xpowers(x,n) for x in xs]
    data = make_ins(xs,ins)

    out = np.einsum('xk,x->k',ls[0],data[0])
    for i in range(1,len(xs)):
        tmp = np.einsum('xk,x->k',ls[i],data[i])
        out = np.einsum('i,j,ijk->k',out,tmp,ADD)
    if coeff is None:
        return out
    else:
        return np.dot(out,coeff)
def general(xs,coeff,ins=None):
    # coeff: m*n*N tensor
    m,n,N = coeff.shape
    m -= 1
    data = make_ins(xs,ins)
    def make_c(cki):
        c = np.zeros((m+1,2))
        c[0,0] = 1
        c[:,1] = cki.copy()
        return c

    x = xpowers(xs[0],m)
    out = np.einsum('dm,mi->di',x,make_c(coeff[:,0,0]))
    for k in range(1,n):
        q = np.einsum('dm,mi->di',x,make_c(coeff[:,k,0]))
        out = np.einsum('d...,di->d...i',out,q)
    out = np.einsum('di...,d->i...',out,data[0])

    ADD = np.zeros((2,)*3)
    ADD[1,0,1] = ADD[0,1,1] = ADD[0,0,0] = 1
    for i in range(1,N-1):
        x = xpowers(xs[i],m)
        q = np.einsum('dm,mi->di',x,make_c(coeff[:,0,i]))
        out = np.einsum('j...,di,ijk->d...k',out,q,ADD)
        for k in range(1,n):
            q = np.einsum('dm,mi->di',x,make_c(coeff[:,k,i]))
            out = np.einsum('dj...,di,ijk->d...k',out,q,ADD)
        out = np.einsum('di...,d->i...',out,data[i])

    x = xpowers(xs[-1],m)
    q = np.einsum('dm,mi->di',x,make_c(coeff[:,0,-1]))
    out = np.einsum('j...,di,ij->d...',out,q,ADD[:,:,1])
    for k in range(1,n):
        q = np.einsum('dm,mi->di',x,make_c(coeff[:,k,-1]))
        out = np.einsum('dj...,di,ij->d...',out,q,ADD[:,:,1])
    out = np.einsum('d,d->',out,data[-1])
    return out

def gen_arithmetic(xs,coeff,ins=None,contract_left=[],contract_right=[],label=''):
    # coeff: m*n*N tensor
    m,n,N = coeff.shape
    m -= 1
    data = make_ins(xs,ins)
    def make_c(cki):
        c = np.zeros((m+1,2))
        c[0,0] = 1
        c[:,1] = cki.copy()
        return c
    ADD = np.zeros((2,)*3)
    ADD[1,0,1] = ADD[0,1,1] = ADD[0,0,0] = 1

    ls = []
    linds = []
    rinds = []
    for i in range(N):
        x = xpowers(xs[i],m)
        d = x.shape[0]
        CP = np.zeros((d,)*3)
        for p in range(d):
            CP[p,p,p] = 1
        for k in range(n):
            l = label+'i{},{},'.format(i,k)
            r = label+'i{},{},'.format(i+1,k)
            t = label+'d{},{},'.format(i,k)
            b = label+'d{},{},'.format(i,k+1)
            tags = {label,'q{},{},'.format(i,k)}
            if i==0:
                linds.append(l)
            if i==N-1:
                rinds.append(r)
            q = np.einsum('dm,mi->di',x,make_c(coeff[:,k,i]))
            q = np.einsum('di,ijk->djk',q,ADD)
            if k==n-1:
                inds = t,l,r
            elif k==0:
                q = np.einsum('djk,d->djk',q,data[i])
                inds = b,l,r
            else:
                q = np.einsum('djk,def->efjk',q,CP)
                inds = t,b,l,r
            ls.append(qtn.Tensor(data=q,inds=inds,tags=tags))
    for k in contract_left:
        t = ls[k]
        inds = list(t.inds).copy()
        rm = inds.pop(-2)
        data = np.einsum('...jk,j->...k',t.data,np.array([1,0]))
        t.modify(data=data,inds=inds)
        ls[k] = t
        linds.remove(rm)
    for k in contract_right:
        t = ls[(N-1)*n+k]
        inds = list(t.inds).copy()
        rm = inds.pop(-1)
        data = np.einsum('...jk,k->...j',t.data,np.array([0,1]))
        t.modify(data=data,inds=inds)
        ls[(N-1)*n+k] = t
        rinds.remove(rm)

    tn = qtn.TensorNetwork([])
    for i in range(len(ls)):
        tn.add_tensor(ls[i],tid=i)
    return tn, linds, rinds
def compress(tn,Lx,Ly,linds,rinds,cutoff=0.0,max_bond=None):
    # tid[i,k] = i*n+k
    N,n = Lx,Ly
    def contr(i,k):
        tid1 = i*n
        tid2 = i*n+k
        t1 = tn._pop_tensor(tid1)
        t2 = tn._pop_tensor(tid2)
        t12 = qtc.tensor_contract(t1,t2,preserve_tensor=True)
        tn.add_tensor(t12,tid=tid1)
        return 
    for k in range(1,n):
        contr(0,k)
        for i in range(1,N):
            contr(i,k)
            tn._compress_between_tids((i-1)*n,i*n,max_bond=max_bond,cutoff=cutoff)
    output_inds = linds+rinds
    out = qtc.tensor_contract(*tn.tensors,output_inds=output_inds,
                              preserve_tensor=True)
    return out 
def general_compressed(xs,coeff,ins=None,cutoff=0.0,max_bond=None):
    m,n,N = coeff.shape
    contract_left = list(range(n))
    contract_right = list(range(n))
    tn, linds, rinds = gen2D(xs,coeff,ins=ins,
                             contract_left=contract_left,
                             contract_right=contract_right)
    out = compress(tn,N,n,linds,rinds,cutoff=cutoff,max_bond=max_bond)
    return out.data
def bounded_width_product(xs,coeff,init_inds,ins=None,cutoff=0.0,max_bond=None):
    # init_inds: ls, ls[i]=j for xi
    m,c,N = coeff.shape
    m -= 1
    finds = list(set(init_inds))
    finds.sort()
    fdict = {}
    for j in finds:
        fdict.update({j:[]})
    for i in range(N):
        j = init_inds[i]
        fdict[j] = fdict[j]+[i]
    tn = qtn.TensorNetwork([])
    for b in range(len(finds)):
        j = finds[b]
        xinds = fdict[j]
        xs_ = [xs[i] for i in xinds]
        coeff_ = coeff[:,:,tuple(xinds)]
        ins_ = None if ins is None else [ins[i] for i in fdict[j]]
        low = 0 if b==0 else max(0,finds[b-1]+c-j)
        high = c if b==len(finds)-1 else min(c,finds[b+1]-j)
        cl = list(range(low,c))
        cr = list(range(0,high))
        tn_,l_,r_ = gen2D(xs_,coeff_,ins=ins_,label='f{},'.format(j),
                          contract_left=cl,contract_right=cr)
        data = compress(tn_,len(xs_),c,l_,r_,cutoff=cutoff,max_bond=max_bond)
        l = ['b{},{},'.format(b,i) for i in range(c-len(cl))]
        r = ['b{},{},'.format(b+1,i) for i in range(c-len(cr))]
        tn.add_tensor(qtn.Tensor(data=data.data,inds=l+r),tid=b)
    for b in range(1,len(finds)):
        tn._compress_between_tids(b-1,b,max_bond=max_bond,cutoff=cutoff)
    out = tn.contract(output_inds=[])
    return out 

if __name__=='__main__':
    d = 20
    N = 50
    n = 5
    xs = [np.random.rand(d) for i in range(N)]
    ins = [np.random.randint(0,d) for i in range(N)]

    coeff = np.random.rand(n+1)
    out = contract(xs,coeff,ins=ins)
    y = sum([xs[i][ins[i]] for i in range(N)])
    out_ = 0.0
    for k in range(n+1):
        out_ += coeff[k]*y**k
    print(abs(out-out_)/out_)
