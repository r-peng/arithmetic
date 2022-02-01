import numpy as np
import quimb.tensor as qtn
import quimb.tensor.tensor_core as qtc
import math
def get_input_function(f):
    out = np.zeros(f.shape+(2,))
    out[...,1] = f.copy()
    out[...,0] = np.ones_like(f)
    return out
def get_input_vectors(xs,coords):
    out = []
    for i in range(len(xs)):
        if coords=='all': 
            data = np.ones(len(xs[i]))/len(xs[i])
        else:
            data = np.zeros(len(xs[i]))
            data[coords[i]] = 1
        out.append(data)
    return out
def get_powers(x,n):
    x = get_input_function(x) if len(x.shape)==1 else x
    def _projector(k):
        out = np.zeros((k+1,2,k+2))
        out[:,0,:k+1] = np.eye(k+1)
        out[k,1,k+1] = 1
        return out
    out = x.copy()
    for k in range(1,n):
        out = np.einsum('dm,di,min->dn',out,x,_projector(k))
    return out
def get_poly1(x,coeff):
    x = get_powers(x) if len(x.shape)==1 else x
    c = np.zeros((len(coeff),2))
    c[0,0] = 1
    c[:,1] = coeff.copy()
    return np.einsum('dm,mi->di',x,c) 

def arithmetic_contract(xs,coeff,coords='all'):
    # coeff: m*n*N tensor
    m,n,N = coeff.shape
    m -= 1
    data = get_input_vectors(xs,coords)

    x = get_powers(xs[0],m)
    out = get_poly1(x,coeff[:,0,0])
    for k in range(1,n):
        out = np.einsum('d...,di->d...i',out,get_poly1(x,coeff[:,k,0]))
    out = np.einsum('di...,d->i...',out,data[0])

    ADD = np.zeros((2,)*3)
    ADD[1,0,1] = ADD[0,1,1] = ADD[0,0,0] = 1
    for i in range(1,N-1):
        x = get_powers(xs[i],m)
        out = np.einsum('j...,di,ijk->d...k',out,get_poly1(x,coeff[:,0,i]),ADD)
        for k in range(1,n):
            out = np.einsum('dj...,di,ijk->d...k',out,get_poly1(x,coeff[:,k,i]),ADD)
        out = np.einsum('di...,d->i...',out,data[i])

    x = get_powers(xs[-1],m)
    out = np.einsum('j...,di,ij->d...',out,get_poly1(x,coeff[:,0,-1]),ADD[:,:,1])
    for k in range(1,n):
        out = np.einsum('dj...,di,ij->d...',out,get_poly1(x,coeff[:,k,-1]),ADD[:,:,1])
    out = np.einsum('d,d->',out,data[-1])
    return out
def exact_compressible_tn(xs,coeffN,coeff1=None,coords=None):
    ''' coeffN: coeff for poly(q1(x1)+...+qN(xN))
        coeff1: coeff for qi
        coords: 'all': integrate
                'None': leave x legs open
                ls: single coord
    '''       
    n = len(coeffN)-1
    N = len(xs)
    if coeff1 is None:
        ls = [get_powers(x,n) for x in xs]
    else:
        m,N = coeff1.shape
        m -= 1
        ls = [get_powers(x,m) for x in xs]
        ls = [get_poly1(ls[i],coeff1[:,i]) for i in range(N)]
        ls = [get_powers(q,n) for q in ls]

    fac = [math.factorial(k) for k in range(n+1)]
    ADD = np.zeros((n+1,)*3)
    for k in range(n+1):
        for i in range(k+1):
            j = k-i
            ADD[i,j,k] = fac[k]/(fac[i]*fac[j])
    if coords is not None:
        data = get_input_vectors(xs,coords)
        ls = [np.einsum('xm,x->m',ls[i],data[i]) for i in range(N)]
        inds = [['m{},'.format(i)] for i in range(N)]
    else:
        inds = [('d{},0,'.format(i),'m{},'.format(i)) for i in range(N)]
    tn = qtn.TensorNetwork([])
    for i in range(N):
        tn.add_tensor(qtn.Tensor(data=ls[i],inds=inds[i]))
    for i in range(1,N):
        i1 = 'm{},'.format(i-1) if i==1 else '+{},'.format(i-1)
        inds = i1,'m{},'.format(i),'+{},'.format(i)
        tn.add_tensor(qtn.Tensor(data=ADD,inds=inds))
    tn.add_tensor(qtn.Tensor(data=coeffN,inds=['+{},'.format(N-1)]))
    return tn 
def arithmetic_tn(xs,coeff,coords=None,contract_left='all',contract_right='all',label=''):
    # coeff: m*n*N tensor
    m,n,N = coeff.shape
    m -= 1
    if coords is not None: 
        data = get_input_vectors(xs,coords)
    if contract_left=='all':
        contract_left = list(range(n))
    if contract_right=='all':
        contract_right = list(range(n))
    ADD = np.zeros((2,)*3)
    ADD[1,0,1] = ADD[0,1,1] = ADD[0,0,0] = 1

    tn = qtn.TensorNetwork([])
    linds = []
    rinds = []
    for i in range(N):
        x = get_powers(xs[i],m)
        d = x.shape[0]
        CP = np.zeros((d,)*3)
        for p in range(d):
            CP[p,p,p] = 1
        for k in range(n):
            l = label+'i{},{},'.format(i,k)
            r = label+'i{},{},'.format(i+1,k)
            t = label+'d{},{},'.format(i,k)
            b = label+'d{},{},'.format(i,k+1)
            tags = {label,'q','{},{},'.format(i,k)}
            if i==0:
                linds.append(l)
            if i==N-1:
                rinds.append(r)
            q = np.einsum('di,ijk->djk',get_poly1(x,coeff[:,k,i]),ADD)
            if k==n-1:
                inds = t,l,r
            elif k==0 and coords is not None:
                q = np.einsum('djk,d->djk',q,data[i])
                inds = b,l,r
            else:
                q = np.einsum('djk,def->efjk',q,CP)
                inds = t,b,l,r
            tn.add_tensor(qtn.Tensor(data=q,inds=inds,tags=tags),tid=i*n+k)
    for k in contract_left:
        t = tn.tensor_map[k]
        inds = list(t.inds).copy()
        linds.remove(inds.pop(-2))
        data = np.einsum('...jk,j->...k',t.data,np.array([1,0]))
        t.modify(data=data,inds=inds)
    for k in contract_right:
        t = tn.tensor_map[(N-1)*n+k]
        inds = list(t.inds).copy()
        rinds.remove(inds.pop(-1))
        data = np.einsum('...jk,k->...j',t.data,np.array([0,1]))
        t.modify(data=data,inds=inds)
    return tn,linds,rinds 
def bounded_width_product(xs,coeff,init_inds,coords='all'):
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
        tn_ops = {}
        tn_ops.update({'xs':[xs[i] for i in xinds]})
        tn_ops.update({'coeff':coeff[:,:,tuple(xinds)]})
        if coords is None or 'all':
            tn_ops.update({'coords':coords})
        else:
            tn_ops.update({'coords':[coords[i] for i in fdict[j]]})
        low = 0 if b==0 else max(0,finds[b-1]+c-j)
        high = c if b==len(finds)-1 else min(c,finds[b+1]-j)
        tn_ops.update({'contract_left':list(range(low,c))})
        tn_ops.update({'contract_right':list(range(0,high))})
        tn_ops.update({'label':'f{},'.format(j)})
        blk_tn,l,r = arithmetic_tn(**tn_ops)
        data = qtc.tensor_contract(*blk_tn.tensors,output_inds=l+r,
                                   preserve_tensor=True)
        l = ['b{},{},'.format(b,i) for i in range(low)]
        r = ['b{},{},'.format(b+1,i) for i in range(c-high)]
        tn.add_tensor(qtn.Tensor(data=data.data,inds=l+r),tid=b)
    out = tn.contract(output_inds=[])
    return out 

if __name__=='__main__':
    def poly(xs,coeff):
        m,n,N = coeff.shape
        m -= 1
        ls = []
        for k in range(n):
            pk = 0.0
            for i in range(N):
                powers = np.array([xs[i]**p for p in range(m+1)])
                qki = np.dot(powers,coeff[:,k,i])
                pk += qki
            if pk > 1e-10:
                ls.append(pk)
    
        out = 1.0
        for pk in ls:
            out *= pk
        return out
    import time
    d = 20
    N = 4
    m = 3
    n = 4
    coords = 'all'
    xs = [np.random.rand(d) for i in range(N)]

    print('#### random #####')
    coeff = np.random.rand(m+1,n,N)
    print('tn contracting...')
    t = time.time()
    out1 = arithmetic_contract(xs,coeff,coords=coords)
    print('time: ', time.time()-t)
    t = time.time()
    tn,_,_ = arithmetic_tn(xs,coeff,coords=coords)
    out2 = tn.contract(output_inds=[])
    print('time: ', time.time()-t)
    print('integrating...')
    t = time.time()
    out = 0.0
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    out += poly([xs[0][i],xs[1][j],xs[2][k],xs[3][l]],coeff)/d**N
    print('time: ', time.time()-t)
    print('integral,err1,err2',out,out-out1,out-out2)
    
    print('#### exact compressible #####')
    coeff1 = np.random.rand(m+1,N)
    coeffN = np.zeros(n+1)
    coeffN[-1] = 1
    print('tn contracting...')
    t = time.time()
    tn = exact_compressible_tn(xs,coeffN,coeff1=coeff1,coords=coords)
    out1 = tn.contract(output_inds=[])
    print('time: ', time.time()-t)
    coeff = np.zeros((m+1,n,N))
    for i in range(n):
        coeff[:,i,:] = coeff1.copy()
    print('integrating...')
    t = time.time()
    out = 0.0
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    out += poly([xs[0][i],xs[1][j],xs[2][k],xs[3][l]],coeff)/d**N
    print('time: ', time.time()-t)
    print('integral,err',out,out-out1)
    
    print('#### bounded width product #####')
    c = 3
    high = 6
    init_inds = [np.random.randint(low=0,high=high) for i in range(N)]
    print(init_inds)
    coeff = np.random.rand(m+1,c,N)
    print('tn contracting...')
    t = time.time()
    out1 = bounded_width_product(xs,coeff,init_inds,coords=coords)
    print('time: ', time.time()-t)
    n = max(init_inds)+c
    coeff_full = np.zeros((m+1,n,N))
    for i in range(N):
        j = init_inds[i]
        coeff_full[:,j:j+c,i] = coeff[:,:,i]
    print('integrating...')
    t = time.time()
    out = 0.0
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    out += poly([xs[0][i],xs[1][j],xs[2][k],xs[3][l]],coeff_full)/d**N
    print('time: ', time.time()-t)
    print('integral,err',out,out-out1)
    
