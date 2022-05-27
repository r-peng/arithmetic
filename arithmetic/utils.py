import numpy as np
import quimb.tensor as qtn
import pickle
ADD = np.zeros((2,)*3)
ADD[0,1,1] = ADD[1,0,1] = ADD[0,0,0] = 1.
CP2 = np.zeros((2,)*3)
CP2[0,0,0] = CP2[1,1,1] = 1.
def get_cheb_coeff(fxn,order,a=-1.,b=1.):
    N = order + 1
    c = []
    theta = np.array([np.pi*(k-0.5)/N for k in range(1,N+1)])
    x = np.cos(theta)*(b-a)/2.+(b+a)/2.
    for j in range(order+1):
        v1 = np.array([fxn(xk) for xk in x])
        v2 = np.array([np.cos(j*thetak) for thetak in theta])
        c.append(np.dot(v1,v2)*2./N)
    coeff = np.polynomial.chebyshev.cheb2poly(c)
    coeff[0] -= 0.5*c[0]

    A,B = 2./(b-a),-(b+a)/(b-a)
    c = np.zeros_like(coeff)
    fac = [1]
    for i in range(1,order+1):
        fac.append(fac[-1]*i)
    for i in range(order+1):
        for j in range(i+1):
            c[j] += coeff[i]*A**j*B**(i-j)*fac[i]/(fac[i-j]*fac[j])
    return c
def scale(tn):
    for tid in tn.tensor_map:
        T = tn.tensor_map[tid]
        fac = np.amax(np.absolute(T.data))
        T.modify(data=T.data/fac)
        tn.exponent += np.log10(fac)
    return tn
def compress1D(tn,tag,maxiter=10,final='left',shift=0,iprint=0,**compress_opts):
    L = tn.num_tensors
    max_bond = tn.max_bond()
    lrange = range(shift,L-1+shift)
    rrange = range(L-1+shift,shift,-1)
    if iprint>0:
        print('init max_bond',max_bond)
    def canonize_from_left():
        if iprint>1:
            print('canonizing from left...')
        for i in lrange:
            if iprint>2:
                print(f'canonizing between {tag}{i},{i+1}...')
            tn.canonize_between(f'{tag}{i}',f'{tag}{i+1}',absorb='right')
    def canonize_from_right():
        if iprint>1:
            print('canonizing from right...')
        for i in rrange:
            if iprint>2:
                print(f'canonizing between {tag}{i},{i-1}...')
            tn.canonize_between(f'{tag}{i-1}',f'{tag}{i}',absorb='left')
    def compress_from_left():
        if iprint>1:
            print('compressing from left...')
        for i in lrange:
            if iprint>2:
                print(f'compressing between {tag}{i},{i+1}...')
            tn.compress_between(f'{tag}{i}',f'{tag}{i+1}',absorb='right',**compress_opts)
    def compress_from_right():
        if iprint>1:
            print('compressing from right...')
        for i in rrange:
            if iprint>2:
                print(f'compressing between {tag}{i},{i-1}...')
            tn.compress_between(f'{tag}{i-1}',f'{tag}{i}',absorb='left',**compress_opts)
    if final=='left':
        canonize_from_left()
        def sweep():
            compress_from_right()
            compress_from_left()
    elif final=='right':
        canonize_from_right()
        def sweep():
            compress_from_left()
            compress_from_right()
    else:
        raise NotImplementedError(f'{final} canonical form not implemented!')
    for i in range(maxiter):
        sweep()
        max_bond_new = tn.max_bond()
        if iprint>0:
            print(f'iter={i},max_bond={max_bond_new}')
        if max_bond==max_bond_new:
            break
        max_bond = max_bond_new
    return tn
def load_tn_from_disc(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    tn = qtn.TensorNetwork([])
    for tid,ten in data['tensors'].items():
        T = qtn.Tensor(ten.data, inds=ten.inds, tags=ten.tags)
        tn.add_tensor(T,tid=tid,virtual=True)
    extra_props = dict()
    for name,prop in data['tn_info'].items():
        extra_props[name[1:]] = prop
    tn.exponent = data['exponent']
    tn = tn.view_as_(data['class'], **extra_props)
    return tn
def write_tn_to_disc(tn,fname):
    data = dict()
    data['class'] = type(tn)
    data['tensors'] = dict()
    for tid,T in tn.tensor_map.items():
        data['tensors'][tid] = T 
    data['tn_info'] = dict()
    for e in tn._EXTRA_PROPS:
        data['tn_info'][e] = getattr(tn, e)
    data['exponent'] = tn.exponent
    with open(fname, 'wb') as f:
        pickle.dump(data, f)
    return 
