import numpy as np
import quimb.tensor as qtn
import scipy.linalg
from .utils import scale,ADD,CP2
import itertools
def build(xs,theta_map,ltag='l',utag='u',itag='i',xtag='x'):
    nl = len(theta_map)
    print('number of variables=',2*(2**nl-1))
    T = np.zeros((2,)*3)
    T[0,0,0] = T[1,0,1] = T[0,1,1] = T[1,1,1] = 1.

    tn = qtn.TensorNetwork([])
    # form circuit
    # adding unitary
    for l in range(1,nl+1):
        U = np.zeros((2,)*4,dtype=complex)
        for i in range(2):
            for j in range(2):
                U[i,j,i,j] = np.exp(1j*theta_map[l]*i*j)
        for i in range(1,2**l):
            tag = f'{ltag}{l}_{utag}{i}'
            inds = [qtn.rand_uuid() for _ in range(4)]
            tn.add_tensor(qtn.Tensor(data=U.copy(),inds=inds,tags=tag))
    # adding isometry
    for l in range(1,nl):
        for i in range(1,2**l):
            tag = f'{ltag}{l}_{itag}{2*i-1}'
            ix3 = tn[f'{ltag}{l}_{utag}{i}'].inds[0]
            ix1 = tn[f'{ltag}{l+1}_{utag}{2*i-1}'].inds[3]
            ix2 = tn[f'{ltag}{l+1}_{utag}{2*i}'].inds[2]
            tn.add_tensor(qtn.Tensor(data=CP2.copy(),inds=(ix1,ix2,ix3),tags=tag))

            tag = f'{ltag}{l}_{itag}{2*i}'
            ix3 = tn[f'{ltag}{l}_{utag}{i}'].inds[1]
            ix1 = tn[f'{ltag}{l+1}_{utag}{2*i}'].inds[3]
            ix2 = tn[f'{ltag}{l+1}_{utag}{2*i+1}'].inds[2]
            tn.add_tensor(qtn.Tensor(data=CP2.copy(),inds=(ix1,ix2,ix3),tags=tag))
    # trace borders
    tag = f'{ltag}{nl}_{utag}1'
    ix1 = tn[tag].inds[2]
    for l in range(nl-1,0,-1):
        tag = f'{ltag}{l}_{utag}1'
        ix2 = tn[tag].inds[2]
        ix3 = qtn.rand_uuid()
        tn.add_tensor(qtn.Tensor(data=T.copy(),inds=(ix1,ix2,ix3)))
        ix1 = ix3
    lix = ix3
    tag = f'{ltag}{nl}_{utag}{2**nl-1}'
    ix1 = tn[tag].inds[3]
    for l in range(nl-1,0,-1):
        tag = f'{ltag}{l}_{utag}{2**l-1}'
        ix2 = tn[tag].inds[3]
        ix3 = qtn.rand_uuid()
        tn.add_tensor(qtn.Tensor(data=T.copy(),inds=(ix1,ix2,ix3)))
        ix1 = ix3
    rix = ix3
    tn.add_tensor(qtn.Tensor(data=T[:,:,1].copy(),inds=(lix,rix)))

    # add fields
    N,ng = len(xs),len(xs[1])
    for i in range(1,2**nl):
        tag = f'{ltag}{nl}_{utag}{i}'
        ix1,ix2 = tn[tag].inds[:2]
        data = np.ones((ng,2))
        data[:,1] = xs[2*i-1]
        tn.add_tensor(qtn.Tensor(data=data,inds=(f'{xtag}{2*i-1}',ix1),
                                 tags=f'{xtag}{2*i-1}'))
        data = np.ones((ng,2))
        data[:,1] = xs[2*i]
        tn.add_tensor(qtn.Tensor(data=data,inds=(f'{xtag}{2*i}',ix2),
                                 tags=f'{xtag}{2*i}'))
    return tn
def contract_norm(tr,tn,ltag='l',utag='u',itag='i',xtag='x',
                  mangle_append='*',cutoff=1e-15):
    norm = tn.make_norm(mangle_append=mangle_append)
    N,ng = len(tr),len(tr[1])
    nl = int(np.log2(N//2+1.1))
    assert (2**nl-1)*2==N
    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1.
    # contract fiels & weights
    for i in range(1,2**nl):
        for x in [2*i-1,2*i]:
            tag = f'{xtag}{x}'
            norm[tag,'KET'].reindex_({tag:tag+'1'})
            norm[tag,'BRA'].reindex_({tag:tag+'2'})
            norm.add_tensor(qtn.Tensor(data=CP,inds=(tag,tag+'1',tag+'2'),tags=tag))
            norm.add_tensor(qtn.Tensor(data=tr[x],inds=(tag,),tags=tag))

            norm.contract_tags(tag,which='any',inplace=True)
            fac = np.amax(np.abs(norm[tag].data))
            norm.exponent += np.log10(fac)
            norm[tag].modify(data=norm[tag].data/fac,tags=f'{ltag}{nl}_{utag}{i}')
    # contract circuit 
    for l in range(nl,0,-1):
        # split unitary
        for i in range(1,2**l):
            tag = f'{ltag}{l}_{utag}{i}'
            lix,rix = norm[tag,'KET'].inds[2:]
            norm.contract_tags(tag,which='any',inplace=True)
            norm.split_tensor(tag,(lix,lix+mangle_append),
                              ltags='L',rtags='R',rtagscutoff=cutoff)
            norm[tag,'L'].modify(tags=f'{ltag}{l-1}_{itag}{i-1}')
            norm[tag,'R'].modify(tags=f'{ltag}{l-1}_{itag}{i}')
        # contract isometry
        for i in range(1,2**l-1):
            tag = f'{ltag}{l-1}_{itag}{i}'
            norm.contract_tags(tag,which='any',inplace=True)
            norm[tag].modify(tags=f'{ltag}{l-1}_{utag}{(i+1)//2}')
        print(f'spliting layer {l}, max_bond={norm.max_bond()}')
        print(norm)
    out = norm.contract()
    if abs(out.imag)>cutoff:
        print('nonzero imaginary part=',out.imag)
    if out.real<0.:
        print('negative real part=',out.real)
    return np.log10(abs(out.real))+norm.exponent

def get_phase(theta_map,ixs):
    nl = len(theta_map)
    ixs = list(ixs)
    phase = 0.
    for l in range(nl,0,-1):
        fac = 0.
        for i in range(1,len(ixs)):
            fac += ixs[i-1]*ixs[i]
        lix,rix = ixs[:len(ixs)//2],ixs[len(ixs)//2:]
        rix.reverse()
        lix,rix = lix[2:],rix[2:]
        for i in range(len(lix)):
            fac += lix[i]**2*(2**(i+1)-1)
            fac += rix[i]**2*(2**(i+1)-1)
        phase += theta_map[l]*fac
        ixs.pop(0)
        ixs.pop()
    return phase
def get_phase_map(length,theta_map):
    ls = itertools.product(range(2),repeat=length)
    phase_map = dict()
    for ixs in ls:
        out = 0
        for r in range(1,length+1):
            combs = itertools.combinations(ixs,r)
            out += (-1)**(r-1)*sum([np.product(comb) for comb in combs])
        if out==1:
            phase_map[ixs] = get_phase(theta_map,ixs)
    print('number of terms=',len(phase_map))
    return phase_map
def evaluate(xs,phase_map,probability=False):
    xs = list(xs) 
    ni = len(list(phase_map.keys())[0])//2
    xs_ = [] 
    for i in range(ni):
        tmp,xs = xs[:2**i],xs[2**i:]
        xs_.append(np.product(tmp))
    for i in range(ni-1,-1,-1):
        tmp,xs = xs[:2**i],xs[2**i:]
        xs_.append(np.product(tmp))
    out = 0.
    for key,phase in phase_map.items():
        out += np.product([x**i for x,i in zip(xs_,key)])*np.exp(1j*phase)
    if probability:
        out = out.real**2+out.imag**2
    return out
#def get_rand_unitary():
#    K = np.random.rand(*(4,)*2)
#    K -= K.T
#    U = scipy.linalg.expm(K)
#    return U.reshape(*(2,)*4)
#def get_rand_trace(normalize=False):
#    v = np.random.rand(2)
#    if normalize:
#        v /= np.linalg.norm(v)
#    return v 
#def get_rand_isometry(normalize=False):
#    I = get_rand_unitary()
#    v = get_rand_trace(normalize=normalize)
#    return np.einsum('ijkl,l->ijk',I,v)
