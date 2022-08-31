import numpy as np
import findiff
import quimb.tensor as qtn
np.set_printoptions(suppress=True,linewidth=1000,precision=6)
N = 1000
ng = 10
qs = np.random.rand(N,ng)

# mps
ADD = np.zeros((2,)*3)
ADD[0,0,0] = ADD[1,0,1] = ADD[0,1,1] = 1.
arrs = []
for i in range(N):
    data = np.ones((2,ng))
    data[1,:] = qs[i,:].copy()
    if i>0:
        data = np.einsum('ijk,jx->ikx',ADD,data)
    if i==N-1:
        data = data[:,1,:]
    arrs.append(data)
mps2 = qtn.MatrixProductState(arrs,shape='lrp')
# check mps2
idxs = [np.random.randint(low=0,high=ng) for i in range(N)]
val = 0.
for i in range(N):
    val += qs[i,idxs[i]]
arrs = []
for i in range(N):
    data = np.zeros(ng)
    data[idxs[i]] = 1.
    if i==0 or i==N-1:
        data = np.reshape(data,(1,ng))
    else:
        data = np.reshape(data,(1,1,ng))
    arrs.append(data)
bra = qtn.MatrixProductState(arrs,shape='lrp')
def contract(ket,bra):
    tmp = ket.copy()
    tmp.add_tensor_network(bra,check_collisions=True)
    for i in range(1,N):
        tmp.contract_tags((tmp.site_tag(i),tmp.site_tag(i-1)),inplace=True)
        tmp.strip_exponent(tmp[0],1.)
    return tmp.contract(),tmp.exponent
out,exp = contract(mps2,bra)
print('check mps2=',np.fabs(out*10.**exp/val-1.),val)
BB,expBB = contract(mps2,mps2)
normsq = np.log10(BB)+expBB
print('<mps2|mps2>=',normsq)

# CP rank-N
if N<50:
    arrs = []
    CP = np.zeros((N,)*3)
    for i in range(N):
        CP[i,i,i] = 1.
    for i in range(N):
        data = np.ones((N,ng))
        data[i,:] = qs[i,:]
        if i>0:
            data = np.einsum('ijk,jx->ikx',CP,data)
        if i==N-1:
            data = np.einsum('ikx,k->ix',data,np.ones(N))
        arrs.append(data) 
    mpsN = qtn.MatrixProductState(arrs,shape='lrp')
    # check mpsN
    AA,expAA = contract(mpsN,mpsN)
    AB,expAB = contract(mpsN,mps2)
    print('check mpsN=',np.fabs(1.+AA/BB*10.**(expAA-expBB)-2.*AB/BB*10.**(expAB-expBB)))

# CP rank-log(N)
precision = np.finfo(BB).eps
err = 1e-6
log_precision = np.log10(precision)
log_err = np.log10(err)
log_N = np.log10(N)
rank = 4
#exit()
for alpha in [.5,.1,.01,.001]:
    print('alpha=',alpha)
    #log_alpha = log_precision / (rank+1.)
    #log_alpha = log_precision / (rank+1e-15)
    #log_alpha = -3.
    log_alpha = np.log10(alpha)
    log_h = log_alpha - log_N
    center = findiff.coefficients(deriv=1,acc=rank,symbolic=False)['center']
    coeffs = np.array(center['coefficients'],dtype=float)
    offset = np.array(center['offsets'],dtype=float)
    offset = offset[np.fabs(coeffs)>err]
    coeffs = coeffs[np.fabs(coeffs)>err]
    cond = np.linalg.norm(coeffs)
    log_cond = np.log10(cond)
    print(f'condition={log_cond},h={log_h},precision={log_precision},cond/h*precision={log_cond - log_h + log_precision}')
    #exit()
    offset *= 10.**log_h
    CP = np.zeros((rank,)*3)
    for i in range(rank):
        CP[i,i,i] = 1.
    exp = -log_h
    arrs = []
    for i in range(N):
        data = np.ones((rank,ng))
        data += np.einsum('l,x->lx',offset,qs[i,:])
        fac = np.linalg.norm(data)
        data /= fac
        exp += np.log10(fac)
        if i>0:
            data = np.einsum('ijk,jx->ikx',CP,data)
        if i==N-1:
            data = np.einsum('ikx,k->ix',data,coeffs)
        arrs.append(data) 
    mpslogN = qtn.MatrixProductState(arrs,shape='lrp')
    mpslogN.exponent = exp
    AA,expAA = contract(mpslogN,mpslogN)
    AB,expAB = contract(mpslogN,mps2)
    print('check mpslogN=',np.fabs(1.+AA/BB*10.**(expAA-expBB)-2.*AB/BB*10.**(expAB-expBB)))
