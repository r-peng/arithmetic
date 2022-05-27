import numpy as np
import quimb.tensor as qtn
from arithmetic.gaussian import (
    get_field,
    get_quadratic,
    get_quartic,
    add_exponent,
    trace_pol_compress_col,
)
N = 10
ng = 10
n = 6
xs = np.random.rand(ng)
A = np.random.rand(N,N)
B = np.random.rand(N,N,N,N)
idxs = [np.random.randint(low=0,high=ng) for i in range(N)]
tr = {i:np.zeros(ng) for i in range(1,N+1)}
for i in range(1,N+1):
    tr[i][idxs[i-1]] = 1.
vec = np.array([xs[idxs[i]] for i in range(N)])
coeff = {i:np.random.rand() for i in range(n+1)}

tny = get_field(xs,'y',N,iprint=2)
tnA = get_quadratic(tny,A,'y',iprint=2) 
tnB = get_quartic(tny,B,'y',iprint=2) 
tnAB = add_exponent(tnA,tnB,'y',iprint=2)
#out_row = trace_pol(tnAB,'y',tr,coeff,iprint=1,cutoff=1e-15)
out_col = trace_pol_compress_col(tnAB,'y','a',tr,coeff,iprint=2,cutoff=1e-10)

for i in range(1,N+1):
    tny.add_tensor(qtn.Tensor(data=tr[i],inds=(f'y{i}',)))
out = tny.contract()
out.transpose_('p','k')
print('check get_field[0]=',np.linalg.norm(np.ones(N)-out.data[:,0])/np.sqrt(N))
print('check get_field[1]=',np.linalg.norm(vec-out.data[:,1])/np.linalg.norm(vec))

for i in range(1,N+1):
    tnA.add_tensor(qtn.Tensor(data=tr[i],inds=(f'y{i}',)))
out = tnA.contract()
print('check get_exponent[0]=',abs(1.-out.data[0]))
A = np.einsum('ij,i,j->',A,vec,vec)
print('check get_exponent[1]=',abs(A-out.data[1])/abs(A))

for i in range(1,N+1):
    tnB.add_tensor(qtn.Tensor(data=tr[i],inds=(f'y{i}',)))
out = tnB.contract()
print('check get_exponent[0]=',abs(1.-out.data[0]))
B = np.einsum('ijkl,i,j,k,l->',B,vec,vec,vec,vec)
print('check get_exponent[1]=',abs(B-out.data[1])/abs(B))

for i in range(1,N+1):
    tnAB.add_tensor(qtn.Tensor(data=tr[i],inds=(f'y{i}',)))
out = tnAB.contract()
print('check add_exponent[0]=',abs(1.-out.data[0]))
AB = A+B 
print('check add_exponent[1]=',abs(AB-out.data[1])/abs(AB))
exit()

terms = [coeff[i]*AB**i for i in range(n+1)]
pol = sum(terms) 
#print('check trace_pol_compress_row=',abs(pol-out_row)/abs(pol))
print('check trace_pol_compress_col=',abs(pol-out_col)/abs(pol))
