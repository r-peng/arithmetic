import numpy as np
import scipy.linalg
import quimb.tensor as qtn
from arithmetic.gaussian import (
    get_field,
    get_quadratic,
    simplify_exponent
)
from arithmetic.gaussian_disentangle import (
    disentangle,
)
np.set_printoptions(precision=8,suppress=True)
N = 10
tag,utag = 'x','u'

D = np.random.rand(N)
K = np.random.rand(N,N)
K -= K.T
U = scipy.linalg.expm(K)
A = np.linalg.multi_dot([U,np.diag(D),U.T])

ng = 4
xs,ws = np.polynomial.legendre.leggauss(ng)

cutoff = 1e-15
max_bond = 4
rand_scale = 1e-3
tny = get_field(xs,tag,N,iprint=0,cutoff=cutoff)
tnA = get_quadratic(tny,A,tag,iprint=0,cutoff=cutoff)
tnA = simplify_exponent(tnA,tag,cutoff=cutoff)
tnA = disentangle(tnA,tag,utag,max_bond,cutoff=cutoff,rand_scale=rand_scale)
