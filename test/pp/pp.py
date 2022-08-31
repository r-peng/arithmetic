import numpy as np
import arithmetic.pp.gauss as gauss 
import scipy,functools
import quimb.tensor as qtn
import pickle

xmax = 2.
beta = 1.
density_fxn = functools.partial(gauss.lennard_jones_density,beta=beta)

ng = 50
ndim = 2
ranges = [(0.,xmax)] * ndim
points,weights = gauss.get_quad(ranges,ng)
density = np.array([density_fxn(point) for point in points])
I = np.dot(weights,density)
print(f'ng={ng},I={I}')

nbasis = 3
normalize = False
centers,variances,coeffs = gauss.initialize(xmax,ng,nbasis,ndim,density_fxn,
                                            normalize=normalize)
density /= I
if normalize:
    coeffs /= I
    basis,_,_ = gauss.batched_basis_fxns(centers,variances,points,normalize=normalize)
    print('check sum coeffs=',np.fabs(np.sum(coeffs)-1.))
    print('check integration=',np.dot(weights,basis)-np.ones(len(coeffs)))
print('coeffs=',coeffs)
print('variainces=',variances)
print('centers=')
print(centers.T)

objects = gauss.optimize(points,density,weights,centers,variances,coeffs,
                         normalize=normalize,macroiter=100,atol=1e-5,
                         microiter=100,gtol=1e-5)
print('coeffs=',objects['coeffs'])
print('variainces=',objects['variances'])
print('centers=')
print(objects['centers'].T)
