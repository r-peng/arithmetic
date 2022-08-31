import numpy as np
import itertools
import scipy
def approx(xs,ndim,L):
    ng = len(xs)
    idxs = list(itertools.product(range(ng),repeat=ndim))
    rs = np.array([np.sqrt(sum([xs[i]**2 for i in idx])) for idx in idxs])
    sidxs = np.argsort(rs)
    rs = rs[sidxs]
    idxs = idxs[sidxs]

    Y = np.zeros((L,2*L+1,ng**ndim))
    rnorm = np.zeros((L,ng**ndim))
    for i in range(ng**ndim):
        if ndim==2:
            theta = np.pi/2. # [0,pi]
            phi = np.arccos(xs[idx[0]]/rs[i]) #[0,2*pi]
        else:
            theta = np.arccos(xs[idx[-1]]/rs[i])
            phi = np.arctan(xs[idx[1]]/xs[idx[0]])
        for l in range(L):
            for m in range(-l,l+1):
                Y[l,m+l,i] = scipy.special.sph_harm(m,l,phi,theta)

    f = np.zeros((ng**ndim,)*2)
    for i in range(ng**ndim):
        r1 = np.array([xs[ix] for ix in idxs[i]])
        for j in range(i+1,ng**ndim):
            r2 = np.array([xs[ix] for ix in idxs[j]])
            f[i,j] = 1./np.linalg.norm(r1-r2)

    return Y
if __name__=='__main__':
    xmax = 1.
    dx = .1
    xs = np.arange(0.,xmax,dx)
    
