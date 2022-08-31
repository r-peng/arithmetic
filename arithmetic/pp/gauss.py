import numpy as np
import itertools,scipy
np.set_printoptions(suppress=True,precision=6,linewidth=1000)
SQRT_PI = np.sqrt(np.pi)
def lennard_jones_density(point,beta,thresh=1e-2):
    normsq = np.dot(point,point)
    if normsq < thresh:
        return 0.
    else:
        return np.exp(-beta*(1./normsq**6.-1./normsq**3))
def get_quad(ranges,ng):
    ndim = len(ranges)
    x,w = np.polynomial.legendre.leggauss(ng)
    rs = []
    ws = []
    for i in range(ndim):
        a,b = ranges[i]
        rs.append(x*(b-a)/2. + (a+b)/2.)
        ws.append(w*(b-a)/2.)

    idxs = list(itertools.product(range(ng),repeat=ndim))
    points = np.zeros((len(idxs),ndim))
    weights = np.zeros((len(idxs),ndim))
    for i,idx in enumerate(idxs):
        points[i,:] = [rs[k][ix] for k,ix in enumerate(idx)]
        weights[i,:] = [ws[k][ix] for k,ix in enumerate(idx)]
    weights = np.prod(weights,axis=-1)
    return points,weights
def compute_center(ranges,ng,density_fxn):
    points,weights = get_quad(ranges,ng)
    density = np.array([density_fxn(point) for point in points])
    I = np.dot(weights,density)
    vol = np.prod([b-a for a,b in ranges])
    mean = I / vol

    err = np.array([(y-mean)**2 for y in density])
    r = np.linalg.norm(points[np.argsort(err)[0],:])
    center = np.array([(a+b)/2. for a,b in ranges])
    center *= r/np.linalg.norm(center)
    return center,I
def compute_variance(center,xmax,thresh=1e-6,incre=.1,normalize=False):
    min_dist_sq = np.amin([np.amin([x**2,(xmax-x)**2]) for x in center]) \
                  if normalize else np.amin([x**2 for x in center])
    variance = np.sqrt(-np.log(thresh)/min_dist_sq)
    ndim = len(center)
    while True:
        val = np.exp(-variance**2*min_dist_sq)
        if normalize:
            val *=(variance/SQRT_PI)**ndim 
        if val < thresh:
            break
        variance *= (1.+incre)
    return variance
def initialize(xmax,ng,nbasis,ndim,density_fxn,thresh=1e-6,incre=.1,normalize=False):
    dx = xmax/nbasis
    idxs = list(itertools.product(range(nbasis),repeat=ndim))
    vols = [[(ix*dx,(ix+1.)*dx) for ix in idx] for idx in idxs]
    centers = np.zeros((len(vols),ndim))
    coeffs = np.zeros(len(vols))
    variances = np.zeros(len(vols))
    for i,ranges in enumerate(vols):
        centers[i,:],coeffs[i] = compute_center(ranges,ng,density_fxn)
        variances[i] = compute_variance(centers[i,:],xmax,
                       thresh=thresh,incre=incre,normalize=normalize)
    return centers,variances,coeffs
def batched_basis_fxns(centers,variances,points,normalize=False):
    npoints,ndim = points.shape
    nbasis = len(variances)

    displacements = points.reshape(npoints,1,ndim)\
                  - centers.reshape(1,nbasis,ndim)
    normsq = np.einsum('riv,riv->ri',displacements,displacements)
    basis = np.exp(-np.square(variances).reshape(1,nbasis)*normsq)
    if normalize:
        basis *= (variances.reshape(1,nbasis) / SQRT_PI) ** ndim
    return basis,normsq,displacements
def err(coeff,basis,density):
    return density - np.dot(basis,coeff)
def loss(coeffs,basis,density,weights):
    return np.dot(weights,np.square(err(coeffs,basis,density)))
#############################################################################
# optimize coeffs
#############################################################################
def quad_loss(const,quad,lin,coeffs):
    return const+np.dot(coeffs,np.dot(quad,coeffs))-2.*np.dot(lin,coeffs)
def quad_params(basis,density,weights):
    const = np.dot(weights,np.square(density))
    quad = np.einsum('ri,rj,r->ij',basis,basis,weights)
    lin = np.dot(weights*density,basis)
    return const,quad,lin
def optimize_coeff_inv(basis,density,weights,affine=False):
    const,quad,lin = quad_params(basis,density,weights)
    coeffs = np.dot(np.linalg.inv(quad),lin)
    sum_coeffs = np.sum(coeffs)
    if affine:
        coeffs /= sum_coeffs
    else:
        print('sum coeffs=',sum_coeffs)
    loss = quad_loss(const,quad,lin,coeffs)
    return coeffs,loss
def optimize_coeffs_affine(basis,density,weights,affine_idx=None):
    if affine_idx is None:
        _basis = basis
        _density = density
    else:
        npoints = len(density)
        _basis = np.delete(basis,affine_idx,axis=-1)
        _basis -= basis[:,affine_idx].reshape(npoints,1)
        _density = density - basis[:,affine_idx]

    const,quad,lin = quad_params(_basis,_density,weights)
    _,nbasis = _basis.shape
    import cvxpy as cp
    coeffs = cp.Variable(nbasis)
    objective = cp.Minimize(cp.norm(quad@coeffs - lin))
    if affine_idx is None:
        constraints = [coeffs>=0, cp.sum(coeffs)==1]
    else:
        constraints = [coeffs>=0, cp.sum(coeffs)<=1]
    prob = cp.Problem(objective,constraints)
    prob.solve()
    coeffs = coeffs.value
    print("status:", prob.status)
    loss = quad_loss(const,quad,lin,coeffs)
    if affine_idx is not None:
        coeffs = np.insert(coeffs,affine_idx,1.-np.sum(coeffs))
    return coeffs,loss
def optimize_coeffs(basis,density,weights,affine=False):
    data = dict()
    data['free'] = optimize_coeff_inv(basis,density,weights,affine=False)
    if affine:
        data['inverse'] = optimize_coeff_inv(basis,density,weights,affine=True)
        _,nbasis = basis.shape
        for affine_idx in [None] + list(range(nbasis)):
            data[affine_idx] = optimize_coeffs_affine(basis,density,weights,
                                                      affine_idx=affine_idx)
    return data 
#############################################################################
# optimize variances & centers 
#############################################################################
class OptimizeCoeffs:
    def __init__(self,centers,variances,points,density,weights,normalize=False):
        self.basis,_,_ = batched_basis_fxns(centers,variances,points,
                         normalize=normalize)
        self.density = density
        self.weights = weights
        self.normalize = normalize

        self.nf = 0
        self.ng = 0
        self.niter = 0
    def loss(self,x):
        self.f = loss(np.square(x),self.basis,self.density,self.weights)
        self.nf += 1
        return self.f
    def grad(self,x):
        _err = err(np.square(x),self.basis,self.density)
        f = np.dot(self.weights,np.square(_err))
        g = -4.*np.dot(self.weights*_err,self.basis) * x 
        self.ng += 1
        self.f = f
        self.g = g
        return f,g
    def callback(self,x):
        self.niter += 1
        if self.niter % self.every==0:
            print(f'niter={self.niter},loss={self.f},g={np.linalg.norm(self.g)}')
            print('coeffs=',np.square(x))
        return
    def kernel(self,coeffs,method='BFGS',maxiter=10,gtol=1e-5,every=1):
        from scipy.optimize import minimize
        self.ng = 0
        self.ne = 0
        self.niter = 0
        self.every = every
        options = {'maxiter':maxiter,'gtol':gtol}
        results = minimize(fun=self.grad,jac=True,method=method,x0=np.sqrt(coeffs),
                           callback=self.callback,options=options)
        results['x'] = np.square(results['x'])
        return results
class OptimizeVariances:
    def __init__(self,coeffs,centers,points,density,weights,normalize=False):
        self.coeffs = coeffs
        self.centers = centers
        self.points = points
        self.density = density
        self.weights = weights
        self.normalize = normalize

        self.nbasis,self.ndim = centers.shape

        self.nf = 0
        self.ng = 0
        self.niter = 0
    def loss(self,x):
        basis,_,_ = batched_basis_fxns(self.centers,x,self.points,
                                       normalize=self.normalize)
        self.f = loss(self.coeffs,basis,self.density,self.weights)
        self.nf += 1
        return self.f
    def grad(self,x):
        basis,normsq,_ = batched_basis_fxns(self.centers,x,self.points,
                                            normalize=self.normalize)
        _err = err(self.coeffs,basis,self.density)
        f = np.dot(self.weights,np.square(_err))

        g = - 2.*normsq*x.reshape(1,self.nbasis)
        if self.normalize:
            g += self.ndim*np.reciprocal(x).reshape(1,self.nbasis)
        g = -2.*np.dot(self.weights*_err,g*basis) * self.coeffs

        self.ng += 1
        self.f = f
        self.g = g
        return f,g
    def callback(self,x):
        self.niter += 1
        if self.niter % self.every==0:
            print(f'niter={self.niter},loss={self.f},g={np.linalg.norm(self.g)}')
            print('variance=',x)
        return
    def kernel(self,variances,method='BFGS',maxiter=10,gtol=1e-5,every=1):
        from scipy.optimize import minimize
        self.ng = 0
        self.ne = 0
        self.niter = 0
        self.every = every
        options = {'maxiter':maxiter,'gtol':gtol}
        return minimize(fun=self.grad,jac=True,method=method,x0=variances,
                        callback=self.callback,options=options)
class OptimizeCenters:
    def __init__(self,coeffs,variances,points,density,weights,normalize=False):
        self.coeffs = coeffs
        self.variances = variances
        self.points = points
        self.density = density
        self.weights = weights
        self.normalize = normalize

        self.nbasis = len(coeffs)
        _,self.ndim = points.shape

        self.nf = 0
        self.ng = 0
        self.niter = 0
    def loss(self,x):
        centers = x.reshape(self.nbasis,self.ndim)
        basis,_,_ = batched_basis_fxns(centers,self.variances,self.points,
                                       normalize=self.normalize)
        self.f = loss(self.coeffs,basis,self.density,self.weights)
        self.nf += 1
        return self.f
    def grad(self,x):
        centers = x.reshape(self.nbasis,self.ndim)
        basis,_,disp = batched_basis_fxns(centers,self.variances,self.points,
                                          normalize=self.normalize)
        _err = err(self.coeffs,basis,self.density)
        f = np.dot(self.weights,np.square(_err))
        g = np.einsum('r,ri,i,riv->iv',self.weights*_err,basis,
                      -4.*self.coeffs*np.square(self.variances),disp)
        self.ng += 1
        self.f = f
        self.g = g
        return f,g.reshape(-1)
    def callback(self,x):
        self.niter += 1
        if self.niter % self.every==0:
            print(f'niter={self.niter},loss={self.f},g={np.linalg.norm(self.g)}')
            print(x.reshape(self.nbasis,self.ndim).T)
        return
    def kernel(self,centers,method='L-BFGS-B',maxiter=10,gtol=1e-5,every=1):
        from scipy.optimize import minimize
        self.ng = 0
        self.ne = 0
        self.niter = 0
        self.every = every
        options = {'maxiter':maxiter,'gtol':gtol}

        xmax = np.amax(self.points)
        bounds = [(0.,xmax)] * (self.nbasis*self.ndim)
        results = minimize(fun=self.grad,jac=True,method=method,
                           x0=centers.reshape(-1),bounds=bounds,
                           callback=self.callback,options=options)
        results['x'] = results['x'].reshape(self.nbasis,self.ndim)
        return results
def optimize(points,density,weights,centers,variances,coeffs,normalize=False,
             seq=['coeffs','variances','centers'],macroiter=10,atol=1e-5,
             microiter=100,gtol=1e-5):
    objects = {'coeffs':coeffs,'variances':variances,'centers':centers}
    errs = dict() 
    for it in range(macroiter):
        for key in seq:
            if key == 'coeffs':
                opt = OptimizeCoeffs(objects['centers'],objects['variances'],
                          points,density,weights,normalize=normalize)
            elif key == 'variances':
                opt = OptimizeVariances(objects['coeffs'],objects['centers'],
                          points,density,weights,normalize=normalize)
            else:
                opt = OptimizeCenters(objects['coeffs'],objects['variances'],
                          points,density,weights,normalize=normalize)
            results = opt.kernel(objects[key],maxiter=microiter,every=microiter,
                                 gtol=gtol)
            errs[key] = np.linalg.norm(results['x'] - objects[key])
            objects[key] = results['x']
            print(f'\t{key} loss={opt.f},g={np.linalg.norm(opt.g)}')
        print(f'macroiter={it},errs=',errs)
        if sum(list(errs.values()))<atol:
            break
    return objects 
if __name__=='__main__':
    import functools
    xmax = 2.
    beta = 1.
    density_fxn = functools.partial(lennard_jones_density,beta=beta)

    ndim = 2
    ng = 50
    ranges = [(0.,xmax)] * ndim
    points,weights = get_quad(ranges,ng)
    density = np.array([density_fxn(point) for point in points])
    I = np.dot(weights,density)
    print(f'ng={ng},I={I}')

    nbasis = 3
    normalize = False 
    centers,variances,coeffs = initialize(xmax,ng,nbasis,ndim,density_fxn,
                                          normalize=normalize)
    density /= I
    if normalize:
        coeffs /= I
        basis,_,_ = batched_basis_fxns(centers,variances,points,normalize=normalize)
        print('check sum coeffs=',np.fabs(np.sum(coeffs)-1.))
        print('check integration=',np.fabs(np.linalg.multi_dot([weights,basis,coeffs])-1.))
    print('coeffs=',coeffs)

    from scipy.optimize import optimize
    epsilon = 1e-6
    opt = OptimizeCoeffs(centers,variances,points,density,weights,
                         normalize=normalize)
    x = np.random.rand(len(coeffs))
    f,g = opt.grad(x)
    sf = optimize._prepare_scalar_function(
         opt.loss,x0=x,jac=None,epsilon=epsilon,
         finite_diff_rel_step=epsilon) 
    g_ = sf.grad(x)
    gnorm = np.linalg.norm(g_)
    print(f'g={gnorm},err={np.linalg.norm(g_-g)/gnorm}')

    opt = OptimizeVariances(coeffs,centers,points,density,weights,
                            normalize=normalize)
    x = np.random.rand(len(coeffs))
    f,g = opt.grad(x)
    sf = optimize._prepare_scalar_function(
         opt.loss,x0=x,jac=None,epsilon=epsilon,
         finite_diff_rel_step=epsilon) 
    g_ = sf.grad(x)
    gnorm = np.linalg.norm(g_)
    print(f'g={gnorm},err={np.linalg.norm(g_-g)/gnorm}')

    opt = OptimizeCenters(coeffs,variances,points,density,weights,
                          normalize=normalize)
    x = np.random.rand(len(coeffs)*ndim)
    f,g = opt.grad(x)
    sf = optimize._prepare_scalar_function(
         opt.loss,x0=x,jac=None,epsilon=epsilon,
         finite_diff_rel_step=epsilon) 
    g_ = sf.grad(x)
    gnorm = np.linalg.norm(g_)
    print(f'g={gnorm},err={np.linalg.norm(g_-g)/gnorm}')
