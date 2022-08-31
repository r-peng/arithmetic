import numpy as np
import itertools,scipy
np.set_printoptions(suppress=True,precision=6,linewidth=1000)
class Scheme1:
    def __init__(self,rs,nbasis,target):
        self.nr,self.ndim = rs.shape
        self.nbasis = nbasis
        self.rs = rs
        self.target = target
        self.triu_idxs = np.triu_indices(nbasis)

        self.ng = 0
        self.nf = 0
        self.niter = 0
    def param2vec(self,a,b):
        return np.concatenate([a[self.triu_idxs],b.flatten()])
    def vec2param(self,x):
        x,b = np.split(x,[len(self.triu_idxs[0])])
        a = np.zeros((self.nbasis,)*2)
        for ix,(i,j) in enumerate(zip(self.triu_idxs[0],self.triu_idxs[1])):
            a[i,j] = a[j,i] = x[ix]
        return a,np.reshape(b,(self.nbasis,self.ndim+1))
    def loss(self,x):
        a,b = self.vec2param(x)
        vec = np.reshape(self.rs,(self.nr,1,self.ndim))\
            - np.reshape(b[:,:self.ndim],(1,self.nbasis,self.ndim))
        normsq = - np.einsum('xiv,xiv->xi',vec,vec)
        b0 = np.broadcast_to(b[:,-1],(self.nr,self.nbasis))
 
        bf = np.exp(normsq*np.reshape(b[:,-1],(1,self.nbasis)))
        err = self.target - np.einsum('xi,yj,ij->xy',bf,bf,a)
        f = np.sum(np.square(err))

        self.nf += 1
        self.f = f
        return f
    def grad(self,x):
        a,b = self.vec2param(x)
        vec = np.reshape(self.rs,(self.nr,1,self.ndim))\
            - np.reshape(b[:,:self.ndim],(1,self.nbasis,self.ndim))
        normsq = - np.einsum('xiv,xiv->xi',vec,vec)
        b0 = np.broadcast_to(b[:,-1],(self.nr,self.nbasis))
 
        bf = np.exp(normsq*np.reshape(b[:,-1],(1,self.nbasis)))
        dbf = np.concatenate((vec*np.reshape(2.*b[:,-1],(1,self.nbasis,1)),
                              np.reshape(normsq,(self.nr,self.nbasis,1))),axis=-1)
        dbf = dbf * np.reshape(bf,(self.nr,self.nbasis,1))

        err = self.target - np.einsum('xi,yj,ij->xy',bf,bf,a)
        tmp = np.einsum('xy,xi->yi',-2.*err,bf)
        da = np.dot(bf.T,tmp)
        da[np.triu_indices(self.nbasis,k=1)] *= 2.
        tmp = np.dot(tmp,a)
        db = 2.*np.einsum('xi,xiv->iv',tmp,dbf)

        f = np.sum(np.square(err))
        g = self.param2vec(da,db)

        self.ng += 1
        self.f = f
        self.g = g
        self.a = a
        self.b = b
        return f,g
    def callback(self,x):
        self.niter += 1
        if self.niter % self.every==0:
            print(f'niter={self.niter},loss={self.f},g={np.linalg.norm(self.g)}')
            print(self.a)
            print(self.b)
        return
    def kernel(self,a0,b0,method='BFGS',maxiter=1000,gtol=1e-5,every=100):
        from scipy.optimize import minimize
        self.ng = 0
        self.ne = 0
        self.niter = 0
        self.every = every
        options = {'maxiter':maxiter,'gtol':gtol}
        x0 = self.param2vec(a0,b0)
        minimize(fun=self.grad,jac=True,method=method,x0=x0,
                 callback=self.callback,options=options)
        return
if __name__=='__main__':
    xmax = 2.
    dx = .2
    xs = np.arange(0.,xmax,dx)

    ndim = 2
    ng = len(xs)
    idxs = list(itertools.product(range(ng),repeat=ndim))
    rs = np.zeros((ng**ndim,ndim))
    for i in range(ng**ndim):
        rs[i,:] = np.array([xs[ix] for ix in idxs[i]])

    beta = 1.
    target = np.zeros((ng**ndim,ng**ndim))
    for i in range(ng**ndim):
        for j in range(i+1,ng**ndim):
            r = np.linalg.norm(rs[i]-rs[j])
            target[i,j] = target[j,i] = np.exp(-beta*(1./r**12.-1./r**6))

    nbasis = 5
    opt = Scheme1(rs,nbasis,target)
    a0 = np.ones((nbasis,)*2)
    b0 = np.concatenate((rs[-nbasis:,:],np.ones((nbasis,1))),axis=-1)
    x = opt.param2vec(a0,b0)
    a,b = opt.vec2param(x)
    print(f'check a={np.linalg.norm(a0-a)/np.linalg.norm(a0)},b={np.linalg.norm(b0-b)/np.linalg.norm(b0)}')

    #test_grad = True
    test_grad = False 
    if test_grad:
        f,g = opt.grad(x)
        epsilon = 1e-6
        from scipy.optimize import optimize
        sf = optimize._prepare_scalar_function(
             opt.loss,x0=x,jac=None,epsilon=epsilon,
             finite_diff_rel_step=epsilon) 
        g_ = sf.grad(x)
        gnorm = np.linalg.norm(g_)
        print(f'epsilon={epsilon},g={gnorm},err={np.linalg.norm(g_-g)/gnorm}',)

    opt.kernel(a0,b0)
