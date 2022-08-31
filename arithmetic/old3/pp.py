import numpy as np
import quimb.tensor as qtn
import scipy
np.set_printoptions(suppress=True,linewidth=1000,precision=6)
class Optimize:
    # class for minimize f=|A-B(P)|/<A|A> given fixed rank 
    def __init__(self,N,U,P0,order='C',mode='norm'):
        self.N = N
        self.U = U
        self.P = P0
        self.order = order
        self.mode = mode
        self.ng = 0
        self.nf = 0
        self.it = 0

        UU = np.dot(U,U.T) 
        fac = np.linalg.norm(UU) if self.mode == 'norm' else np.amax(UU)
        exp = np.log10(fac)
        self.eAA = np.log10(np.sum(np.power(UU/fac,np.ones_like(UU)*N))) + N*exp
        self.const = 1.
    def loss(self,x):
        P = np.reshape(x,self.P.shape,order=self.order)

        UP = np.dot(self.U,P.T)
        fac = np.linalg.norm(UP) if self.mode == 'norm' else np.amax(np.fabs(UP))
        exp = np.log10(fac)
        fAB,eAB = np.sum(np.power(UP/fac,np.ones_like(UP)*self.N)),self.N*exp

        PP = np.dot(P,P.T)
        fac = np.linalg.norm(PP) if self.mode == 'norm' else np.amax(np.fabs(PP))
        exp = np.log10(fac)
        fBB,eBB = np.sum(np.power(PP/fac,np.ones_like(PP)*self.N)),self.N*exp

        self.nf += 1
        f = 1. - 2.*fAB*10.**(eAB-self.eAA) + fBB*10.**(eBB-self.eAA)
        return f / self.const
    def grad(self,x): 
        P = np.reshape(x,self.P.shape,order=self.order)

        UP = np.dot(self.U,P.T)
        fac = np.linalg.norm(UP) if self.mode == 'norm' else np.amax(np.fabs(UP))
        exp = np.log10(fac)
        gAB = np.dot(np.power(UP/fac,np.ones_like(UP)*(self.N-1)).T,self.U)
        fAB,eAB = np.einsum('al,al->',gAB,P),(self.N-1)*exp

        PP = np.dot(P,P.T)
        fac = np.linalg.norm(PP) if self.mode == 'norm' else np.amax(np.fabs(PP))
        exp = np.log10(fac)
        gBB = np.dot(np.power(PP/fac,np.ones_like(PP)*(self.N-1)),P)
        fBB,eBB = np.einsum('al,al->',gBB,P),(self.N-1)*exp

        f = 1. - 2.*fAB*10.**(eAB-self.eAA) + fBB*10.**(eBB-self.eAA)
        g = 2.*self.N*(- gAB*10.**(eAB-self.eAA) + gBB*10.**(eBB-self.eAA))
        g = g.flatten(order=self.order)

        self.f = f
        self.g = g
        self.ng += 1
        return f/self.const, g/self.const
    def callback(self,x):
        self.it += 1
        self.P = np.reshape(x,self.P.shape,order=self.order)
        print(f'iter={self.it},ngrad={self.ng},loss={self.f},g={np.linalg.norm(self.g)}')
    def kernel(self,method='BFGS',maxiter=200,gtol=1e-5):
        self.ng = 0
        self.ne = 0
        self.niter = 0
        options = {'maxiter':maxiter,'gtol':gtol}
        x0 = self.P.flatten(order=self.order)
        scipy.optimize.minimize(fun=self.grad,jac=True,
                 method=method,x0=x0,
                 callback=self.callback,options=options)
        return
def get_hypergraph(T,N):
    tn = qtn.TensorNetwork([])
    for i in range(N):
        for j in range(i+1,N):
            inds = f'x{i}',f'x{j}'
            tn.add_tensor(qtn.Tensor(data=T.copy(),inds=inds,tags=inds))
    return tn
def resolve(tn,N,remove_lower=True):
    from .gaussian import resolve as _resolve
    return _resolve(tn,N,remove_lower=remove_lower)
