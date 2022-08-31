import numpy as np
import itertools
import quimb.tensor as qtn
np.set_printoptions(suppress=True,precision=6,linewidth=1000)
def cheb(Nnodes,Ndim,xmin,xmax,fxn,Lobatto=False):
    alpha,beta = (xmax-xmin)/2.,(xmin+xmax)/2.
    if Lobatto:
        c = np.ones(Nnodes+1)
        c[0] = c[Nnodes] = 2.
        theta = np.array([k*np.pi/Nnodes for k in range(Nnodes+1)])
    else:
        c = np.ones(Nnodes+1) * 2.
        c[0] = 1.
        theta = np.array([(2.*k+1.)/(2*(Nnodes+1.))*np.pi for k in range(Nnodes+1)])

    z = np.cos(theta)
    x = alpha * z + beta
 
    f = np.zeros((Nnodes+1,)*Ndim)
    idxs = list(itertools.product(range(Nnodes+1),repeat=Ndim))
    for idx in idxs:
        f[idx] = fxn(*[x[i] for i in idx])
        if Lobatto:
            const = np.prod([c[i] for i in idx])
            f[idx] /= const

    t = np.zeros((Nnodes+1,)*2)
    for n in range(Nnodes+1):
        t[n,:] = np.cos(n*theta)
    const = 2./Nnodes if Lobatto else 1./(Nnodes+1.)
    t *= const

    tn = qtn.TensorNetwork([])
    for i in range(Ndim):
        tn.add_tensor(qtn.Tensor(data=t.copy(),inds=(f'n{i}',f'k{i}')))
    tn.add_tensor(qtn.Tensor(data=f,inds=[f'k{i}' for i in range(Ndim)]))
    a = tn.contract(output_inds=[f'n{i}' for i in range(Ndim)]).data
    for idx in idxs:
        const = np.prod([c[i] for i in idx])
        if Lobatto:
            const = 1./const
        a[idx] *= const

    def _t(xs):
        ng = len(xs)
        t = np.zeros((Nnodes+1,ng))
        for k in range(Nnodes+1):
            coeff = np.zeros(Nnodes+1)
            coeff[k] = 1.
            pol = np.polynomial.chebyshev.Chebyshev(coeff)
            for g in range(ng):
                t[k,g] = pol((xs[g]-beta)/alpha)
        return t 
    return a, _t
if __name__=='__main__':
    xmin = 0.
    xmax = 2.
    dx = .5
    xs = np.arange(xmin,xmax,dx)
    ng = len(xs)
    Nnodes = 5
    def fxn(x1,y1,x2,y2):
        rsq = (x1-x2)**2 + (y1-y2)**2
        if rsq < 1e-2:
            return 0.
        else:
            return np.exp(-(1./rsq**6-1./rsq**3))
    f = np.array([[[[fxn(x1,y1,x2,y2) for x1 in xs] for y1 in xs] for x2 in xs] for y2 in xs])
    a, _t = cheb(Nnodes,4,xmin,xmax,fxn,Lobatto=True)
    t = _t(xs)
    f_ = np.einsum('klmn,kx,ly,mX,nY->xyXY',a,t,t,t,t)
    f = f.reshape((ng**2,)*2)
    f_ = f_.reshape((ng**2,)*2)
    a = a.reshape(((Nnodes+1)**2,)*2)
    print(np.linalg.norm(f-f_)/np.linalg.norm(f))
    w,v = np.linalg.eigh(f)
    print(len(w))
    w,v = np.linalg.eigh(a)
    print(len(w[np.fabs(w)>1e-3]))
