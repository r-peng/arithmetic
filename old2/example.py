import numpy as np
import math
import arithmetic
np.set_printoptions(suppress=True)

def _gauss1(mu,sigma,xs,p,thresh=1e-12,bdim=20,max_iter=50):
    # mu: list, centers for each variable
    # sigma: positive definite covariance matrix
    # xs: list of variables
    # p: int, highe degree of Taylor expansion
    assert len(mu)==sigma.shape[0]
    assert len(xs)==len(mu)
    sigma_inv = np.linalg.inv(sigma)
    print('det',np.linalg.det(sigma))
    print('inv',sigma_inv)
    print('sigma',sigma)
    ds = [len(xs[i]) for i in range(len(xs))]
    xs_ = []
    for i in range(len(mu)):
        b1 = [] if i==0 else arithmetic._const(ds[:i],1.0,0,thresh,bdim,max_iter)
        b2 = [] if i==len(mu)-1 else arithmetic._const(ds[i+1:],1.0,0,
                                                       thresh,bdim,max_iter)
        x = b1+[xs[i].reshape(1,ds[i],1)-mu[i]]+b2
        assert len(x)==len(mu)
        xs_.append(x)
    print('xs reformed')
    sxs = []
    for i in range(len(mu)):
        sx = arithmetic._multiply_const(xs_[0],sigma_inv[i,0],None,thresh,bdim,max_iter)
        for j in range(1,len(mu)):
            tmp = arithmetic._multiply_const(xs_[j],sigma_inv[i,j],None,
                                             thresh,bdim,max_iter)
            sx = arithmetic._add(sx,tmp,thresh,bdim,max_iter)
        sxs.append(sx)
    assert len(sxs)==len(mu)
    print('sxs formed')
    exponent = arithmetic._multiply(xs_[0],sxs[0],thresh,bdim,max_iter)
    for i in range(1,len(mu)):
        tmp = arithmetic._multiply(xs_[i],sxs[i],thresh,bdim,max_iter)
        exponent = arithmetic._add(tmp,exponent,thresh,bdim,max_iter)
    exponent = arithmetic._multiply_const(exponent,0.5,None,thresh,bdim,max_iter)
    print('exponent formed')
    ins = np.zeros(ng)
    ins[0] = 1.0
    ins = [ins for i in range(n)]
    out1 = arithmetic._contract(exponent,ins)
    print('exponent',out1)
    print('max exponent',np.amax(np.absolute(exponent[0].flatten())))
    print('max exponent',np.amax(np.absolute(exponent[1].flatten())))
    print('max exponent',np.amax(np.absolute(exponent[2].flatten())))

    coeff = [(-1)**i/math.factorial(i) for i in range(p+1)]
    out = arithmetic._poly2(exponent,coeff,thresh,bdim,max_iter)
    det = np.linalg.det(sigma)
    det = math.sqrt(det*(2*math.pi)**len(mu)) 
    return arithmetic._multiply_const(out,1.0/det,None,thresh,bdim,max_iter)
def _gauss2(mu,sigma,xs,p,thresh=1e-12,bdim=20,max_iter=50):
    assert len(mu)==sigma.shape[0]
    assert len(xs)==len(mu)
    sigma_inv = np.linalg.inv(sigma)
    ds = [len(xs[i]) for i in range(len(xs))]
    xs_ = []
    for i in range(len(mu)):
        b1 = [] if i==0 else arithmetic._const(ds[:i],1.0,0,thresh,bdim,max_iter)
        b2 = [] if i==len(mu)-1 else arithmetic._const(ds[i+1:],1.0,0,
                                                       thresh,bdim,max_iter)
        x = b1+[xs[i].reshape(1,ds[i],1)-mu[i]]+b2
        assert len(x)==len(mu)
        xs_.append(x)
    coeff = [(-1)**i/math.factorial(i) for i in range(p+1)]
    out = arithmetic._const(ds,1.0,0,thresh,bdim,max_iter)
    for i in range(len(mu)):
        for j in range(len(mu)):
            exponent = arithmetic._multiply(xs_[i],xs_[j],thresh,bdim,max_iter)
            exponent = arithmetic._multiply_const(exponent,0.5*sigma_inv[i,j],None,
                                                  thresh,bdim,max_iter)
            exp = arithmetic._poly2(exponent,coeff,thresh,bdim,max_iter)
            out = arithmetic._multiply(exp,out,thresh,bdim,max_iter)
    det = np.linalg.det(sigma)
    det = math.sqrt(det*(2*math.pi)**len(mu)) 
    return arithmetic._multiply_const(out,1.0/det,None,thresh,bdim,max_iter)
if __name__=='__main__':
    import scipy.linalg
    n = 3
    ng = 20
    xmin = -1.0
    xmax = 1.0
    mu = [0 for i in range(n)]
    p = 10
    thresh = 1e-12
    bdim = 50
    max_iter = 10
    sigma = np.random.rand(n)*10
    tmp = np.random.rand(n,n)
    tmp -= tmp.T
    tmp = scipy.linalg.expm(tmp)
    sigma = np.linalg.multi_dot([tmp,np.diag(sigma),tmp.T])
    w, v = np.linalg.eigh(sigma)
    print('eigval',w)

    dx = (xmax-xmin)/ng
    xs = [np.arange(xmin,xmax,dx) for i in range(n)]
    tn1 = _gauss1(mu,sigma,xs,p,thresh,bdim,max_iter)
    tn2 = _gauss2(mu,sigma,xs,p,thresh,bdim,max_iter)

    i = 0
    ins = np.zeros(ng)
    ins[i] = 1.0
    ins = [ins for i in range(n)]
    out1 = arithmetic._contract(tn1,ins)
    out2 = arithmetic._contract(tn2,ins)

    sigma_inv = np.linalg.inv(sigma)
    x = np.array([xs[j][i]-mu[j] for j in range(n)])
    exponent = -0.5*np.einsum('i,ij,j->',x,sigma_inv,x)
    print('exponent',exponent)
    out = np.exp(exponent)/math.sqrt(np.linalg.det(sigma)*(2*math.pi)**n)
    print('err1={},bdim={}'.format(abs(out-out1),arithmetic._get_bdim(tn1)))
    print('err2={},bdim={}'.format(abs(out-out2),arithmetic._get_bdim(tn2)))
