import numpy as np
import quimb.tensor as qtn
import arithmetic.utils as utils
import functools,scipy
ADD = np.zeros((2,)*3)
ADD[0,0,0] = ADD[1,0,1] = ADD[0,1,1] = 1.0
CP2 = np.zeros((2,)*3)
CP2[0,0,0] = CP2[1,1,1] = 1.0
def get_energy(xs,A,B=None,tr=None,from_which=None,**compress_opts):
    N,d = xs.shape
    arrays = []
    # 1-variable terms
    row = [] 
    for i in range(N):
        data = np.ones((2,d))
        xsq = np.square(xs[i,:])
        data[1,:] = xsq*A[i,i]
        if B is not None:
            xqd = np.square(xsq)
            data[1,:] += xqd*B[i,i,i,i]
        if i>0:
            data = np.einsum('ip,ijk->jkp',data,ADD)
        row.append(data)
    row.append(np.eye(2))
    arrays.append(row)
    # 2-variable terms
    row_map = dict()
    dummy = np.einsum('ij,pq->ijpq',np.eye(2),np.eye(d))
    CP = np.zeros((d,)*3)
    for i in range(d):
        CP[i,i,i] = 1.0
    def get_row(fac,inds,vecs):
        row = []
        i,xi = inds[0],vecs[0]
        for k in range(i):
            newshape = (1,d,d) if k==0 else (1,1,d,d)
            row.append(np.reshape(np.eye(d),newshape))
        # xi
        data = np.ones((2,d))
        data[1,:] = xi*fac
        data = np.einsum('ip,pqr->iqr',data,CP)
        if i>0:
            data = np.reshape(data,(1,2,d,d))
        row.append(data)
         
        for vind in range(1,len(inds)):
            i,j = inds[vind-1],inds[vind]
            xj = vecs[vind]
            for k in range(i+1,j):
                row.append(dummy)
            # xj
            data = np.ones((2,d))
            data[1,:] = xj
            row.append(np.einsum('ip,ijk,pqr->jkqr',data,CP2,CP))
        j = inds[-1]
        for k in range(j+1,N):
            row.append(dummy)
        row.append(ADD.transpose(0,2,1))
        return row
    for i in range(N):
        for j in range(i+1,N):
            row_map[i,j] = []
            fac = A[i,j]+A[j,i]
            xi = xs[i,:].copy()
            xj = xs[j,:].copy()
#            arrays.append(get_row(fac,[i,j],[xi,xj]))
            row_map[i,j].append(get_row(fac,[i,j],[xi,xj]))
            if B is not None:
                #xi^3xj
                fac = get_fac_iiij(B,i,j)
                xi = np.square(xs[i,:])
                xi = np.multiply(xi,xs[i,:])
                xj = xs[j,:].copy()
#                arrays.append(get_row(fac,[i,j],[xi,xj]))
                row_map[i,j].append(get_row(fac,[i,j],[xi,xj]))
                #xixj^3
                fac = get_fac_iiij(B,j,i)
                xi = xs[i,:].copy()
                xj = np.square(xs[j,:])
                xj = np.multiply(xj,xs[j,:])
#                arrays.append(get_row(fac,[i,j],[xi,xj]))
                row_map[i,j].append(get_row(fac,[i,j],[xi,xj]))
                #xi^2xj^2
                fac = get_fac_iijj(B,i,j)
                xi = np.square(xs[i,:])
                xj = np.square(xs[j,:])
#                arrays.append(get_row(fac,[i,j],[xi,xj]))
                row_map[i,j].append(get_row(fac,[i,j],[xi,xj]))
    if B is not None:
        for i in range(N):
            for j in range(i+1,N):
                for k in range(j+1,N):
                    row_map[i,j,k] = []
                    # xi^2xjxk
                    fac = get_fac_iijk(B,i,j,k)
                    xi = np.square(xs[i,:])
                    xj = xs[j,:].copy()
                    xk = xs[k,:].copy()
#                    arrays.append(get_row(fac,[i,j,k],[xi,xj,xk]))
                    row_map[i,j,k].append(get_row(fac,[i,j,k],[xi,xj,xk]))
                    # xixj^2xk
                    fac = get_fac_iijk(B,j,i,k)
                    xi = xs[i,:].copy()
                    xj = np.square(xs[j,:])
                    xk = xs[k,:].copy()
#                    arrays.append(get_row(fac,[i,j,k],[xi,xj,xk]))
                    row_map[i,j,k].append(get_row(fac,[i,j,k],[xi,xj,xk]))
                    # xixjxk^2
                    fac = get_fac_iijk(B,k,i,j)
                    xi = xs[i,:].copy()
                    xj = xs[j,:].copy()
                    xk = np.square(xs[k,:])
#                    arrays.append(get_row(fac,[i,j,k],[xi,xj,xk]))
                    row_map[i,j,k].append(get_row(fac,[i,j,k],[xi,xj,xk]))
        for i in range(N):
            for j in range(i+1,N):
                for k in range(j+1,N):
                    for l in range(k+1,N):
                        fac = get_fac_ijkl(B,i,j,k,l)
                        xi = xs[i,:].copy()
                        xj = xs[j,:].copy()
                        xk = xs[k,:].copy()
                        xl = xs[l,:].copy()
#                        arrays.append(get_row(fac,[i,j,k,l],[xi,xj,xk,xl]))
                        row_map[i,j,k,l] = [get_row(fac,[i,j,k,l],[xi,xj,xk,xl])]
    for i in range(N,-1,-1):
        for j in range(N,i,-1):
            for k in range(N,j,-1):
                for l in range(N,k,-1):
                    keys = [(k,l),(j,l),(j,k),(i,l),(i,k),(i,j)]
                    keys += [(j,k,l),(i,k,l),(i,j,l),(i,j,k)]
                    keys += [(i,j,k,l)]
                    for key in keys:
                        if key in row_map:
                            for row in row_map[key]:
                                arrays.append(row)
                            row_map.pop(key)
    peps = make_peps_with_legs(arrays)
    if tr is None:
        return contract_from_bottom(peps,**compress_opts)
    else:
        peps = trace_open(peps,tr)
        return contract(peps,from_which=from_which,**compress_opts) 
def get_pol(E,coeff,tr=None):
    tmp = np.einsum('ijk,klm->ijlm',CP2,ADD)
    N = len(E)-1
    d = E[0].shape[-1]
    M = len(coeff)-1
    CP = np.zeros((d,)*3)
    for i in range(d):
        CP[i,i,i] = 1.0

    arrays = []
    for k in range(M,0,-1):
        # xk/k
        row = [arr.copy() for arr in E]
        row[-1] = np.einsum('li,i->li',row[-1],np.array([1.0,1.0/k]))
        mps = qtn.MatrixProductState(row,shape='lrp')
        mps.equalize_norms_()
        row = [mps.tensor_map[i].data for i in range(mps.L)] 

        data = np.einsum('ijlm,l->ijm',tmp,np.array([1.0,coeff[k-1]]))
        row[-1] = np.einsum('...i,ijm->...mj',row[-1],data)
        if k==M:
            row[-1] = np.einsum('...mj,j->...m',row[-1],np.array([1.0,coeff[-1]]))
        else:
            for i in range(N):
                row[i] = np.einsum('...p,pqr->...qr',row[i],CP)
        arrays.append(row)
    peps = make_peps_with_legs(arrays)
    if tr is not None:
        peps = trace_open(peps,tr) 
    return peps
def get_coeff(M,a=0.0):
    # coeff for exp(x) w/out factorial
    if abs(a)<1e-10:
        coeff = np.ones(M+1)
    else:
        log_fac = np.array([0]+[np.log10(i) for i in range(1,M+1)])
        log_fac = np.cumsum(log_fac)
        log_inn = [l*np.log10(abs(a))-log_fac[l] for l in range(M+1)]
        sign = [(-1)**l for l in range(M+1)] if a>0.0 else np.ones(M+1)
        coeff = np.zeros(M+1)
        for k in range(M+1):
            coeff[k] = sum([sign[l]*10**log_inn[l] for l in range(M-k+1)])
    return np.exp(a)*coeff
def get_coeff_cheb(xmin,xmax,n,m):
    a = 2.0/(xmax-xmin)
    b = (xmin+xmax)/(xmin-xmax)
    def f(x):
        return np.exp((x-b)/a)
    assert m>=n+1
    r = np.array([-np.cos((2.0*k-1.0)/(2.0*m)*np.pi) for k in range(1,m+1)])
    y = np.array([f(rk) for rk in x])
    T = np.zeros((n+1,m))
    for i in range(n+1):
        c = np.zeros(i+1)
        c[-1] = 1.0
        T[i,:] = np.polynomial.chebyshev.chebval(r,c)
    a = np.array([np.dot(y,T[i,:])/np.dot(T[i,:],T[i,:]) for i in range(n+1)])
    print('cheb:',a)
    return np.polynomial.chebyshev.cheb2poly(a)

permute_1d = utils.permute_1d
contract_1d = utils.contract_1d
make_peps_with_legs = utils.make_peps_with_legs
trace_open = utils.trace_open
contract_from_bottom = utils.contract_from_bottom
contract = utils.contract

get_fac_ijkl = utils.get_fac_ijkl
get_fac_iijk = utils.get_fac_iijk
get_fac_iijj = utils.get_fac_iijj
get_fac_iiij = utils.get_fac_iiij

quad = utils.quad
exact = utils.exact 
####################### saved fxns #######################
def _get_cheb_coeff(M):
    def w(x):
        return 2.0/(np.pi*np.sqrt(1-x**2))
    c = np.zeros(M+1)
    for n in range(M+1):
        c_ = np.zeros(n+1)
        c_[-1] = 1.0
        Tn = functools.partial(np.polynomial.chebyshev.chebval,c=c_) 
        def f(x):
            return np.exp(x)*Tn(x)*w(x)
        c[n] = scipy.integrate.quad(f,-1.0,1.0)[0]
    c[0] /= 2.0
    print('cheb:',c)
    return np.polynomial.chebyshev.cheb2poly(c)
def get_scaled_powers(E,tr,M,from_which=None,**compress_opts):
    N,d = tr.shape
    CP = np.zeros((d,)*3)
    for i in range(d):
        CP[(i,)*3] = 1.0

    row_map = dict()
    for k in range(1,M+1):
        arrays = [arr.copy() for arr in E]
        arrays[-1] = np.einsum('li,i->li',arrays[-1],np.array([1.0,1.0/k]))
        mps = qtn.MatrixProductState(arrays,shape='lrp')
        mps.equalize_norms_()
        row_map[k] = [mps.tensor_map[i].data for i in range(mps.L)] 

    orders = np.zeros(M+1)
    orders[0] = np.prod(np.sum(tr,axis=1)) 
    orders[1] = contract_1d(E,tr)
    for k in range(2,M+1):
        arrays = []
        for j in range(1,k+1):
            row = [arr.copy() for arr in row_map[j]]
            if j>1:
                for i in range(N+1):
                    data = CP2 if i==N else CP
                    row[i] = np.einsum('...p,pqr->...qr',row[i],data)
            arrays.append(row)
        peps = make_peps_with_legs(arrays)
        peps = trace_open(peps,tr)
        orders[k] = contract(peps,from_which=from_which,**compress_opts)
    return orders
def _get_coeffs(a,M):
    # M degree taylor approximation of exp(x) centered at a
    coeff = []
    fac = [math.factorial(k) for k in range(M+1)]
    for k in range(M+1):
        out = 0.0
        for l in range(M-k+1):
            out += (-a)**l/fac[l]
        print(k,out)
        coeff.append(np.exp(a)*out/fac[k])
    return coeff
