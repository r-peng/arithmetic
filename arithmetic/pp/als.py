import numpy as np
import scipy.linalg
import itertools
def compute_density(ng,xmax,ndim,density_fxn):
    idxs = list(itertools.product(range(ng),repeat=ndim))
    x,w = np.polynomial.legendre.leggauss(ng)
    x = x*xmax/2.+xmax/2.
    w = w*xmax/2.

    T = np.zeros((len(idxs),)*2)
    for i in range(len(idxs)):
        ri = np.array([x[ix] for ix in idxs[i]])
        for j in range(i,len(idxs)):
            rj = np.array([x[ix] for ix in idxs[j]])
            T[i,j] = T[j,i] = density_fxn(ri-rj)
    D,U = np.linalg.eigh(T)
    #U = np.einsum('pk,k->pk',U,np.sqrt(np.fabs(D)))
    #D = np.sign(D)
    U = np.einsum('pk,k->pk',U,D)
    D = np.reciprocal(D)
    assert np.linalg.norm(T-np.einsum('pk,qk,k->pq',U,U,D))/np.linalg.norm(T)<1e-10

    idxs = list(range(U.shape[0]))
    idxs.sort(key=lambda i:np.dot(U[i,:],U[i,:]))
    idxs = np.array(idxs)
    U = U[idxs,:]
    return U,D,w
def als(U,mps=None,bdim=None,N=None,macroiter=1000,atol=1e-5,solver='inv'):
    if mps is None:
        CP = np.zeros((bdim,)*3)
        for i in range(bdim):
            CP[i,i,i] = 1.
        mps = [None] * N
        for i in range(N):
            if i==0 or i==N-1:
                mps[i] = U[-bdim:,:].copy()
            else:
                mps[i] = np.einsum('pk,pqr->qrk',U[-bdim:,:],CP)
    else:
        bdim = mps[0].shape[0]
    N,kdim = len(mps),U.shape[-1]

    UU = np.dot(U,U.T)
    fac = np.linalg.norm(UU)
    BB = np.log10(np.sum((UU/fac)**N)) + N*np.log10(fac)

    AA_larr = [None] * N
    AA_rarr = [None] * N
    AB_larr = [None] * N
    AB_rarr = [None] * N
    AA_lexp = np.zeros(N) 
    AA_rexp = np.zeros(N) 
    AB_lexp = np.zeros(N) 
    AB_rexp = np.zeros(N) 

    def update_renvs(i):
        if i==N-1:
            return

        if i==N-2:
            arr = np.dot(mps[i+1],mps[i+1].T)
        else:
            arr = np.einsum('rR,lrk,LRk->lL',AA_rarr[i+1],mps[i+1],mps[i+1])
        fac = np.linalg.norm(arr)
        AA_rarr[i] = arr/fac
        AA_rexp[i] = np.log10(fac) + AA_rexp[i+1]

        if i==N-2:
            arr = np.dot(mps[i+1],U.T)
        else:
            arr = np.einsum('rP,lrk,Pk->lP',AB_rarr[i+1],mps[i+1],U)
        fac = np.linalg.norm(arr)
        AB_rarr[i] = arr/fac
        AB_rexp[i] = np.log10(fac) + AB_rexp[i+1]
        return
    def update_lenvs(i):
        if i==0:
            return

        if i==1:
            arr = np.dot(mps[i-1],mps[i-1].T)
        else:
            arr = np.einsum('lL,lrk,LRk->rR',AA_larr[i-1],mps[i-1],mps[i-1])
        fac = np.linalg.norm(arr)
        AA_larr[i] = arr/fac
        AA_lexp[i] = np.log10(fac) + AA_lexp[i-1]

        if i==1:
            arr = np.dot(mps[i-1],U.T)
        else:
            arr = np.einsum('lP,lrk,Pk->rP',AB_larr[i-1],mps[i-1],U)
        fac = np.linalg.norm(arr)
        AB_larr[i] = arr/fac
        AB_lexp[i] = np.log10(fac) + AB_lexp[i-1]
        return
    def update_mps(i):
        if i>0 and i<N-1:
            A = np.einsum('lL,rR->lrLR',AA_larr[i],AA_rarr[i])
            A = A.reshape((bdim**2,)*2)
            y = np.einsum('lP,rP,Pk->lrk',AB_larr[i],AB_rarr[i],U)
            y = y.reshape(bdim**2,kdim)
        else:
            A = AA_rarr[i] if i==0 else AA_larr[i]
            y = np.dot(AB_rarr[i],U) if i==0 else np.dot(AB_larr[i],U)
        Aexp, yexp = AA_lexp[i]+AA_rexp[i], AB_lexp[i] + AB_rexp[i] 
        if solver=='inv':
            x = np.dot(np.linalg.inv(A),y)
        else:
            x = np.stack([scipy.linalg.solve(A,y[:,k],assume_a='sym') \
                          for k in range(kdim)],axis=-1)
        x *= 10.**(yexp-Aexp)
        AA = np.log10(np.sum(A*np.dot(x,x.T))) + Aexp
        AB = np.log10(np.sum(x*y)) + yexp
        err = 10.**(AA-BB) + 1. - 2.*10.**(AB-BB)
        if i>0 and i<N-1:
            x = x.reshape(bdim,bdim,kdim)
        mps[i] = x
        return err 
    for i in range(N-1,-1,-1):
        update_renvs(i)
    err_old = 0.
    for it in range(macroiter):
        print('macroiter=',it)
        for i in range(N): # left sweep
            err = update_mps(i)
            print(f'\ti={i},err={err}')
            if i<N-1:
                update_lenvs(i+1)
            if i>0:
                AA_rarr[i-1] = None
                AB_rarr[i-1] = None 
                AA_rexp[i-1] = 0.
                AB_rexp[i-1] = 0.
        if np.fabs(err-err_old) < atol:
            break
        err_old = err

        for i in range(N-1,-1,-1): # right sweep
            err = update_mps(i)
            print(f'\ti={i},err={err}')
            if i>0:
                update_renvs(i-1)
            if i<N-1:
                AA_larr[i+1] = None
                AB_larr[i+1] = None
                AA_lexp[i+1] = 0. 
                AB_lexp[i+1] = 0.
        if np.fabs(err-err_old) < atol:
            break
        err_old = err
    return mps
if __name__=='__main__':
    from arithmetic.pp.gauss import lennard_jones_density
    import functools 
    ng = 10
    xmax = 2.
    ndim = 2
    beta = 1.
    density_fxn = functools.partial(lennard_jones_density,beta=beta)
    U,sign,w = compute_density(ng,xmax,ndim,density_fxn)

    bdim = 20
    N = 6
    arrs = als(U,bdim=bdim,N=N,macroiter=10)
    import quimb.tensor as qtn
    mps1 = qtn.MatrixProductState(arrs,shape='lrp')
    CP = np.zeros((ng**ndim,)*3)
    for i in range(ng**ndim):
        CP[i,i,i] = 1.
    arrs = [None] * N
    for i in range(N):
        if i==0 or i==N-1:
            arrs[i] = U.copy()
        else:
            arrs[i] = np.einsum('pk,pqr->qrk',U,CP)
    mps2 = qtn.MatrixProductState(arrs,shape='lrp')
    AA = mps1.copy()
    AA.add_tensor_network(mps1,check_collisions=True)
    AB = mps1.copy()
    AB.add_tensor_network(mps2,check_collisions=True)
    BB = mps2.copy()
    BB.add_tensor_network(mps2,check_collisions=True)
    def contract(tn):
        for i in range(N-1):
            tn.contract_tags((tn.site_tag(i),tn.site_tag(i+1)),inplace=True)
        tn.strip_exponent(tn[i],1.)
        return np.log10(tn.contract()) + tn.exponent
    AA = contract(AA)
    AB = contract(AB)
    BB = contract(BB)
    print('rel err=',10.**(AA-BB)+1.-2.*10.**(AB-BB))
    
