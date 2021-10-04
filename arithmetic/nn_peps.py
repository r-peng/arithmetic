import numpy as np
import quimb.tensor as qtn
CP2 = np.zeros((2,)*3)
CP2[0,0,0] = CP2[1,1,1] = 1.0
ADD = np.zeros((2,)*3)
ADD[0,0,0] = ADD[0,1,1] = ADD[1,0,1] = 1.0
TMP = np.einsum('ijk,klm->ijlm',CP2,ADD)
def get_CP(d):
    out = np.zeros((d,)*3)
    for i in range(d):
        out[i,i,i] = 1.0
    return out
def get_weighted_sum1(X,ws,b):
    # weighted sum
    arrays = []
    for i in range(len(ws)):
        tmp = np.einsum('ipq,i->ipq',X,np.array([1.0,ws[i]]))
        if i>0:
            tmp = np.einsum('ipq,ijk->jkpq',tmp,ADD)
        if i==len(ws)-1:
            tmp = np.einsum('ijpq,jkl,k->ilpq',tmp,ADD,np.array([1.0,b]))
        arrays.append(tmp)
    return [arrays]
def append(arrays,ts):
    nblk,blk_size = len(arrays),len(arrays[0])
    assert nblk==len(ts)
    out = []
    for b in range(nblk):
        blk = arrays[b]
        for k in range(blk_size):
            if k==blk_size-1:
                tmp = ts[b].copy()
            else:
                tmp = np.ones(1).reshape(1,1,1,1) if b==0 else \
                      np.eye(2).reshape(1,1,2,2) 
            out.append([t.copy() for t in blk[k]]+[tmp])
    return out
def get_pol(h,coeff):
    # coeff = a0,...,an
    nblk = len(coeff)-1
    ts = []
    for i in range(nblk-1,-1,-1):
        tmp = np.einsum('ltib,i->ltb',TMP,np.array([1.0,coeff[i]]))
        tmp = tmp.reshape(2,1,2,2)
        if i==nblk-1:
            tmp = np.einsum('lrtb,t->lrb',tmp,np.array([1.0,coeff[-1]]))
            tmp = tmp.reshape(2,1,1,2)
        if i==0:
            tmp = np.swapaxes(tmp,axis1=1,axis2=-1)
        ts.append(tmp)
    arrays = [h.copy() for i in range(nblk)]
    return append(arrays,ts)
def get_weighted_sum(hs,ws,b):
    nblk = len(ws)
    ts = []
    for i in range(nblk):
        tmp = np.einsum('ljtb,j->ltb',TMP,np.array([1.0,ws[i]]))
        tmp = tmp.reshape(2,1,2,2)
        if i==0:
            tmp = tmp[:,:,0,:]
            tmp = tmp.reshape(2,1,1,2)
        if i==nblk-1:
            tmp = np.swapaxes(tmp,axis1=1,axis2=-1)
            tmp = np.einsum('litb,ijr,j->lrtb',tmp,ADD,np.array([1.0,b]))
        ts.append(tmp)
    return append(hs,ts)
def get_layer1(xs,W,B,coeff):
    # hi = Wijxj+Bi
    d = len(xs)
    X = np.zeros((d,2))
    X[:,0] = np.ones(d)
    X[:,1] = xs.copy()
    X = np.einsum('ri,pqr->ipq',X,get_CP(d))
    ls = []
    for i in range(len(B)):
        h = get_weighted_sum1(X,W[i,:],B[i])
        ls.append(get_pol(h,coeff))
    return ls
def get_layer(hs,W,B,coeff):
    ls = []
    for i in range(len(B)):
        h = get_weighted_sum(hs,W[i,:],B[i])
        ls.append(get_pol(h,coeff))
    return ls
def get_peps(arrays,N,tr):
    nrow,ncol = len(arrays),len(arrays[0])
    out = []
    for i in range(nrow):
        row = []
        for j in range(ncol):
            t = arrays[i][j].copy()
            if i==0:
                tmp = tr if j < N else np.ones(1)
                t = np.einsum('...pq,p->...q',t,tmp)
            if i==nrow-1:
                tmp = np.ones_like(tr) if j < N else np.ones(1)
                t = np.einsum('...q,q->...',t,tmp)
            row.append(t.reshape(t.shape+(1,)))
        row[-1] = row[-1][:,-1,...]
        out.append(row)
    return qtn.PEPS(arrays=out,shape='lrdup')
