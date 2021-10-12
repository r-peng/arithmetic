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
def to_mpo(mps,tmpL):
    arrays = []
    site_tag = mps._site_tag_id.format(0)
    tid = list(mps._get_tids_from_tags(site_tag,which='any'))[0]
    t = mps.tensor_map[tid]
    pind = mps._site_ind_id.format(0)
    rind = list(set(t._inds)-{pind})[0]
    t.transpose(rind,pind,inplace=True)
    CPd = get_CP(t._data.shape[1])
    arrays.append(np.einsum('...p,pqr->...qr',t._data,CPd))
    for i in range(1,mps.L):
        lind = rind
        site_tag = mps._site_tag_id.format(i)
        tid = list(mps._get_tids_from_tags(site_tag,which='any'))[0]
        t = mps.tensor_map[tid]
        pind = mps._site_ind_id.format(i)
        if i==mps.L-1:
            t.transpose(lind,pind,inplace=True)
        else:
            rind = list(set(t._inds)-{pind,lind})[0]
            t.transpose(lind,rind,pind,inplace=True)
        tmp = tmpL if i==mps.L-1 else CPd
        arrays.append(np.einsum('...p,pqr->...qr',t._data,tmp))
    return qtn.MatrixProductOperator(arrays,shape='lrdu')
def rescale(tn,value=1.0):
    for tid in tn.tensor_map:
        t = tn.tensor_map[tid]
        size = np.prod(np.array(t.data.shape))
        tn.strip_exponent(tid,value=np.sqrt(size)*value)
#        tn.strip_exponent(tid,value=size*value)
    return tn
def get_weighted_sum1(X,ws,b):
    # weighted sum
    arrays = []
    for i in range(len(ws)):
        tmp = np.einsum('ip,i->ip',X,np.array([1.0,ws[i]]))
        if i>0:
            tmp = np.einsum('ip,ijk->jkp',tmp,ADD)
        arrays.append(tmp)
    tmp = np.einsum('ijk,i->jk',ADD,np.array([1.0,b]))
    arrays.append(tmp)
    return qtn.MatrixProductState(arrays,shape='lrp')
def get_weighted_sum(hs,ws,b,**compress_opts):
    mps = hs[0].copy()
    tmp = np.einsum('ijk,j->ik',CP2,np.array([1.0,ws[0]])) 

    site_tag = mps._site_tag_id.format(mps.L-1)
    tid = list(mps._get_tids_from_tags(site_tag,which='any'))[0]
    t = mps.tensor_map[tid]
    pind = mps._site_ind_id.format(mps.L-1)
    lind = list(set(t._inds)-{pind})[0]
    t.transpose(lind,pind,inplace=True)
    data = np.einsum('...p,pq->...q',t._data,tmp)
    t.modify(data=data,inds=(lind,pind))
    mps = rescale(mps)
    for i in range(1,len(ws)):
        tmp = np.einsum('ijlm,j->ilm',TMP,np.array([1.0,ws[i]]))
        if i==len(ws)-1:
            tmp = np.einsum('...i,ijk,j->...k',tmp,ADD,np.array([1.0,b]))
        mpo = to_mpo(hs[i].copy(),tmp)
        mps = mpo._apply_mps(mps,**compress_opts)
        mps = rescale(mps)
    mps.distribute_exponent()
    return mps 
def get_pol(h,coeff,**compress_opts):
    mps = h.copy()
    an,ai = coeff[-1],coeff[-2]
    tmp = np.einsum('ijlm,j,l->im',TMP,np.array([1.0,an]),np.array([1.0,ai])) 

    site_tag = mps._site_tag_id.format(mps.L-1)
    tid = list(mps._get_tids_from_tags(site_tag,which='any'))[0]
    t = mps.tensor_map[tid]
    pind = mps._site_ind_id.format(mps.L-1)
    lind = list(set(t._inds)-{pind})[0]
    t.transpose(lind,pind,inplace=True)
    data = np.einsum('...p,pq->...q',t._data,tmp)
    t.modify(data=data,inds=(lind,pind))
    mps = rescale(mps)
    for i in range(len(coeff)-3,-1,-1):
        tmp = np.einsum('ijlm,l->ijm',TMP,np.array([1.0,coeff[i]]))
        mpo = to_mpo(h.copy(),tmp)
        mps = mpo._apply_mps(mps,**compress_opts)
        mps = rescale(mps)
    mps.distribute_exponent()
    return mps
def get_layer1(xs,W,B,coeff,**compress_opts):
    # hi = Wijxj+Bi
    d = len(xs)
    X = np.zeros((d,2))
    X[:,0] = np.ones(d)
    X[:,1] = xs.copy()
    ls = []
    for i in range(len(B)):
        h = get_weighted_sum1(X.T,W[i,:],B[i])
        ls.append(get_pol(h,coeff,**compress_opts))
    return ls
def get_layer(hs,W,B,coeff,**compress_opts):
    ls = []
    for i in range(len(B)):
        h = get_weighted_sum(hs,W[i,:],B[i],**compress_opts)
        ls.append(get_pol(h,coeff,**compress_opts))
    return ls
def integrate(mps,tr):
    for i in range(mps.L):
        inds = [mps._site_ind_id.format(i)]
        data = np.array([0.0,1.0]) if i==mps.L-1 else tr
        mps.add(qtn.Tensor(data=data,inds=inds))
    return qtn.tensor_contract(*mps)*(10**mps.exponent)
