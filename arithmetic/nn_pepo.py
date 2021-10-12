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
def get_row(mps,tmpL,first=False):
    # lrud
    arrays = []

    site_tag = mps._site_tag_id.format(mps.L-1)
    tid = list(mps._get_tids_from_tags(site_tag,which='any'))[0]
    t = mps.tensor_map[tid]
    pind = mps._site_ind_id.format(mps.L-1)
    lind = list(set(t._inds)-{pind})[0]
    t.transpose(lind,pind,inplace=True)
    arrays.insert(0,np.einsum('lp,p...->l...',t._data,tmpL))
    CPd = get_CP(t._data.shape[1])
    for i in range(mps.L-2,-1,-1):
        rind = lind
        site_tag = mps._site_tag_id.format(i)
        tid = list(mps._get_tids_from_tags(site_tag,which='any'))[0]
        t = mps.tensor_map[tid]
        pind = mps._site_ind_id.format(i)
        if i==0:
            t.transpose(rind,pind,inplace=True)
        else:
            lind = list(set(t._inds)-{pind,rind})[0]
            t.transpose(lind,rind,pind,inplace=True)
        data = t._data.copy() if first else np.einsum('...p,pqr->...qr',t._data,CPd)
        arrays.insert(0,data)
    return arrays
def to_mpo(mps,tmpL):
    arrays = get_row(mps,tmpL,first=False)
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
def get_pol1(h,coeff,**compress_opts):
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
        ls.append(get_pol1(h,coeff,**compress_opts))
    return ls
def get_weighted_sum(hs,ws,b):
    arrays = []
    for i in range(len(ws)):
        tmp = np.einsum('ijk,j->ik',CP2,np.array([1.0,ws[i]])) if i==0 else\
              np.einsum('ijlm,j->ilm',TMP,np.array([1.0,ws[i]]))
            if i==len(ws)-1:
                tmp = np.einsum('...i,ijk,j->...k',tmp,ADD,np.array([1.0,b]))
        row = get_row(hs[i].copy(),tmp,first=(i==0))
        if i<len(ws)-1:
            row = [t.reshape(t.shape+(1,)) for t in row]
        arrays.append(row)
    return arrays
def get_arrays(arrays,tmpL,first=False):
    nrow,ncol = len(arrays),len(arrays[0])
    out = []
    for i in range(nrow-1):
        row = [t.copy() for t in arrays[i]] if first else\
              [t.reshape(t.shape+(1,)) for t in arrays[i]]
        out.append(row)
    row = []
    CPd = get_CP(arrays[-1][0].shape[-1])
    for i in range(ncol-1):
        data = arrays[-1][i] if first else\
                   np.einsum('...p,pqr->...qr',arrays[-1][i],CPd)
        row.append(data)
    data = np.einsum('lup,p..->lu..',arrays[-1][-1],tmpL) 
    row.append(data)
    out.append(row)
    return out
def get_pol(arrays,coeff):
    an,ai = coeff[-1],coeff[-2]
    tmp = np.einsum('ijlm,j,l->im',TMP,np.array([1.0,an]),np.array([1.0,ai]))
    peps = get_arrays(arrays,tmp,first=True)
    peps = qtn.PEPS(arrays=peps,shape='lrudp')
    for i in range(len(coeff)-3,-1,-1):
        tmp = np.einsum('ijlm,l->ijm',TMP,np.array([1.0,coeff[i]]))
        pepo = get_arrays(arrays,tmp,first=False)
        pepo = qtn.PEPO(arrays=peps,shape='lrudkb')
