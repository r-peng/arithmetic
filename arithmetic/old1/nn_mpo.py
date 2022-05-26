import numpy as np
import quimb.tensor as qtn
CP2 = np.zeros((2,)*3)
CP2[0,0,0] = CP2[1,1,1] = 1.0
ADD = np.zeros((2,)*3)
ADD[0,0,0] = ADD[0,1,1] = ADD[1,0,1] = 1.0
def get_CP(d):
    out = np.zeros((d,)*3)
    for i in range(d):
        out[i,i,i] = 1.0
    return out
def get_data_ai(data,tmp):
    return np.einsum('li,ijk->ljk',data[:,:,0],tmp)
def get_data_an(data,tmp):
    data = np.einsum('ijk,k->ij',data,tmp)
    ldim,ddim = data.shape
    return data.reshape(ldim,1,ddim)
def modify(mpo,tmp,get_data):
    site_tag = mpo._site_tag_id.format(mpo.L-1)
    tid = list(mpo._get_tids_from_tags(site_tag,which='any'))[0]
    t = mpo.tensor_map[tid]
    uind = mpo._upper_ind_id.format(mpo.L-1)
    dind = mpo._lower_ind_id.format(mpo.L-1)
    lind = list(set(t._inds)-{uind,dind})[0]
    t.transpose(lind,dind,uind,inplace=True)
    data = get_data(t._data,tmp)
    t.modify(data=data,inds=(lind,uind,dind))
    return mpo 
def rescale(mpo,value=1.0):
    for tid in mpo.tensor_map:
        t = mpo.tensor_map[tid]
#        print(t.data.shape)
        size = np.prod(np.array(t.data.shape))
        mpo.strip_exponent(tid,value=np.sqrt(size)*value)
#        mpo.strip_exponent(tid,value=size*value)
    return mpo
def get_weighted_sum1(X,ws,b):
    # weighted sum
    arrays = []
    for i in range(len(ws)):
        tmp = np.einsum('ipq,i->ipq',X,np.array([1.0,ws[i]]))
        if i>0:
            tmp = np.einsum('ipq,ijk->jkpq',tmp,ADD)
        arrays.append(tmp)
    tmp = np.einsum('ijk,i->jk',ADD,np.array([1.0,b]))
    ldim,ddim = tmp.shape
    arrays.append(tmp.reshape(ldim,1,ddim))
    return qtn.MatrixProductOperator(arrays,shape='lrud')
def get_weighted_sum(hs,ws,b,**compress_opts):
    for i in range(len(ws)):
        other = hs[i].copy()
        tmp = np.eye(2).reshape(2,1,2) if i==0 else ADD
        tmp = np.einsum('ijk,klm,j->ilm',CP2,tmp,np.array([1.0,ws[i]]))
        other = modify(other,tmp,get_data_ai)
        mpo = other.copy() if i==0 else \
              mpo._apply_mpo(other,compress=True,**compress_opts)
        mpo = rescale(mpo)
    mpo = modify(mpo,ADD,get_data_ai)
    return modify(mpo,np.array([1.0,b]),get_data_an)
def get_pol(h,coeff,**compress_opts):
    for i in range(len(coeff)-2,-1,-1):
        other = h.copy()
        tmp = np.einsum('ijk,klm,l->ijm',CP2,ADD,np.array([1.0,coeff[i]])) 
        other = modify(other,tmp,get_data_ai)
        mpo = other.copy() if i==len(coeff)-2 else \
              mpo._apply_mpo(other,compress=True,**compress_opts)
        mpo = rescale(mpo)
    return modify(mpo,np.array([1.0,coeff[-1]]),get_data_an)
def get_layer1(xs,W,B,coeff,**compress_opts):
    # hi = Wijxj+Bi
    d = len(xs)
    X = np.zeros((d,2))
    X[:,0] = np.ones(d)
    X[:,1] = xs.copy()
    X = np.einsum('ri,pqr->ipq',X,get_CP(d))
    ls = []
    for i in range(len(B)):
        h = get_weighted_sum1(X,W[i,:],B[i])
        ls.append(get_pol(h,coeff,**compress_opts))
    return ls
def get_layer(hs,W,B,coeff,**compress_opts):
    ls = []
    for i in range(len(B)):
        h = get_weighted_sum(hs,W[i,:],B[i],**compress_opts)
        ls.append(get_pol(h,coeff,**compress_opts))
    return ls
def integrate(mpo,tr):
    for i in range(mpo.L):
        inds = [mpo._upper_ind_id.format(i)]
        data = np.ones(1) if i==mpo.L-1 else tr
        mpo.add(qtn.Tensor(data=data,inds=inds))
        inds = [mpo._lower_ind_id.format(i)]
        data = np.array([0.0,1.0]) if i==mpo.L-1 else np.ones_like(tr)
        mpo.add(qtn.Tensor(data=data,inds=inds))
    return qtn.tensor_contract(*mpo)*(10**mpo.exponent)
def get_max(mpo):
    out = []
    for tid in mpo.tensor_map:
         out.append(np.amax(np.abs(mpo.tensor_map[tid].data)))
    return np.prod(np.array(out)), out
