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
def get_data_ai(data,ai):
    return np.einsum('ijk,klm,l->ijm',data,ADD,np.array([1.0,ai]))
def get_data_an(data,an):
    data = np.einsum('ijk,j->ik',data,np.array([1.0,an]))
    ldim,ddim = data.shape
    return data.reshape(ldim,1,ddim)
def modify(mpo,get_data,coeff):
    site_tag = mpo._site_tag_id.format(mpo.L-1)
    tid = list(mpo._get_tids_from_tags(site_tag,which='any'))[0]
    t = mpo.tensor_map[tid]
    uind = mpo._upper_ind_id.format(mpo.L-1)
    dind = mpo._lower_ind_id.format(mpo.L-1)
    lind = list(set(t._inds)-{uind,dind})[0]
    t.transpose(lind,uind,dind,inplace=True)
    data = get_data(t._data,coeff)
    t.modify(data=data,inds=(lind,uind,dind))
    return mpo 
def get_sum(qs):
    arrays = []
    for i in range(len(qs)):
        data = np.stack([np.ones_like(qs[i]),qs[i]],axis=0)
        data = np.einsum('ir,pqr->ipq',data,get_CP(len(qs[i])))
        if i > 0:
            data = np.einsum('ipq,ijk->jkpq',data,ADD)
        arrays.append(data)
    arrays.append(CP2)
    return qtn.MatrixProductOperator(arrays,shape='lrud')
def get_pol(qs,coeff,**compress_opts):
    for i in range(len(coeff)-2,-1,-1):
        other = get_sum(qs)
        other = modify(other,get_data_ai,coeff[i])
        mpo = other.copy() if i==len(coeff)-2 else \
              mpo._apply_mpo(other,compress=True,**compress_opts)
        print(mpo)
    return modify(mpo,get_data_an,coeff[-1])
def get_product(qs,**compress_opts):
    k,N = qs.shape
    for i in range(k):
        other = get_sum(qs[i,:])
        mpo = other.copy() if i==len(coeff)-2 else \
              mpo._apply_mpo(other,compress=True,**compress_opts)
    return modify(mpo,get_data_an,1.0)
def integrate(mpo,tr):
    if isinstance(tr,np.ndarray):
        tr = [tr for i in range(mpo.L-1)]
    for i in range(mpo.L):
        inds = [mpo._upper_ind_id.format(i)]
        data = np.ones(1) if i==mpo.L-1 else tr[i] 
        mpo.add(qtn.Tensor(data=data,inds=inds))
        inds = [mpo._lower_ind_id.format(i)]
        data = np.array([0.0,1.0]) if i==mpo.L-1 else np.ones_like(tr[i])
        mpo.add(qtn.Tensor(data=data,inds=inds))
    return qtn.tensor_contract(*mpo)#,mpo.exponent 
    
