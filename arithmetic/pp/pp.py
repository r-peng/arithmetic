import numpy as np
import quimb.tensor as qtn
np.set_printoptions(suppress=True,linewidth=1000)
def resolve(mps,sign,remove_lower=True):
    arrs = []
    ng = mps[0].shape[0]
    for i in range(mps.L):
        row = []
        for j in range(mps.L):
            if i>j+1:
                data = np.ones(1).reshape((1,)*4)
                if i==0 or i==mps.L-1:
                    data = data[...,0]
                if j==0 or j==mps.L-1:
                    data = data[...,0]
            elif i==j+1:
                data = np.eye(ng).reshape((1,ng,ng,1))
                if i==mps.L-1:
                    data = data[...,0]
                if j==0:
                    data = data[0,...]
            else:
                ti,tj = mps[i].copy(),mps[j].copy()
                output_inds = []
                if j>0:
                    tj.reindex_({tj.inds[0]:'l'})
                    output_inds.append('l')
                if j<mps.L-1:
                    tj.reindex_({tj.inds[-2]:'r'})
                    output_inds.append('r')
                if i>0:
                    ti.reindex_({ti.inds[0]:'d'})
                    output_inds.append('d')
                if i<mps.L-1:
                    ti.reindex_({ti.inds[-2]:'u'})
                    output_inds.append('u')
                ti.reindex_({ti.inds[-1]:'p'})
                tj.reindex_({tj.inds[-1]:'p'})
                data = qtn.tensor_contract(ti,tj,qtn.Tensor(data=sign,inds=('p',)),
                                            output_inds=output_inds).data
            row.append(data.reshape(data.shape+(1,)))
        arrs.append(row)
    peps = qtn.PEPS(arrs,shape='lrdup')
    for i in range(peps.Lx):
        for j in range(peps.Ly):
            T = peps[peps.site_tag(i,j)]
            T.modify(data=T.data[...,0],inds=T.inds[:-1])
    if remove_lower:
        for i in range(peps.Lx):
            for j in range(peps.Ly):
                T = peps[peps.site_tag(i,j)]
                for ix,sh in enumerate(T.shape):
                    if sh==1:
                        peps.add_tensor(qtn.Tensor(data=np.ones(1),inds=(T.inds[ix],),tags=T.tags))
                peps.contract_tags(T.tags,which='all',inplace=True)
                if len(peps[peps.site_tag(i,j)].inds)==0:
                    tid = list(peps._get_tids_from_tags(peps.site_tag(i,j)))[0]
                    peps._pop_tensor(tid)
    return peps

