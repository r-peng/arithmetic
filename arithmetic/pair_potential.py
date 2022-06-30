import numpy as np
import quimb.tensor as qtn
from tqdm import tqdm 
import cotengra as ctg
np.set_printoptions(suppress=True,linewidth=100)
def morse(ri,rj,De=1.,a=1.,re=1.):
    r = np.linalg.norm(ri-rj)
    return De*(1.0-np.exp(-a*(r-re)))**2-De
def get_hypergraph1D(x,wx,N,
                   beta=1.,v_params={'De':1.,'a':1.,'re':1.}):
    ngx = len(x)
    data = np.zeros((ngx,)*2)
    for xi in range(ngx):
        ri = np.array([x[xi]])
        for xj in range(ngx):
            rj = np.array([x[xj]])
            hij = morse(ri,rj,**v_params)
            data[xi,xj] = np.exp(-beta*hij)

    wx_ = np.power(wx,np.ones_like(wx)/(N-1))
    data = np.einsum('ij,i,j->ij',data,wx_,wx_).reshape(data.shape+(1,))

    tn = qtn.TensorNetwork([])
    for i in range(N):
        for j in range(i+1,N):
            bix = qtn.rand_uuid()

            inds = f'x{i}',f'x{j}',bix
            tags = f'x{i}',f'x{j}','x'
            tn.add_tensor(qtn.Tensor(data=data.copy(),inds=inds,tags=tags))
    return tn
def get_hypergraph2D(x,y,wx,wy,N,
                   beta=1.,v_params={'De':1.,'a':1.,'re':1.}):
    ngx,ngy = len(x),len(y)
    data = np.zeros((ngx,)*2+(ngy,)*2)
    for xi in range(ngx):
        for yi in range(ngy):
            ri = np.array([x[xi],y[yi]])
            for xj in range(ngx):
                for yj in range(ngy):
                    rj = np.array([x[xj],y[yj]])
                    hij = morse(ri,rj,**v_params)
                    data[xi,xj,yi,yj] = np.exp(-beta*hij)
    T = qtn.Tensor(data=data,inds=('xi','xj','yi','yj'))
    Tx,Ty  = T.split(left_inds=('xi','xj'),right_inds=('yi','yj'),bond_ind='bix')
    Tx.transpose_('xi','xj','bix')
    Ty.transpose_('yi','yj','bix')

    wx_ = np.power(wx,np.ones_like(wx)/(N-1))
    wy_ = np.power(wy,np.ones_like(wy)/(N-1))
    Tx = np.einsum('ij...,i,j->ij...',Tx.data,wx_,wx_)
    Ty = np.einsum('ij...,i,j->ij...',Ty.data,wy_,wy_)

    tn = qtn.TensorNetwork([])
    for i in range(N):
        for j in range(i+1,N):
            bix = qtn.rand_uuid()

            inds = f'x{i}',f'x{j}',bix
            tags = f'x{i}',f'x{j}','x'
            tn.add_tensor(qtn.Tensor(data=Tx.copy(),inds=inds,tags=tags))

            inds = f'y{i}',f'y{j}',bix
            tags = f'y{i}',f'y{j}','y'
            tn.add_tensor(qtn.Tensor(data=Ty.copy(),inds=inds,tags=tags))
    return tn
def get_hypergraph3D(x,y,z,wx,wy,wz,N,
                   beta=1.,v_params={'De':1.,'a':1.,'re':1.}):
    ngx,ngy,ngz = len(x),len(y),len(z)
    data = np.zeros((ngx,)*2+(ngy,)*2+(ngz,)*2)
    for xi in range(ngx):
        for yi in range(ngy):
            for zi in range(ngz):
                ri = np.array([x[xi],y[yi],z[zi]])
                for xj in range(ngx):
                    for yj in range(ngy):
                        for zj in range(ngz):
                            rj = np.array([x[xj],y[yj],z[zj]])
                            hij = morse(ri,rj,**v_params)
                            data[xi,xj,yi,yj,zi,zj] = np.exp(-beta*hij)
    T = qtn.Tensor(data=data,inds=('xi','xj','yi','yj','zi','zj'))
    Tx,T  = T.split(left_inds=('xi','xj'),right_inds=None,bond_ind='b1')
    Ty,Tz = T.split(left_inds=None,right_inds=('zi','zj'),bond_ind='b2')
    Tx.transpose_('xi','xj','b1')
    Ty.transpose_('yi','yj','b1','b2')
    Tz.transpose_('zi','zj','b2')

    wx_ = np.power(wx,np.ones_like(wx)/(N-1))
    wy_ = np.power(wy,np.ones_like(wy)/(N-1))
    wz_ = np.power(wz,np.ones_like(wz)/(N-1))
    Tx = np.einsum('ij...,i,j->ij...',Tx.data,wx_,wx_)
    Ty = np.einsum('ij...,i,j->ij...',Ty.data,wy_,wy_)
    Tz = np.einsum('ij...,i,j->ij...',Tz.data,wz_,wz_)

    tn = qtn.TensorNetwork([])
    for i in range(N):
        for j in range(i+1,N):
            b1,b2 = [qtn.rand_uuid() for _ in range(2)]

            inds = f'x{i}',f'x{j}',b1
            tags = f'x{i}',f'x{j}','x'
            tn.add_tensor(qtn.Tensor(data=Tx.copy(),inds=inds,tags=tags))

            inds = f'y{i}',f'y{j}',b1,b2
            tags = f'y{i}',f'y{j}','y'
            tn.add_tensor(qtn.Tensor(data=Ty.copy(),inds=inds,tags=tags))

            inds = f'z{i}',f'z{j}',b2
            tags = f'z{i}',f'z{j}','z'
            tn.add_tensor(qtn.Tensor(data=Tz.copy(),inds=inds,tags=tags))
    return tn
def resolve1D(tn,N,remove_lower=True):
    return resolve_braket(tn,N,'x',remove_lower=remove_lower)
def resolve2D(tn,N,remove_lower=True):
    tnx = tn.select('x').copy()
    tny = tn.select('y').copy()
    tnx = resolve_braket(tnx,N,'x',remove_lower=remove_lower)
    tny = resolve_braket(tny,N,'y',remove_lower=remove_lower)
    return tnx|tny
def resolve_braket(tn,N,tag,remove_lower=True):
    ng,_,pdim = tn[f'{tag}0',f'{tag}1'].shape
    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1.

    arrs = []
    for i in range(N-1):
        row = []
        for j in range(1,N):
            if j<i:
                data = np.ones(1).reshape((1,)*5)
                if i==0 or i==N-2:
                    data = data[...,0,:]
                if j==1 or j==N-1:
                    data = data[...,0,:]
            elif j==i:
                data = np.eye(ng).reshape((1,ng,ng,1,1))
                if i==N-2:
                    data = data[...,0,:]
                if j==1:
                    data = data[0,:]
            else:
                try:
                    data = tn[f'{tag}{i}',f'{tag}{j}'].data
                    try:
                        tr = tn[f'{tag}{i}',f'{tag}{j+1}']
                        if j>1:
                            data = np.einsum('x...,xlr->lr...',data,CP)
                    except KeyError:
                        if j<N-1:
                            data = data.reshape((ng,1,ng,pdim))
                    if i>0 and i<N-2:
                        try:
                            tu = tn[f'{tag}{i-1}',f'{tag}{j}']
                            data = np.einsum('...yk,yud->...udk',data,CP)
                        except KeyError:
                            data = data.reshape(data.shape[:-2]+(1,)+data.shape[-2:])
                except KeyError:
                    data = np.ones(1).reshape((1,)*5)
                    if i==0 or i==N-2:
                        data = data[...,0,:]
                    if j==1 or j==N-1:
                        data = data[...,0,:]
            row.append(data)
        arrs.append(row)
    peps = qtn.PEPS(arrs,shape='lrdup')
    if pdim==1:
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
    peps.add_tag(tag)
    return peps
def contract(peps,tags,min_size=3,progbar=False,**compress_opts):
    imin,imax = 0,peps.Lx-1
    jmin,jmax = 0,peps.Ly-1
    step_range = range(peps.Lx,min_size,-2) 
    layer_tags = None if len(tags) else tags
    if progbar:
        step_range = tqdm(step_range)
    num_compress = 0
    for step in step_range:
        # remove left corner
        for tag in tags:
            for i in [imin,imin+1]:
                peps.contract_between(
                    (peps.site_tag(i,jmin),tag),(peps.site_tag(i,jmin+1),tag))
        jmin += 1
        # remove top corner
        for tag in tags:
            for j in [jmax-1,jmax]:
                peps.contract_between(
                    (peps.site_tag(imax,j),tag),(peps.site_tag(imax-1,j),tag))
        imax -= 1

        if jmax-jmin<min_size:
            break
        peps.contract_boundary_from_bottom_(layer_tags=layer_tags,
            xrange=(imin,imin+1),yrange=(jmin,jmax),equalize_norms=1.,
            compress_sweep='right',**compress_opts)
        num_compress += jmax-jmin
        imin += 1

        if imax-imin<min_size:
            break
        peps.contract_boundary_from_right_(layer_tags=layer_tags,
            xrange=(imin,imax),yrange=(jmax-1,jmax),equalize_norms=1.,
            compress_sweep='down',**compress_opts)
        num_compress += imax-imin
        jmax -= 1
        #print(imin,imax,jmin,jmax,peps.num_tensors,peps.max_bond())
        #print(peps)
    print('num_tensors=',peps.num_tensors)
    print('number of compression=',num_compress)
    #out = peps.contract()
    opt = ctg.HyperOptimizer(minimize='combo',parallel='ray')
    tree = peps.contraction_tree(opt)
    out,exp = tree.contract(peps.arrays,strip_exponent=True)
    if out<0.:
        print('contracts to=',out)
    return np.log10(abs(out))+exp+peps.exponent
#def resolve3D(tn,N):
#    ng = {key:tn[f'{key}0',f'{key}1'].shape[0] for key in ['x','y','z']}
#    CP = {key:np.zeros((ng[key],)*3) for key in ['x','y','z']}
#    for key in ['x','y','z']:
#        for i in range(ng[key]):
#            CP[key][i,i,i] = 1.
#
#    for i in range(N):
#        for j in range(i+1,N):
#            for key in ['x','y','z']:
#                tn[f'{key}{i}',f'{key}{j}'].reindex_({f'{key}{i}':qtn.rand_uuid(),
#                                                      f'{key}{j}':qtn.rand_uuid()})
#    for i in range(N):
#        for j in range(i+1,N):
#            if i==0:
#                if j==1 or j==N-1:
#                    continue
#                else:
#                    for key in ['x','y','z']:
#                        lix = tn[f'{key}{i}',f'{key}{j-1}',f'r{i}',f'r{j-1}'].inds[0] if j==2 else\
#                              tn[f'{key}{i}',f'{key}{j-1}','CP','ROW'].inds[-1]
#                        rix = tn[f'{key}{i}',f'{key}{j+1}'].inds[0] if j==N-2 else\
#                              qtn.rand_uuid()
#                        pix = tn[f'{key}{i}',f'{key}{j}',f'r{i}',f'r{j}'].inds[0]
#                        tags = f'{key}{i}',f'{key}{j}','CP','ROW'
#                        tn.add_tensor(qtn.Tensor(data=CP[key],inds=(lix,pix,rix),tags=tags))
#            elif i==N-2:
#                continue
#            else:
#                for key in ['x','y','z']:
#                    if j<N-1:
#                        if j==i+1:
#                            lix = tn[f'{key}{i-1}',f'{key}{j-1}',f'r{i-1}',f'r{j-1}'].inds[1] if i==1 else\
#                                  tn[f'{key}{i-1}',f'{key}{j-1}','CP','COL'].inds[-1]
#                        else:
#                            lix = tn[f'{key}{i}',f'{key}{j-1}','CP','ROW'].inds[-1]
#                        rix = tn[f'{key}{i}',f'{key}{j+1}',f'r{i}',f'r{j+1}'].inds[0] if j==N-2 else\
#                              qtn.rand_uuid()
#                        pix = tn[f'{key}{i}',f'{key}{j}',f'r{i}',f'r{j}'].inds[0]
#                        tags = f'{key}{i}',f'{key}{j}','CP','ROW'
#                        tn.add_tensor(qtn.Tensor(data=CP[key],inds=(lix,pix,rix),tags=tags))
#
#                    tix = tn[f'{key}{i-1}',f'{key}{j}',f'r{i-1}',f'r{j}'].inds[1] if i==1 else\
#                          tn[f'{key}{i-1}',f'{key}{j}','CP','COL'].inds[-1]
#                    if i==N-3:
#                        bix = tn[f'{key}{i+1}',f'{key}{j}',f'r{i+1}',f'r{j}'].inds[1] if j==N-1 else\
#                              tn[f'{key}{i+1}',f'{key}{j+1}',f'r{i+1}',f'r{j+1}'].inds[0]
#                    else:
#                        bix = qtn.rand_uuid()
#                    pix = tn[f'{key}{i}',f'{key}{j}',f'r{i}',f'r{j}'].inds[1]
#                    tags = f'{key}{i}',f'{key}{j}','CP','COL'
#                    tn.add_tensor(qtn.Tensor(data=CP[key],inds=(tix,pix,bix),tags=tags))
#
#    for i in range(N):
#        for j in range(i+1,N):
#            for key in ['x','y','z']:
#                tn.contract_tags((f'{key}{i}',f'{key}{j}'),which='all',inplace=True)
#    return tn

