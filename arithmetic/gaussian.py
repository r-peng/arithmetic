import numpy as np
import quimb.tensor as qtn
from tqdm import tqdm 
import cotengra as ctg
def get_A(N,hw,emax=0.9):
    A = np.random.rand(N,N)
    A += A.T
    A -= 1. 
    for i in range(N):
        A[i,i+hw+1:] = 0.
        A[i+hw+1:,i] = 0.
    D,U = np.linalg.eigh(A)
    a,b = np.floor(D[0]),np.ceil(D[-1])
    D -= a
    D /= (b-a)
    A = np.linalg.multi_dot([U,np.diag(D),U.T])
    return A,D
def get_hypergraph(A,xs,ws,simplify=True,cutoff=1e-10):
    N = A.shape[0]
    tn = qtn.TensorNetwork([])
    for i in range(N):
        inds = f'x{i}',
        tn.add_tensor(qtn.Tensor(data=ws.copy(),inds=inds,tags=f'w{i}'))

        data = np.exp(-A[i,i]*np.square(xs))
        tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags=inds))
        for j in range(i+1,N):
            fac = A[i,j]+A[j,i]
            if abs(fac)>cutoff:
                inds = f'x{i}',f'x{j}'
                data = np.exp(-fac*np.einsum('i,j->ij',xs,xs))
                tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags=inds))

    if simplify:
        for i in range(N):
            xtids = list(tn._get_tids_from_tags(f'x{i}',which='any'))
            wtid = list(tn._get_tids_from_tags(f'w{i}',which='any'))[0]
            for xtid in xtids:
                if len(tn.tensor_map[xtid].shape)==1:
                    break
            wi = tn._pop_tensor(wtid)
            xi = tn._pop_tensor(xtid)
            xtids.remove(xtid)

            data = np.power(np.multiply(wi.data,xi.data),np.ones_like(xi.data)/len(xtids))
            for xtid in xtids:
                T = tn.tensor_map[xtid]
                inds = list(T.inds)
                inds.remove(f'x{i}')
                T.transpose_(f'x{i}',inds[0])
                T.modify(data=np.einsum('x,xy->xy',data,T.data))
    return tn
def resolve(tn,N,remove_lower=True):
    ng = tn[f'x0',f'x1'].shape[0]
    CP = np.zeros((ng,)*3)
    for i in range(ng):
        CP[i,i,i] = 1.

    arrs = []
    for i in range(N-1):
        row = []
        for j in range(1,N):
            if j<i:
                data = np.ones(1).reshape((1,)*4)
                if i==0 or i==N-2:
                    data = data[...,0]
                if j==1 or j==N-1:
                    data = data[...,0]
            elif j==i:
                data = np.eye(ng).reshape((1,ng,ng,1))
                if i==N-2:
                    data = data[...,0]
                if j==1:
                    data = data[0,...]
            else:
                try:
                    tn[f'x{i}',f'x{j}'].transpose_(f'x{i}',f'x{j}')
                    data = tn[f'x{i}',f'x{j}'].data
                    try:
                        tr = tn[f'x{i}',f'x{j+1}']
                        if j>1:
                            data = np.einsum('xy,xlr->lry',data,CP)
                    except KeyError:
                        if j<N-1:
                            data = data.reshape((ng,1,ng))
                    if i>0 and i<N-2:
                        try:
                            tu = tn[f'x{i-1}',f'x{j}']
                            data = np.einsum('...y,yud->...ud',data,CP)
                        except KeyError:
                            data = data.reshape(data.shape[:-1]+(1,data.shape[-1]))
                except KeyError:
                    data = np.ones(1).reshape((1,)*4)
                    if i==0 or i==N-2:
                        data = data[...,0]
                    if j==1 or j==N-1:
                        data = data[...,0]
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
def contract(peps,final=3,total=None,**compress_opts):
    for i in range(peps.Lx-1):
        peps.contract_between(peps.site_tag(i,i),peps.site_tag(i+1,i))
    def corner(peps,imin,imax,jmin,jmax):
        peps.contract_between(peps.site_tag(imin,jmin),peps.site_tag(imin,jmin+1))
        peps.contract_between(peps.site_tag(imax,jmax),peps.site_tag(imax-1,jmax))
        jmin += 1
        imax -= 1
        peps.contract_between(peps.site_tag(imin,jmax),peps.site_tag(imin+1,jmax))
        return peps,imin,imax,jmin,jmax

    imin,imax = 0,peps.Lx-1
    jmin,jmax = 0,peps.Ly-1
    num_compress = 0

    const = 1 if peps.Lx % 2 == 1 else 3
    final_tn_size = (final-1)*final*2 + const*final

    if total is not None:
        progbar = tqdm(total=total*2)
    peps,imin,imax,jmin,jmax = corner(peps,imin,imax,jmin,jmax)
    while True:
        for j in range(jmin,jmax-1):
            peps.contract_between(peps.site_tag(imin,j),peps.site_tag(imin+1,j))
        peps.contract_between(peps.site_tag(imin,jmax-1),peps.site_tag(imin+1,jmax))
        peps.contract_between(peps.site_tag(imin+1,jmax-1),peps.site_tag(imin,jmax))
        for i in range(imin+2,imax+1):
            peps.contract_between(peps.site_tag(i,jmax),peps.site_tag(i,jmax-1))
        imin += 1
        jmax -= 1

        if peps.num_tensors <= final_tn_size:
            break
        peps,imin,imax,jmin,jmax = corner(peps,imin,imax,jmin,jmax)

        seq  = [peps.site_tag(imin,j) for j in range(jmin,jmax)]
        seq += [peps.site_tag(i,jmax) for i in range(imin+1,imax+1)]
        for i in range(len(seq)-1):
            peps.canonize_between(seq[i],seq[i+1],absorb='right',equalize_norms=1.)
            if total is not None:
                progbar.update()
        for i in range(len(seq)-1,0,-1):
            peps.compress_between(seq[i-1],seq[i],absorb='left',equalize_norms=1.,
                                  **compress_opts)
            if total is not None:
                progbar.update()
            num_compress += 1
    if total is None:
        return num_compress
    progbar.close()
    opt = ctg.HyperOptimizer(minimize='combo',parallel='ray')
    tree = peps.contraction_tree(opt)
    out,exp = tree.contract(peps.arrays,strip_exponent=True)
    if out<0.:
        print('contracts to=',out)
    return np.log10(abs(out))+exp+peps.exponent
def contractW(peps,W,total=None,**compress_opts):
    for i in range(peps.Lx-1):
        peps.contract_between(peps.site_tag(i,i),peps.site_tag(i+1,i))

    imin,imax = 0,peps.Lx-1
    jmin,jmax = 0,peps.Ly-1
    num_compress = 0

    while True:
        peps.contract_between(peps.site_tag(imin,jmin),peps.site_tag(imin,jmin+1))
        jmin += 1
        for j in range(jmin,jmin+W):
            peps.contract_between(peps.site_tag(imin,j),peps.site_tag(imin+1,j))
        imin += 1
        seq = [peps.site_tag(imin,j) for j in range(jmin,jmin+W)]
        for i in range(len(seq)-1):
            peps.canonize_between(seq[i],seq[i+1],absorb='right',equalize_norms=1.)
        for i in range(len(seq)-1,0,-1):
            peps.compress_between(seq[i-1],seq[i],absorb='left',equalize_norms=1.,
                                  **compress_opts)
      

        break
    return    
#def contract_scheme1(peps,max_bond=None,cutoff=1e-10,min_size=3,progbar=False):
#    for i in range(peps.Lx-1):
#        peps.contract_between(peps.site_tag(i,i),peps.site_tag(i+1,i))
#    imin,imax = 0,peps.Lx-1
#    jmin,jmax = 0,peps.Ly-1
#    step_range = range(peps.Lx,min_size,-3) 
#    if progbar:
#        step_range = tqdm(step_range) 
#    num_compress = 0
#    for step in step_range:
#        if jmax-jmin-1<min_size:
#            break
#        peps.contract_between(peps.site_tag(imin,jmin),peps.site_tag(imin,jmin+1))
#        peps.contract_boundary_from_bottom_(
#            xrange=(imin,imin+1),yrange=(jmin+1,jmax),equalize_norms=1.,
#            compress_sweep='right',max_bond=max_bond,cutoff=cutoff)
#        num_compress += jmax-jmin-1
#        imin += 1
#        jmin += 1
#
#        if imax-1-imin<min_size:
#            break
#        peps.contract_between(peps.site_tag(imax,jmax),peps.site_tag(imax-1,jmax))
#        peps.contract_boundary_from_right_(
#            xrange=(imin,imax-1),yrange=(jmax-1,jmax),equalize_norms=1.,
#            compress_sweep='up',max_bond=max_bond,cutoff=cutoff)
#        num_compress += imax-1-imin
#        imax -= 1
#        jmax -= 1
#
#        if imax-imin-1<min_size:
#            break
#        diag = list(zip(range(imin,imax+1),range(jmin,jmax+1)))
#        for (i,j) in diag[1:]:
#            peps.contract_between(peps.site_tag(i,j),peps.site_tag(i-1,j))
#        peps.contract_between(peps.site_tag(*diag[0]),peps.site_tag(*diag[1]))
#        diag.pop(0)
#        for i in range(len(diag)-1):
#            peps.canonize_between(peps.site_tag(*diag[i]),peps.site_tag(*diag[i+1]),
#                                  absorb='right',cutoff=cutoff)
#        for i in range(len(diag)-1,0,-1):
#            peps.compress_between(peps.site_tag(*diag[i-1]),peps.site_tag(*diag[i]),
#                absorb='left',max_bond=max_bond,cutoff=cutoff,equalize_norms=1.)
#            num_compress += 1
#        jmin += 1
#        imax -= 1
#        #print(imin,imax,jmin,jmax,peps.num_tensors)
#    print('num_tensors=',peps.num_tensors)
#    print('number of compression=',num_compress)
#    return np.log10(peps.contract())+peps.exponent
#def contract_scheme2(peps,min_size=3,progbar=False,**compress_opts):
#    imin,imax = 0,peps.Lx-1
#    jmin,jmax = 0,peps.Ly-1
#    step_range = range(peps.Lx,min_size,-2) 
#    if progbar:
#        step_range = tqdm(step_range)
#    num_compress = 0
#    for step in step_range:
#        # remove left corner
#        for i in [imin,imin+1]:
#            peps.contract_between(
#                peps.site_tag(i,jmin),peps.site_tag(i,jmin+1))
#        jmin += 1
#        # remove top corner
#        for j in [jmax-1,jmax]:
#            peps.contract_between(
#                peps.site_tag(imax,j),peps.site_tag(imax-1,j))
#        imax -= 1
#
#        if jmax-jmin<min_size:
#            break
#        peps.contract_boundary_from_bottom_(
#            xrange=(imin,imin+1),yrange=(jmin,jmax),equalize_norms=1.,
#            compress_sweep='right',**compress_opts)
#        num_compress += jmax-jmin
#        imin += 1
#
#        if imax-imin<min_size:
#            break
#        peps.contract_boundary_from_right_(
#            xrange=(imin,imax),yrange=(jmax-1,jmax),equalize_norms=1.,
#            compress_sweep='down',**compress_opts)
#        num_compress += imax-imin
#        jmax -= 1
#        #print(imin,imax,jmin,jmax,peps.num_tensors)
#    print('num_tensors=',peps.num_tensors)
#    print('number of compression=',num_compress)
#    opt = ctg.HyperOptimizer(minimize='combo',parallel='ray')
#    tree = peps.contraction_tree(opt)
#    out,exp = tree.contract(peps.arrays,strip_exponent=True)
#    if out<0.:
#        print('contracts to=',out)
#    return np.log10(abs(out))+exp+peps.exponent
#def contract_scheme3(peps,max_bond=None,cutoff=1e-10,min_size=3,progbar=False):
#    split = (peps.Lx-1)//2
#    step_range = range(split)
#    if progbar:
#        step_range = tqdm(step_range) 
#    num_compress = 0
#    for i in step_range:
#        peps.contract_boundary_from_left_(xrange=(0,i+1),yrange=(i,i+1),
#            equalize_norms=1.,compress_sweep='down',max_bond=max_bond,cutoff=cutoff)
#        peps.contract_boundary_from_top_(xrange=(peps.Lx-1-i-1,peps.Lx-1-i),
#            yrange=(peps.Lx-1-i-1,peps.Ly-1),
#            equalize_norms=1.,compress_sweep='left',max_bond=max_bond,cutoff=cutoff)
#        num_compress += (i+1)*2
#
#    imin,imax = 0,peps.Lx-1-split
#    jmin,jmax = split,peps.Ly-1
#    step_range = range(peps.Ly-split,min_size,-2) 
#    if progbar:
#        step_range = tqdm(step_range) 
#    for step in step_range:
#        if jmax-jmin<min_size:
#            break
#        peps.contract_boundary_from_bottom_(
#            xrange=(imin,imin+1),yrange=(jmin,jmax),equalize_norms=1.,
#            compress_sweep='left',max_bond=max_bond,cutoff=cutoff)
#        num_compress += jmax-jmin
#        imin += 1
#
#        if imax-imin<min_size:
#            break
#        peps.contract_boundary_from_left_(
#            xrange=(imin,imax),yrange=(jmin,jmin+1),equalize_norms=1.,
#            compress_sweep='up',max_bond=max_bond,cutoff=cutoff)
#        num_compress += imax-imin
#        jmin += 1
#
#        if jmax-jmin<min_size:
#            break
#        peps.contract_boundary_from_top_(
#            xrange=(imax-1,imax),yrange=(jmin,jmax),equalize_norms=1.,
#            compress_sweep='right',max_bond=max_bond,cutoff=cutoff)
#        num_compress += jmax-jmin
#        imax -= 1
#        
#        if imax-imin<min_size:
#            break
#        peps.contract_boundary_from_right_(
#            xrange=(imin,imax),yrange=(jmax-1,jmax),equalize_norms=1.,
#            compress_sweep='down',max_bond=max_bond,cutoff=cutoff)
#        num_compress += imax-imin
#        jmax -= 1
#        #print(imin,imax,jmin,jmax,peps.num_tensors)
#    print('num_tensors=',peps.num_tensors)
#    print('number of compression=',num_compress)
#    opt = ctg.HyperOptimizer(minimize='combo',parallel='ray')
#    tree = peps.contraction_tree(opt)
#    out,exp = tree.contract(peps.arrays,strip_exponent=True)
#    if out<0.:
#        print('contracts to=',out)
#    return np.log10(abs(out))+exp+peps.exponent
