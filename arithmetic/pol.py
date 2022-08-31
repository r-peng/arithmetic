import numpy as np
import quimb.tensor as qtn
from .utils import ADD,CP2,parallelized_looped_function
def get_proj_left(dim3):
    dim2 = dim3 - 1
    P = np.zeros((2,dim2,dim3))
    for i in range(dim2):
        P[0,i,i] = 1.
    P[1,dim2-1,dim3-1] = 1.
    return P 
def get_proj_right(dim3):
    dim2 = dim3 - 1
    P = np.zeros((2,dim2,dim3))
    for i in range(dim2):
        P[0,i,i] = 1.
        P[1,i,i+1] = 1.
    return P 
def get_peps(xs,ws):
    # get peps for \prod_{j=1}^k(\sum_{i=1}^N q_{ik}(x_i))
    N,k,g = xs.shape
    CP = np.zeros((g,)*3)
    for i in range(g):
        CP[i,i,i] = 1.0
    arrays = []
    for i in range(k):
        rows = []
        for j in range(N):
            data = np.ones((2,g))
            data[1,:] = xs[j,i,:]
            data = np.einsum('ir,rpq->ipq',data,CP)
            if j>0:
                data = np.einsum('i...,ijk->jk...',data,ADD)
            if j==N-1:
                data = data[:,1,...]
            if i==0 or i==k-1:
                data = np.einsum('...pq,q->...p',data,np.sqrt(ws))
            rows.append(data.reshape(data.shape+(1,)))
        arrays.append(rows)
    tn = qtn.PEPS(arrays,shape='lrudp')
    for i in range(k):
        for j in range(N):
            T = tn[tn.site_tag(i,j)]
            T.modify(data=T.data[...,0],inds=T.inds[:-1])
    return tn
def insert_projectors(peps):
    for i in range(1,peps.Lx):
        pl = get_proj_left(i+2)
        pr = get_proj_right(i+2)
        for j in range(1,peps.Ly):
            tl = peps[peps.site_tag(i-1,j-1)] if i==1 else\
                 peps[peps.row_tag(i-1),f'L{j-1},{j}']
            tr = peps[peps.site_tag(i-1,j)] if i==1 else\
                 peps[peps.row_tag(i-1),f'R{j-1},{j}']
            bl = peps[peps.site_tag(i,j-1)]
            br = peps[peps.site_tag(i,j)]
            bix1 = list(tl.bonds(tr))[0]
            bix2 = list(bl.bonds(br))[0]
            bix1_,bix2_,bix = [qtn.rand_uuid() for _ in range(3)]

            peps.add_tensor(qtn.Tensor(data=pl.copy(),inds=(bix2,bix1,bix),
                                       tags=(peps.row_tag(i),f'L{j-1},{j}')))
            peps.add_tensor(qtn.Tensor(data=pr.copy(),inds=(bix2_,bix1_,bix),
                                       tags=(peps.row_tag(i),f'R{j-1},{j}')))
            tr.reindex_({bix1:bix1_})
            br.reindex_({bix2:bix2_})
    return peps
def trace_field(tn,tag,L,scale=True):
    for j in range(1,L-1):
        tn.contract_tags((tag.format(j),tag.format(j+1)),which='any',inplace=True)
        if scale:
            fac = np.amax(np.fabs(tn[tag.format(j)].data))
            tn[tag.format(j)].modify(data=tn[tag.format(j)].data/fac)
            tn.exponent += np.log10(fac)
    out = tn.contract()
    if scale:
        abs_out = np.fabs(out)
        return out/abs_out, np.log10(np.fabs(out)) + tn.exponent
    return out
def compress_wrapper(info,**compress_opts):
    tn,from_which,xrange,yrange = info
    tn.contract_boundary_from_(xrange,yrange,from_which,**compress_opts)
    return tn
def compress_row(tn,scale=True,**compress_opts):
    xrange1,xrange2 = (0,tn.Lx//2-1),(tn.Lx//2,tn.Lx-1)
    tn1 = tn.select([tn.row_tag(i) for i in range(tn.Lx//2)],which='any').copy()
    tn2 = tn.select([tn.row_tag(i) for i in range(tn.Lx//2,tn.Lx)],which='any').copy()
    info1 = tn1,'bottom',xrange1,(0,tn.Ly-1)
    info2 = tn2,'top',   xrange2,(0,tn.Ly-1)
    if scale:
        compress_opts['equalize_norms'] = 1.
    tn1,tn2 = parallelized_looped_function(compress_wrapper,
                                           [info1,info2],[],compress_opts)
    tn1.add_tensor_network(tn2,check_collisions=True)
    return trace_field(tn1,tn._col_tag_id,tn.Ly,scale=scale)
def compress_col(tn,scale=True,**compress_opts):
    yrange1,yrange2 = (0,tn.Ly//2-1),(tn.Ly//2,tn.Ly-1)
    tn1 = tn.select([tn.col_tag(j) for j in range(tn.Ly//2)],which='any').copy()
    tn2 = tn.select([tn.col_tag(j) for j in range(tn.Ly//2,tn.Ly)],which='any').copy()
    info1 = tn1,'left', (0,tn.Lx-1),yrange1
    info2 = tn2,'right',(0,tn.Lx-1),yrange2
    if scale:
        compress_opts['equalize_norms'] = 1.
    tn1,tn2 = parallelized_looped_function(compress_wrapper,
                                           [info1,info2],[],compress_opts)
    tn1.add_tensor_network(tn2,check_collisions=True)
    return trace_field(tn1,tn._row_tag_id,tn.Lx,scale=scale)
def compress(tn,scale=True,**compress_opts):
    if scale:
        compress_opts['equalize_norms'] = 1.
    tn.contract_boundary_(**compress_opts)
    out = tn.contract()
    out = tn.contract()
    if scale:
        abs_out = np.fabs(out)
        return out/abs_out, np.log10(np.fabs(out)) + tn.exponent
    return out
#def get_atn(xs,tr,CP_tag='CP{},{}',X_tag= 'q{},{}',A_tag='A{},{}',
#                  M_tag='M{}',TR_tag='x{}',O_tag='$[0,1]$'):
#    N,k,d = xs.shape
#    CP = np.zeros((d,)*3)
#    for i in range(d):
#        CP[i,i,i] = 1.0
#    tn = qtn.TensorNetwork([])
#    cpv = 'cpv{},{}'
#    cph = 'cph{},{}'
#    av = 'av{},{}'
#    ah = 'ah{},{}'
#    mv = 'mv{}'
#    for i in range(k):
#        for j in range(N):
#            if i<k-1:
#                inds = cpv.format(i,j),cph.format(i,j),cpv.format(i+1,j)
#                tags = CP_tag.format(i,j)
#                tn.add_tensor(qtn.Tensor(CP,inds=inds,tags=tags))
#
#            data = np.ones((2,d))
#            data[1,:] = xs[j,i,:]
#            idx1 = ah.format(i,j) if j==0 else av.format(i,j)
#            idx2 = cpv.format(i,j) if i==k-1 else cph.format(i,j)
#            inds = idx1,idx2
#            tags = X_tag.format(i,j)
#            tn.add_tensor(qtn.Tensor(data,inds=inds,tags=tags))
#
#            if j>0:
#                inds = ah.format(i,j-1),av.format(i,j),ah.format(i,j)
#                tags = A_tag.format(i,j)
#                tn.add_tensor(qtn.Tensor(ADD,inds=inds,tags=tags))
#    for i in range(1,k):
#        idx1 = ah.format(i-1,N-1) if i==1 else mv.format(i-1)
#        inds = idx1,ah.format(i,N-1),mv.format(i)
#        tags = M_tag.format(i) 
#        tn.add_tensor(qtn.Tensor(CP2,inds=inds,tags=tags))
#    for j in range(N):
#        inds = (cpv.format(0,j),)
#        tags = TR_tag.format(j)
#        tn.add_tensor(qtn.Tensor(tr[j,:],inds=inds,tags=tags))
#    inds = (mv.format(k-1),)
#    tn.add_tensor(qtn.Tensor(data=np.array([0.0,1.0]),inds=inds,tags=O_tag)) 
#    return tn
#def get_tril(d,mode='random'):
#    assert mode in {'equal','random',0,1}
#    t = np.zeros((d,2,d+1))
#    t[0,0,0] = t[d-1,1,d] = 1.0
#    for k in range(1,d):
#        if isinstance(mode,int):
#            t[k-mode,mode,k] = 1.0
#        else:
#            eps = 0.5 if mode=='equal' else np.random.rand()*2.0-1.0
#            t[k,0,k] = eps
#            t[k-1,1,k] = 1.0-eps
#    return t
#def get_trir(d):
#    t = np.zeros((d,2,d+1))
#    t[0,0,0] = t[d-1,1,d] = 1.0
#    for k in range(1,d):
#        t[k,0,k] = t[k-1,1,k] = 1.0
#    return t
#def insert_triangles(peps,mode='random',ltag='L{},{}',rtag='R{},{}'):
#    assert mode in {'equal','random',0,1}
#    bmap = dict()
#    for i in range(peps.Lx-1):
#        IR = get_trir(i+2) 
#        for j in range(peps.Ly-1):
#            IL = get_tril(i+2,mode=mode) 
#            if i==0:
#                l0 = peps[peps.site_tag(i,j)]
#                r0 = peps[peps.site_tag(i,j+1)]
#                b0 = list(qtn.bonds(l0,r0))[0]
#            else:
#                l0 = peps[ltag.format(i,j)]
#                r0 = peps[rtag.format(i,j)]
#                b0 = bmap[j,j+1]
#            l1 = peps[peps.site_tag(i+1,j)]
#            r1 = peps[peps.site_tag(i+1,j+1)]
#            b1 = list(qtn.bonds(l1,r1))[0] 
#            b0_,b1_ = b0+'_',b1+'_'
#            bmap[j,j+1] = qtn.rand_uuid()
#
#            inds = b0,b1,bmap[j,j+1]
#            tags = ltag.format(i+1,j)
#            peps.add_tensor(qtn.Tensor(data=IL,inds=inds,tags=tags))
#            inds = b0_,b1_,bmap[j,j+1]
#            tags = rtag.format(i+1,j)
#            peps.add_tensor(qtn.Tensor(data=IR,inds=inds,tags=tags))
#            r0.reindex_({b0:b0_})
#            r1.reindex_({b1:b1_})
#    return peps
#def draw_atn_alt(tn,N,k,fname,hsp=0.5,vsp=0.5,
#             CP_tag='CP{},{}',X_tag= 'q{},{}',A_tag='A{},{}',
#             M_tag='M{}',TR_tag='x{}',O_tag='$[0,1]$',**kwargs):
#    # fix positions of qij,xj
#    fixer = dict()
#    color = 'copy','+','*'
#    for i in range(k):
#        for j in range(N):
#            if i<k-1:
#                tag = CP_tag.format(i,j)
#                tn[tag].modify(tags=color[0])
#            tag = X_tag.format(i,j)
#            fixer[tag] = (j*2+1)*hsp,i*2*vsp
#            if j>0:
#                tag = A_tag.format(i,j)
#                tn[tag].modify(tags=color[1])
#    for i in range(1,k):
#        tag = M_tag.format(i)
#        tn[tag].modify(tags=color[2])
#    for j in range(N):
#        tag = TR_tag.format(j)
#        fixer[tag] = j*2*hsp,-vsp
#    fixer[O_tag] = ((N-1)*2+2)*hsp,((k-1)*2+2)*vsp
#    fig = tn.draw(fix=fixer,color=color,return_fig=True,**kwargs)
#    fig.savefig(fname)
#    return
#def draw_atn(tn,N,k,fname,hsp=0.5,vsp=0.5,
#             CP_tag='CP{},{}',X_tag= 'q{},{}',A_tag='A{},{}',
#             M_tag='M{}',TR_tag='x{}',O_tag='$[0,1]$',**kwargs):
#    fixer = dict()
#    color = 'copy','q','+','*','x','$\delta_{i1}$'
#    for i in range(k):
#        for j in range(N):
#            if i<k-1:
#                tag = CP_tag.format(i,j)
#                fixer[tag] = j*2*hsp,i*2*vsp
#                tn[tag].add_tag(color[0])
#            tag = X_tag.format(i,j)
#            fixer[tag] = (j*2+1)*hsp,i*2*vsp
#            tn[tag].add_tag(color[1])
#            if j>0:
#                tag = A_tag.format(i,j)
#                fixer[tag] = (j*2+1)*hsp,(i*2+1)*vsp
#                tn[tag].add_tag(color[2])
#    for i in range(1,k):
#        tag = M_tag.format(i)
#        fixer[tag] = ((N-1)*2+2)*hsp,(i*2+1)*vsp
#        tn[tag].add_tag(color[3])
#    for j in range(N):
#        tag = TR_tag.format(j)
#        fixer[tag] = j*2*hsp,-vsp
#        tn[tag].add_tag(color[4])
#    tn[O_tag].modify(tags=color[-1])
#    fixer[color[-1]] = ((N-1)*2+2)*hsp,((k-1)*2+2)*vsp
#    fig = tn.draw(fix=fixer,color=color,return_fig=True,**kwargs)
#    fig.savefig(fname)
#    return
#def draw_peps(peps,fname,hsp=0.5,vsp=0.5,**kwargs): 
#    fixer = dict()
#    color = ('site',)
#    for i in range(peps.Lx):
#        for j in range(peps.Ly):
#            tag = peps.site_tag(i,j)
#            fixer[tag] = hsp*j,vsp*i
#            peps[tag].add_tag(color[0])
#    fig = peps.draw(fix=fixer,color=color,return_fig=True,**kwargs)
#    fig.savefig(fname)
#    return
#def draw_tpeps(peps,fname,hsp=0.5,vsp=0.5,ltag='L{},{}',rtag='R{},{}',**kwargs): 
#    fixer = dict()
#    site_hsep = 2*(peps.Lx-1)+1
#    site_vsep = 2
#    color = 'site','IL','Ir'
#    for i in range(peps.Lx):
#        for j in range(peps.Ly):
#            x_site = site_vsep*i
#            y_site = site_hsep*j
#            tag = peps.site_tag(i,j)
#            fixer[tag] = y_site*hsp,x_site*vsp
#            peps[tag].add_tag(color[0])
#            if i<peps.Lx-1 and j<peps.Ly-1:
#                x = x_site+1
#
#                y = y_site+(i+1)
#                tag = ltag.format(i+1,j)
#                fixer[tag] = y*hsp,x*vsp
#                peps[tag].add_tag(color[1])
#
#                y = y_site+site_hsep-(i+1)
#                tag = rtag.format(i+1,j)
#                fixer[tag] = y*hsp,x*vsp
#                peps[tag].add_tag(color[2])
#    fig = peps.draw(fix=fixer,color=color,return_fig=True,**kwargs)
#    fig.savefig(fname)
#    return
