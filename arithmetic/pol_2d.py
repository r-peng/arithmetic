import numpy as np
import quimb.tensor as qtn
import arithmetic.utils as utils
ADD = np.zeros((2,)*3)
ADD[0,0,0] = ADD[1,0,1] = ADD[0,1,1] = 1.0
CP2 = np.zeros((2,)*3)
CP2[0,0,0] = CP2[1,1,1] = 1.0
def get_atn(xs,tr,CP_tag='CP{},{}',X_tag= 'q{},{}',A_tag='A{},{}',
                  M_tag='M{}',TR_tag='x{}',O_tag='$[0,1]$'):
    N,k,d = xs.shape
    CP = np.zeros((d,)*3)
    for i in range(d):
        CP[i,i,i] = 1.0
    tn = qtn.TensorNetwork([])
    cpv = 'cpv{},{}'
    cph = 'cph{},{}'
    av = 'av{},{}'
    ah = 'ah{},{}'
    mv = 'mv{}'
    for i in range(k):
        for j in range(N):
            if i<k-1:
                inds = cpv.format(i,j),cph.format(i,j),cpv.format(i+1,j)
                tags = CP_tag.format(i,j)
                tn.add_tensor(qtn.Tensor(CP,inds=inds,tags=tags))

            data = np.ones((2,d))
            data[1,:] = xs[j,i,:]
            idx1 = ah.format(i,j) if j==0 else av.format(i,j)
            idx2 = cpv.format(i,j) if i==k-1 else cph.format(i,j)
            inds = idx1,idx2
            tags = X_tag.format(i,j)
            tn.add_tensor(qtn.Tensor(data,inds=inds,tags=tags))

            if j>0:
                inds = ah.format(i,j-1),av.format(i,j),ah.format(i,j)
                tags = A_tag.format(i,j)
                tn.add_tensor(qtn.Tensor(ADD,inds=inds,tags=tags))
    for i in range(1,k):
        idx1 = ah.format(i-1,N-1) if i==1 else mv.format(i-1)
        inds = idx1,ah.format(i,N-1),mv.format(i)
        tags = M_tag.format(i) 
        tn.add_tensor(qtn.Tensor(CP2,inds=inds,tags=tags))
    for j in range(N):
        inds = (cpv.format(0,j),)
        tags = TR_tag.format(j)
        tn.add_tensor(qtn.Tensor(tr[j,:],inds=inds,tags=tags))
    inds = (mv.format(k-1),)
    tn.add_tensor(qtn.Tensor(data=np.array([0.0,1.0]),inds=inds,tags=O_tag)) 
    return tn
def get_peps(xs):
    N,k,d = xs.shape
    CP = np.zeros((d,)*3)
    for i in range(d):
        CP[i,i,i] = 1.0
    arrays = []
    for i in range(k):
        rows = []
        for j in range(N):
            data = np.ones((2,d))
            data[1,:] = xs[j,i,:]
            if i>0:
                data = np.einsum('ir,rpq->ipq',data,CP)
            if j>0:
                data = np.einsum('i...,ijk->jk...',data,ADD)
            if j==N-1:
                data = data[:,1,...]
            rows.append(data)
        arrays.append(rows)
    return make_peps_with_legs(arrays)
def get_tril(d,mode='random'):
    assert mode in {'equal','random',0,1}
    t = np.zeros((d,2,d+1))
    t[0,0,0] = t[d-1,1,d] = 1.0
    for k in range(1,d):
        if isinstance(mode,int):
            t[k-mode,mode,k] = 1.0
        else:
            eps = 0.5 if mode=='equal' else np.random.rand()*2.0-1.0
            t[k,0,k] = eps
            t[k-1,1,k] = 1.0-eps
    return t
def get_trir(d):
    t = np.zeros((d,2,d+1))
    t[0,0,0] = t[d-1,1,d] = 1.0
    for k in range(1,d):
        t[k,0,k] = t[k-1,1,k] = 1.0
    return t
def insert_triangles(peps,mode='random',ltag='L{},{}',rtag='R{},{}'):
    assert mode in {'equal','random',0,1}
    bmap = dict()
    for i in range(peps.Lx-1):
        IR = get_trir(i+2) 
        for j in range(peps.Ly-1):
            IL = get_tril(i+2,mode=mode) 
            if i==0:
                l0 = peps[peps.site_tag(i,j)]
                r0 = peps[peps.site_tag(i,j+1)]
                b0 = list(qtn.bonds(l0,r0))[0]
            else:
                l0 = peps[ltag.format(i,j)]
                r0 = peps[rtag.format(i,j)]
                b0 = bmap[j,j+1]
            l1 = peps[peps.site_tag(i+1,j)]
            r1 = peps[peps.site_tag(i+1,j+1)]
            b1 = list(qtn.bonds(l1,r1))[0] 
            b0_,b1_ = b0+'_',b1+'_'
            bmap[j,j+1] = qtn.rand_uuid()

            inds = b0,b1,bmap[j,j+1]
            tags = ltag.format(i+1,j)
            peps.add_tensor(qtn.Tensor(data=IL,inds=inds,tags=tags))
            inds = b0_,b1_,bmap[j,j+1]
            tags = rtag.format(i+1,j)
            peps.add_tensor(qtn.Tensor(data=IR,inds=inds,tags=tags))
            r0.reindex_({b0:b0_})
            r1.reindex_({b1:b1_})
    return peps
def draw_atn_alt(tn,N,k,fname,hsp=0.5,vsp=0.5,
             CP_tag='CP{},{}',X_tag= 'q{},{}',A_tag='A{},{}',
             M_tag='M{}',TR_tag='x{}',O_tag='$[0,1]$',**kwargs):
    # fix positions of qij,xj
    fixer = dict()
    color = 'copy','+','*'
    for i in range(k):
        for j in range(N):
            if i<k-1:
                tag = CP_tag.format(i,j)
                tn[tag].modify(tags=color[0])
            tag = X_tag.format(i,j)
            fixer[tag] = (j*2+1)*hsp,i*2*vsp
            if j>0:
                tag = A_tag.format(i,j)
                tn[tag].modify(tags=color[1])
    for i in range(1,k):
        tag = M_tag.format(i)
        tn[tag].modify(tags=color[2])
    for j in range(N):
        tag = TR_tag.format(j)
        fixer[tag] = j*2*hsp,-vsp
    fixer[O_tag] = ((N-1)*2+2)*hsp,((k-1)*2+2)*vsp
    fig = tn.draw(fix=fixer,color=color,return_fig=True,**kwargs)
    fig.savefig(fname)
    return
def draw_atn(tn,N,k,fname,hsp=0.5,vsp=0.5,
             CP_tag='CP{},{}',X_tag= 'q{},{}',A_tag='A{},{}',
             M_tag='M{}',TR_tag='x{}',O_tag='$[0,1]$',**kwargs):
    fixer = dict()
    color = 'copy','q','+','*','x','$\delta_{i1}$'
    for i in range(k):
        for j in range(N):
            if i<k-1:
                tag = CP_tag.format(i,j)
                fixer[tag] = j*2*hsp,i*2*vsp
                tn[tag].add_tag(color[0])
            tag = X_tag.format(i,j)
            fixer[tag] = (j*2+1)*hsp,i*2*vsp
            tn[tag].add_tag(color[1])
            if j>0:
                tag = A_tag.format(i,j)
                fixer[tag] = (j*2+1)*hsp,(i*2+1)*vsp
                tn[tag].add_tag(color[2])
    for i in range(1,k):
        tag = M_tag.format(i)
        fixer[tag] = ((N-1)*2+2)*hsp,(i*2+1)*vsp
        tn[tag].add_tag(color[3])
    for j in range(N):
        tag = TR_tag.format(j)
        fixer[tag] = j*2*hsp,-vsp
        tn[tag].add_tag(color[4])
    tn[O_tag].modify(tags=color[-1])
    fixer[color[-1]] = ((N-1)*2+2)*hsp,((k-1)*2+2)*vsp
    fig = tn.draw(fix=fixer,color=color,return_fig=True,**kwargs)
    fig.savefig(fname)
    return
def draw_peps(peps,fname,hsp=0.5,vsp=0.5,**kwargs): 
    fixer = dict()
    color = ('site',)
    for i in range(peps.Lx):
        for j in range(peps.Ly):
            tag = peps.site_tag(i,j)
            fixer[tag] = hsp*j,vsp*i
            peps[tag].add_tag(color[0])
    fig = peps.draw(fix=fixer,color=color,return_fig=True,**kwargs)
    fig.savefig(fname)
    return
def draw_tpeps(peps,fname,hsp=0.5,vsp=0.5,ltag='L{},{}',rtag='R{},{}',**kwargs): 
    fixer = dict()
    site_hsep = 2*(peps.Lx-1)+1
    site_vsep = 2
    color = 'site','IL','Ir'
    for i in range(peps.Lx):
        for j in range(peps.Ly):
            x_site = site_vsep*i
            y_site = site_hsep*j
            tag = peps.site_tag(i,j)
            fixer[tag] = y_site*hsp,x_site*vsp
            peps[tag].add_tag(color[0])
            if i<peps.Lx-1 and j<peps.Ly-1:
                x = x_site+1

                y = y_site+(i+1)
                tag = ltag.format(i+1,j)
                fixer[tag] = y*hsp,x*vsp
                peps[tag].add_tag(color[1])

                y = y_site+site_hsep-(i+1)
                tag = rtag.format(i+1,j)
                fixer[tag] = y*hsp,x*vsp
                peps[tag].add_tag(color[2])
    fig = peps.draw(fix=fixer,color=color,return_fig=True,**kwargs)
    fig.savefig(fname)
    return
permute_1d = utils.permute_1d
contract_1d = utils.contract_1d
make_peps_with_legs = utils.make_peps_with_legs
trace_open = utils.trace_open
contract_from_bottom = utils.contract_from_bottom
contract = utils.contract
quad = utils.quad
