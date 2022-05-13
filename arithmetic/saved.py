
def get_B(v0,vg,tau,xs,max_bond=None,cutoff=1e-15,equalize_norms=True):
    norb,_,nf = vg.shape
    ng = len(xs)
    sqrt_tau = np.sqrt(tau)
    tn = qtn.TensorNetwork([])
    CP = np.zeros((norb,)*3)
    for i in range(norb):
        CP[i,i,i] = 1.

    data = np.ones((norb,norb,2))
    data[:,:,1] = -tau*v0
    inds = [qtn.rand_uuid() for i in range(3)]
    tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags={'T0','v0'}))
    xix = []
    for g in range(nf):
        data = np.ones((norb,norb,ng,2),dtype=vg.dtype)
        for i in range(ng):
            data[:,:,i,1] = sqrt_tau*xs[i]*vg[:,:,g]
        inds = [qtn.rand_uuid() for i in range(4)]
        xix.append(inds[2])
        tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags={f'T{g+1}',f'v{g+1}'}))

    pix = [qtn.rand_uuid() for i in range(nf)]
    for g in range(nf):
        rix = tn[f'v{g+1}'].inds[0] if g==nf-1 else pix[g+1]
        inds = pix[g],rix,tn[f'v{g}'].inds[0]
        tn.add_tensor(qtn.Tensor(data=CP,inds=inds,tags={f'T{g}',f'p{g}'}))
    qix = [qtn.rand_uuid() for i in range(nf)]
    for g in range(nf):
        rix = tn[f'v{g+1}'].inds[1] if g==nf-1 else qix[g+1]
        inds = qix[g],rix,tn[f'v{g}'].inds[1]
        tn.add_tensor(qtn.Tensor(data=CP,inds=inds,tags={f'T{g}',f'q{g}'}))
    iix = [qtn.rand_uuid() for i in range(nf)]
    for g in range(nf):
        rix = tn[f'v{g+1}'].inds[-1] if g==nf-1 else iix[g+1]
        inds = rix,tn[f'v{g}'].inds[-1],iix[g]
        tn.add_tensor(qtn.Tensor(data=ADD,inds=inds,tags={f'T{g}',f'a{g}'}))
    print(tn)

#    for g in range(nf):
#        tn.contract_tags(f'T{g}',which='any',inplace=True)
#    tn.fuse_multibonds_()
#
#    # canonize from right
#    for g in range(nf,0,-1):
#        tn.canonize_between(f'T{g-1}',f'T{g}',absorb='left')
#    # compress from right
#    for g in range(nf):
#        tn.compress_between(f'T{g}',f'T{g+1}',absorb='right',
#                            max_bond=max_bond,cutoff=cutoff)
#    # canonize from left
#    for g in range(nf):
#        tn.canonize_between(f'T{g}',f'T{g+1}',absorb='right')
#    # compress from right
#    for g in range(nf,0,-1):
#        tn.compress_between(f'T{g-1}',f'T{g}',absorb='left',
#                            max_bond=max_bond,cutoff=cutoff)
    return tn,pix[0],qix[0],iix[0],xix 
def _get_B(v0,vg,tau,xs,step):
    norb,_,nf = vg.shape
    ng = len(xs)
    sqrt_tau = np.sqrt(tau)
    tn = qtn.TensorNetwork([])
    idxs = [qtn.rand_uuid() for g in range(nf+1)]
    sdelta = np.zeros((2,)*3)
    sdelta[0,0,0] = 1./norb
    sdelta[1,1,1] = 1.
    for g in range(nf):
        data = np.ones((norb,norb,ng,2),dtype=vg.dtype)
        for i in range(ng):
            data[:,:,i,1] = scipy.linalg.expm(sqrt_tau*xs[i]*vg[:,:,g])
        if g == nf-1:
            data = np.einsum('pq,qr...->pr...',scipy.linalg.expm(-tau*v0),data)
        inds = idxs[g+1],idxs[g],f's{step}_x{g}',f's{step}_i{g}'
        tags = f's{step}',f'x{g}'
        tn.add_tensor(qtn.Tensor(data=data,inds=inds,tags=tags))
        inds = idxs[g+1]+'_',idxs[g]+'_',f's{step}_i{g}'
        tags = f's{step}',f'del{g}'
        tn.add_tensor(qtn.Tensor(data=sdelta,inds=inds,tags=tags))
    return tn,idx[-1],idxs[0]
if __name__=='__main__':
    norb = 3
    t = 1.
    u = 1.

    h1 = np.zeros((norb,)*2)
    for i in range(norb):
        if i-1>0:
            h1[i,i-1] = -t
        if i+1<norb:
            h1[i,i+1] = -t
    eri = np.zeros((norb,)*4)
    for i in range(norb):
        eri[i,i,i,i] = u
    v0,vg = get_Hmc(h1,eri)
    # check exponent
    ng = 4
    xs = np.random.rand(ng)
    tau = 1.
    step = 0
    tn,pix,qix,iix,xix = get_B(v0,vg,tau,xs)
    print(tn)
    idxs = [np.random.randint(low=0,high=ng) for g in range(len(xix))]
    for g in range(len(xix)):
        data = np.zeros(ng)
        data[idxs[g]] = 1.
        tn.add_tensor(qtn.Tensor(data=data,inds=(xix[g],)))
    out1 = tn.contract(output_inds=(pix,qix,iix))
    out1 *= 10*tn.exponent

    CP = np.zeros((norb,)*3)
    for i in range(norb):
        CP[i,i,i] = 1.
    data = np.ones((norb,norb,2))
    data[:,:,1] = -tau*v0
    out3 = np.einsum('pqi,pxy,quv,ijk->xukyvj',data,CP,CP,ADD)
    for g in range(vg.shape[-1]-1):
        data = np.ones((norb,norb,2),dtype=vg.dtype)
        data[:,:,1] = np.sqrt(tau)*xs[idxs[g]]*vg[:,:,g]
        data = np.einsum('pqi,pxy,quv,ijk->xukyvj',data,CP,CP,ADD)
        out3 = np.einsum('xukXUK,XUKyvj->xukyvj',out3,data)
    data = np.ones((norb,norb,2),dtype=vg.dtype)
    data[:,:,1] = np.sqrt(tau)*xs[idxs[-1]]*vg[:,:,-1]
    out3 = np.einsum('xukXUK,XUK->xuk',out3,data)
    print(out3[:,:,0])

    out2 = np.array(-tau*v0,dtype=vg.dtype)
    for g in range(vg.shape[-1]):
        out2 += np.sqrt(tau)*xs[idxs[g]]*vg[:,:,g]
#    print(out1.data[:,:,0])
    print('check sum=',np.linalg.norm(out2-out3[:,:,1]))
    print('check sum=',np.linalg.norm(out2-out1.data[:,:,1]))
    print('check sum=',np.linalg.norm(np.ones((norb,)*2)-out1.data[:,:,0]))
