import numpy as np
from arithmetic.mbpt import (
    load_diagrams,
    distribute_diagrams,
    generate_property,
    get_tn,
)
np.set_printoptions(precision=6,suppress=True,linewidth=1000)
norb = 6
na = 2
nb = na
occ_vec = np.array([1]*na + [0]*(norb-na))
occ_vec_ = np.concatenate([occ_vec,occ_vec])

t_leads = 1.
t_hyb = .4
U = 1.
idot = norb // 2

h1 = np.zeros((norb,norb))
for i in range(norb):
    if i < norb - 1:
        dot = (i == idot or i + 1 == idot)
        h1[i, i + 1] = -t_hyb if dot else -t_leads
    if i > 0:
        dot = (i == idot or i - 1 == idot)
        h1[i, i - 1] = -t_hyb if dot else -t_leads
e,v = np.linalg.eigh(h1)
print('E0=',sum(e[:na])*2)
e_ = np.concatenate([e,e])
print(e_)
print(occ_vec_)

eri = np.zeros((norb,)*4)
eri[idot,idot,idot,idot] = U
eri = np.einsum('ijkl,ip,jq,kr,ls->pqrs',eri,v,v,v,v)
eri_aa = eri - eri.transpose(1,0,2,3)
eri_ab = eri.copy()
#print(np.linalg.norm(eri_aa+eri_aa.transpose(1,0,2,3)))
#print(np.linalg.norm(eri_ab-eri_ab.transpose(1,0,3,2)))
#ov = np.einsum('ijcb,cbaj->ia',eri_aa[:na,:na,na:,na:],eri_aa[na:,na:,na:,:na])
#ov += 2*np.einsum('ijcb,cbaj->ia',eri_ab[:na,:na,na:,na:],eri_ab[na:,na:,na:,:na])
#ov -= np.einsum('kjab,ibkj->ia',eri_aa[:na,:na,na:,na:],eri_aa[:na,na:,:na,:na])
#ov -= 2*np.einsum('kjab,ibkj->ia',eri_ab[:na,:na,na:,na:],eri_ab[:na,na:,:na,:na])
#vv = np.einsum('ikac,bcik->ba',eri_aa[:na,:na,na:,na:],eri_aa[na:,na:,:na,:na])
#vv += 2*np.einsum('ikac,bcik->ba',eri_ab[:na,:na,na:,na:],eri_ab[na:,na:,:na,:na])
#oo = -np.einsum('ikac,acjk->ij',eri_aa[:na,:na,na:,na:],eri_aa[na:,na:,:na,:na])
#oo -= 2*np.einsum('ikac,acjk->ij',eri_ab[:na,:na,na:,na:],eri_ab[na:,na:,:na,:na])
#rdm1 = np.zeros((norb,norb))
#rdm1[:na,:na] = oo
#rdm1[na:,na:] = vv
#rdm1[:na,na:] = ov
#rdm1[na:,:na] = ov.T
#print(rdm1)
eri_ = np.zeros((norb*2,)*4)
eri_[:norb,:norb,:norb,:norb] = eri_aa.copy()
eri_[norb:,norb:,norb:,norb:] = eri_aa.copy()
eri_[:norb,norb:,:norb,norb:] = eri_ab.copy()
eri_[norb:,:norb,norb:,:norb] = eri_ab.copy()
eri_[:norb,norb:,norb:,:norb] = -eri_ab.transpose(1,0,2,3).copy()
eri_[norb:,:norb,:norb,norb:] = -eri_ab.transpose(1,0,2,3).copy()
print('check eri symmetry=',np.linalg.norm(eri_+eri_.transpose(1,0,2,3)))

tf = 1.
dt = .1
t = np.arange(0,tf,dt)
ng = len(t)
w = np.ones(ng)*dt

rdm1_ = np.zeros(norb,dtype=complex)
rdm1_[:na] = 1.
rdm1_ = np.diag(rdm1_)
rdm1_ = np.stack([rdm1_]*ng,axis=0)
for n in [3,4]:
    diags = load_diagrams(n)
    hfc_map = distribute_diagrams(diags)
    print('##################### '+f'order={n-1},ndiag={len(hfc_map[1])}'+' #####################')
    for ix,diag_ in enumerate(hfc_map[1]):
        diag = generate_property(diag_)
        print(ix,diag.idx)
        tn,output_inds = get_tn(diag,e_,occ_vec_,eri_,to=t,ti=t,w=w)
        tn.exponent -= sum([np.log10(float(i)) for i in range(2,n)])
        tn.exponent -= np.log10(2.) * diag.ne
        print(tn)
        #print(output_inds)
        #exit()
        #tn.full_simplify_(seq='ADCRSL',output_inds=output_inds,equalize_norms=1.,progbar=True)
        #print(tn)
        ig = tn.contract(output_inds=output_inds).data
        aabb = np.linalg.norm(ig[:norb,:,:,:norb,:,:]-ig[norb:,:,:,norb:,:,:])
        ab = np.linalg.norm(ig[:norb,:,:,norb:,:,:])
        ba = np.linalg.norm(ig[norb:,:,:,:norb,:,:])
        print('check ab=',aabb,ab,ba)
        g = -1j*ig[:norb,:,:,:norb,:,:]
        symm0 = np.linalg.norm(g[:,:,0,:,:,0]+g[:,:,1,:,:,1]-g[:,:,0,:,:,1]-g[:,:,1,:,:,0])
        symm1 = np.linalg.norm(g[:,:,0,:,:,0]+g[:,:,1,:,:,1].transpose(2,3,0,1).conj())
        symm2 = np.linalg.norm(g[:,:,0,:,:,1]+g[:,:,0,:,:,1].transpose(2,3,0,1).conj())
        symm3 = np.linalg.norm(g[:,:,1,:,:,0]+g[:,:,1,:,:,0].transpose(2,3,0,1).conj())
        print('check g symm=',symm0,symm1,symm2,symm3)
        rdm1 = -1j*np.einsum('piqi->ipq',g[:,:,0,:,:,1])
        print('check rdm1 symm=',np.linalg.norm(rdm1-rdm1.transpose(0,2,1).conj()))
        rdm1_ += rdm1 * 10**tn.exponent * diag.sign
tr = np.einsum('nii->',rdm1_[...])
for i in range(ng):
    print(i,rdm1_[i,idot,idot],tr[i])
