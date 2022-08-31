import numpy as np
norb = 6
na = norb // 2
nb = na

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
w,v = np.linalg.eigh(h1)
print('E0=',sum(w[:na])*2)
eri = np.zeros((norb,)*4)
eri[idot,idot,idot,idot] = U

class Model:
    def __init__(self):
        self.h1 = h1
        self.eri = eri
        self.ll = idot
        self.lr = norb - self.ll - 1
        self.no = na
    def get_tmatS(self):
        return self.h1
    def get_vmatS(self):
        return np.zeros_like(self.h1)
    def get_umatS(self):
        return self.eri
model = Model()

from pyscf.cc import td_roccd
eris = td_roccd.ERIs_SIAM(model,mo_energy=w,mo_coeff=v)
no, nv = na, norb-na
t = np.zeros((nv,nv,no,no))
l = np.zeros((no,no,nv,nv))
dt = 0.1
nsteps = 10
rdm1,_ = td_roccd.kernel(eris,t,l,dt*nsteps,dt,RK=4)
rdm1 = np.einsum('npq,ip,jq->nij',rdm1,v,v)
tr = np.einsum('njj->n',rdm1)
for i in range(rdm1.shape[0]):
    print(i,2*rdm1[i,idot,idot],2*tr[i])
