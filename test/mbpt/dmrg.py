import numpy as np
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP
from pyblock3.algebra.mpe import MPE
from pyblock3.symbolic.expr import OpElement, OpNames
from pyblock3.algebra.symmetry import SZ
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
rdm = np.einsum('pi,qi->pq',v[:,:na],v[:,:na])
for i in range(norb):
    print(i,rdm[i,i]*2)
def build_qc(u=2, t_leads=1, t_hyb=1, idot=None, n=8, cutoff=1E-9):
    fcidump = FCIDUMP(pg='c1', n_sites=n, n_elec=n, twos=0, ipg=0, orb_sym=[0] * n)
    hamil = Hamiltonian(fcidump, flat=False)
    idot = n // 2 if idot is None else idot
    def generate_terms(n_sites, c, d):
        for i in range(0, n_sites):
            if i<norb-1:
                dot = (i == idot or i + 1 == idot)
                fac = -t_hyb if dot else -t_leads
                for s in [0, 1]:
                    yield fac * c[i, s] * d[i + 1, s]
            if i > 0:
                dot = (i == idot or i - 1 == idot)
                fac = -t_hyb if dot else -t_leads
                for s in [0, 1]:
                    yield fac * c[i, s] * d[i - 1, s]
        if abs(u)>0:
            yield u * (c[idot, 0] * c[idot, 1] * d[idot, 1] * d[idot, 0])
    return hamil, hamil.build_mpo(generate_terms, cutoff=cutoff).to_sparse()

hamil, mpo = build_qc(u=0,t_leads=t_leads,t_hyb=t_hyb, idot=idot, n=norb)
mpo, _ = mpo.compress(left=True, cutoff=1E-9, norm_cutoff=1E-9)
mpo = mpo.to_flat()
bdim = 500
mps = hamil.build_mps(bdim)
mps = mps.to_flat()
noises = [1E-4, 1E-5, 1E-6, 0]
davthrds = None
dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bdim], noises=noises,
                            dav_thrds=davthrds, iprint=0, n_sweeps=20, tol=1E-12)
for i in range(norb):
    dop = OpElement(OpNames.D, (i, 0), q_label=SZ(-1, -1, hamil.orb_sym[i]))
    di = hamil.build_site_mpo(dop)
    di = di.to_flat()
    n_expt = np.dot(mps.conj(), mps)
    q = 2 * np.dot((di @ mps).conj(), di @ mps)
    print(f'i={i},norm={n_expt},q={q/n_expt}')
hamil, mpo = build_qc(u=U,t_leads=t_leads,t_hyb=t_hyb,idot=idot,n=norb)
mpo, _ = mpo.compress(left=True, cutoff=1E-9, norm_cutoff=1E-9)
mpo = mpo.to_flat()

isite = idot
dop = OpElement(OpNames.D, (isite, 0), q_label=SZ(-1, -1, hamil.orb_sym[isite]))
di = hamil.build_site_mpo(dop)
di = di.to_flat()

mpo.const -= dmrg.energies[-1]
mpe = MPE(mps,mpo,mps)
dt = 0.1
n_steps = 10
for it in range(n_steps):
    cur_t = (it + 1) * dt
    te = mpe.tddmrg(bdims=[bdim], dt=-dt*1j, iprint=0, n_sweeps=1, n_sub_sweeps=2, normalize=False)
    n_expt = np.dot(mps.conj(), mps)
    q = 2 * np.dot((di @ mps).conj(), di @ mps)
    print(f't={cur_t},norm={n_expt},q={q/n_expt}')

