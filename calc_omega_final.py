import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.special import expit
import matplotlib.colors as mcol
import sys
import os

# Read the MLWF cell map, weights and Hamiltonian:    
print('Reading Wannier Data')
cellMapCart = np.loadtxt(f"wannier.mlwfCellMap", usecols=[3,4,5])
cellMap = np.loadtxt("wannier.mlwfCellMap")[:,0:3].astype(int)
Wwannier = np.fromfile("wannier.mlwfCellWeights")
nCells = cellMap.shape[0]
nBands = int(np.sqrt(Wwannier.shape[0] / nCells))
Wwannier = Wwannier.reshape((nCells,nBands,nBands)).swapaxes(1,2)

# --- Get cell volume, mu and k-point folding from totalE.out:
for line in open('totalE.out'):
    if line.startswith("unit cell volume"):
        Omega = float(line.split()[-1])
    if line.startswith('\tFillingsUpdate:'):
        mu = float(line.split()[2])
    if line.startswith('kpoint-folding'):
        kfold = np.array([int(tok) for tok in line.split()[1:4]])
kfoldProd = np.prod(kfold)
kStride = np.array([kfold[1]*kfold[2], kfold[2], 1])

# --- Read reduced Wannier Hamiltonian, momenta and expand them:
Hreduced = np.fromfile("wannier.mlwfH",dtype=np.complex128).reshape((kfoldProd,nBands,nBands)).swapaxes(1,2)
Preduced = np.fromfile("wannier.mlwfP",dtype=np.complex128).reshape((kfoldProd,3,nBands,nBands)).swapaxes(2,3)
Sreduced = np.fromfile("wannier.mlwfS",dtype=np.complex128).reshape((kfoldProd,3,nBands,nBands)).swapaxes(2,3)
RPreduced =  np.fromfile(f"wannier.mlwfRP", dtype=np.complex128).reshape((kfoldProd,3,3,nBands,nBands)).swapaxes(3,4)

iReduced = np.dot(np.mod(cellMap, kfold[None,:]), kStride)
Hwannier = Wwannier * Hreduced[iReduced]
Pwannier = Wwannier[:,None] * Preduced[iReduced]
Swannier = Wwannier[:,None] * Sreduced[iReduced]
RPwannier = -1j * Wwannier[:,None,None] * RPreduced[iReduced]

print('Reading Phonon Data')
# Read phonon dispersion relation:
cellMapPh = np.loadtxt('totalE.phononCellMap', usecols=[0,1,2]).astype(int)
nCellsPh = cellMapPh.shape[0]
omegaSqR = np.fromfile('totalE.phononOmegaSq')  # just a list of numbers
nModes = int(np.sqrt(omegaSqR.shape[0] // nCellsPh))
omegaSqR = omegaSqR.reshape((nCellsPh, nModes, nModes)).swapaxes(1,2)

## Load atomic mass to convert from Hsub to Fsub
phonon_basis = np.loadtxt('./totalE.phononBasis',skiprows=1,usecols=[2,3,4,5])
amu = 1822.8884862173
mass = phonon_basis[:,-1] * amu

#np.tensordot(phasePrime, Hwannier, axes=1) Read e-ph matrix elements
cellMapEph = np.loadtxt('wannier.mlwfCellMapPh', usecols=[0,1,2]).astype(int)
nCellsEph = cellMapEph.shape[0]
# --- Get phonon supercell from phonon.out:
for line in open('phonon.out'):
    tokens = line.split()
    if len(tokens)==5:
        if tokens[0]=='supercell' and tokens[4]=='\\':
            phononSup = np.array([int(token) for token in tokens[1:4]])
prodPhononSup = np.prod(phononSup)
phononSupStride = np.array([phononSup[1]*phononSup[2], phononSup[2], 1])
# --- Read e-ph cell weights:
nAtoms = nModes // 3
cellWeightsEph = np.fromfile("wannier.mlwfCellWeightsPh").reshape((nCellsEph,nBands,nAtoms)).swapaxes(1,2)
cellWeightsEph = np.repeat(cellWeightsEph.reshape((nCellsEph,nAtoms,1,nBands)), 3, axis=2)  # repeat atom weights for 3 directions
cellWeightsEph = cellWeightsEph.reshape((nCellsEph,nModes,nBands))  # coombine nAtoms x 3 into single dimension: nModes
# --- Read, reshape and expand e-ph matrix elements:
iReducedEph = np.dot(np.mod(cellMapEph, phononSup[None,:]), phononSupStride)
HePhReduced = np.fromfile('wannier.mlwfHePh',dtype=np.complex128).reshape((prodPhononSup,prodPhononSup,nModes,nBands,nBands)).swapaxes(3,4)
HePhWannier = cellWeightsEph[:,None,:,:,None] * cellWeightsEph[None,:,:,None,:] * HePhReduced[iReducedEph][:,iReducedEph]

# Calculate energies, eigenvectors and velocities for given k
def calcE(k):
    # Fourier transform to k:
    phase = np.exp((2j*np.pi)*np.dot(k,cellMap.T))
    H = np.tensordot(phase, Hwannier, axes=1)
    P = np.tensordot(phase, Pwannier,  axes=1)
    # Diagonalize and switch to eigen-basis:
    E,U = np.linalg.eigh(H)  # Diagonalize
    v = np.imag(np.einsum(
        'kba, kibc, kca -> kai', U.conj(), P, U, optimize="optimal"
    ))  # diagonal only
    return E, U, v

def calcESL(k):
    # Fourier transform to k:
    phase = np.exp((2j*np.pi)*np.dot(k,cellMap.T))
    H = np.tensordot(phase, Hwannier, axes=1)
    Sk = np.tensordot(phase, Swannier,  axes=1)
    Pk = np.tensordot(phase, Pwannier, axes=1)

    # Diagonalize and switch to eigen-basis:
    E,U = np.linalg.eigh(H)  # Diagonalize

    S = np.einsum(
        'kba, kibc, kcd -> kiad', U.conj(), Sk, U, optimize="optimal"
    )
    Pk = np.einsum(
        'kba,kibc, kcd -> kiad', U.conj(), Pk, U, optimize="optimal")
    RPk = np.einsum("kba,kijbd->kijad", U.conj(), np.einsum("kijbc,kcd->kijbd", np.tensordot(phase, RPwannier, axes=1), U))

    phasePrime = phase[:, None, :] * (1j * cellMapCart.T)[None, :, :]
    Hpk = np.tensordot(phasePrime, Hwannier, axes=1)
    Hp = np.einsum(
        'kba,kibd->kiad', U.conj(), np.einsum("kibc,kcd->kibd", Hpk, U)
    )
    Dk = Hp * (-getInvEdiff(E)[:, None])
    RPk += 1j * np.einsum("kiab,kjbc->kijac", Dk, Pk)
    RPk = 0.5*(RPk + RPk.conjugate().swapaxes(3,4))
    L = np.einsum('hij, kijab -> khab', leviCivita(), RPk)
    
    return E, U, S, L


# Calculate phonon energies and eigenvectors for given q
def calcPh(q):
    phase = np.exp((2j*np.pi)*np.tensordot(q, cellMapPh.T, axes=1))
    D = np.tensordot(phase, omegaSqR, axes=1)
    D = 0.5 * (D + D.swapaxes(-1,-2).conj())  # enforce Hermitian explicitly
    omegaSq, U = np.linalg.eigh(D)
    omegaPh = np.sqrt(np.maximum(omegaSq, 0.))
    #U = fix_eigenvector_phase(U)
    return omegaPh, U

def calcFph_single(k1, q):
    # Generate k-grid from q
    k2 = k1 - q
    # Electrons:
    E1, U1, v1 = calcE(k1)
    E2, U2, v2 = calcE(k2)
    nk = E1.shape[0]
    # Phonons for all pairs pf k1 - k2:
    omegaPh, Uph = calcPh(q)
    # E-ph matrix elements for all pairs of k1 - k2:
    phase1 = np.exp((2j*np.pi)*np.dot(k1,cellMapEph.T))
    phase2 = np.exp((2j*np.pi)*np.dot(k2,cellMapEph.T))
    Hsub = np.einsum(
        'kac,      kbd,   kr,      kR,     rRxab -> kxcd',
        U1.conj(), U2, phase1.conj(), phase2, HePhWannier,
        optimize='optimal'
    )
    ## Convert to F from Hsub
    F = Hsub * np.sqrt(mass[None,:,None,None])
    return F, omegaPh, Uph, E1, E2

def compute_ph_amplitude_q(wuq, omega_min=1.0E-6):
    A_ql = np.zeros(wuq.shape[0])
    iql = 0
    zmsk = wuq > omega_min
    A_ql[zmsk] = np.sqrt(1.0 / (wuq[zmsk]))
    return A_ql

def leviCivita():
    """Get Levi-Civita tensor"""
    eps = np.zeros((3, 3, 3)) 
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = +1. 
    eps[2, 1, 0] = eps[0, 2, 1] = eps[1, 0, 2] = -1. 
    return eps 

def getInvEdiff(Ek, degenThreshold=1E-6):
    """Get 1/(Ei - Ej), with 0 within degenerate subspaces.
    The band dimension is assumed to be last in input.
    At output, i is the penultimate index and j is the final index."""
    EkDiff = Ek[..., None] - Ek[..., None, :]
    EkInv = np.zeros_like(EkDiff)
    np.reciprocal(EkDiff, out=EkInv, where=(np.abs(EkDiff) > degenThreshold))
    return EkInv

def get_Econ_Inv(Ek1, Ek2, wq, degenThreshold=1E-6):
    """
    Get 1/(Ei - Ej - w), with 0 within degenerate subspaces.
    return EkInv with index [Ei, Ej, w]
    """
    EkDiff = Ek1[...,None,None] - Ek2[...,None,:,None] - wq[...,None,:] 
    EkInv = np.zeros_like(EkDiff)
    np.reciprocal(EkDiff, out=EkInv, where=(np.abs(EkDiff) > degenThreshold))
    return EkInv

def compute_rho_Bpert(energies, S, L, B_field, T=300, Ef=-0.2, Sonly=False):
    g_e = 2.0023193043625635
    kB = 1./3.157750E5 # Ha/K
    
    nk, nb = energies.shape
    rho = np.zeros((nk, nb, nb), dtype=np.complex128)
    
    B_au = np.array(B_field) / 2.35051757077E5
    if Sonly:
        H1 = 0.5 * g_e * np.einsum('i, kiab -> kab', B_au, S)
    else:
        gSL= g_e * S + L 
        H1 = 0.5 * np.einsum('i, kiab -> kab', B_au, gSL)
    for ik in range(nk): 
        eps = energies[ik]
        
        # Fermi-Dirac occupations
        f = expit((Ef - eps) / (kB*T)) 
        rho[ik] = np.diag(f)
        
        for a in range(nb):
            for b in range(nb):
                de = eps[a] - eps[b]
                if abs(de) < 1e-6:
                    
                    fave = 0.5 * (f[a] + f[b])
                    dfde = -fave * (1.0 - fave) / (kB*T)
                else:
                    dfde = (f[a] - f[b]) / de
                rho[ik, a,b] += H1[ik,a,b] * dfde
    return rho

def make_kgrid_2D(nk):
    k1 = []
    klist = np.linspace(-0.5, 0.5, nk)
    for i in klist:
        for j in klist:
            #for k in klist:
            k1.append([i,j,0.0])
    k1 = np.array(k1)
    return k1
    
eps = leviCivita()

def compute_Omega_q(q, kpts, nmodes, nat = 3, 
                    min_freq= 1.0E-6, 
                    B = np.array([0.0,0.0,0.0]),
                    Ef = -0.21457386440278595,
                    reduce = True,
                    nVal = 13,
                    T = 300):

    nk = kpts.shape[0]

    ## F wtih index [k1, mode, e1, e2]
    f_ephF, omega_q, Wq, en1F, en2F = calcFph_single(np.array(kpts), q) ## get F for +q
    f_ephF_m, omega_qm, Wqm, en1F_m, en2F_m = calcFph_single(np.array(kpts), -q) ## get F for -q
    
    if reduce:
        f_ephF = f_ephF[...,nVal-1:nVal+3,nVal-1:nVal+3]
        en1F = en1F[:,nVal-1:nVal+3]
        en2F = en2F[:,nVal-1:nVal+3]
        f_ephF_m = f_ephF_m[...,nVal-1:nVal+3,nVal-1:nVal+3]
        en1F_m = en1F_m[:,nVal-1:nVal+3]
        en2F_m = en2F_m[:,nVal-1:nVal+3]    

    nb = en1F.shape[-1]
    
    ## Get Energy Spin and OAM
    E, _, S, L = calcESL(kpts)

    if reduce:
        E = E[:,nVal-1:nVal+3]
        S = S[:,:,nVal-1:nVal+3,nVal-1:nVal+3]
        L = L[:,:,nVal-1:nVal+3,nVal-1:nVal+3]

    assert np.all(np.isclose(en1F, E))
    assert np.all(np.isclose(en1F, en1F_m))

    rho = compute_rho_Bpert(E, S, L, B, T=T, Ef=-0.2)

    # Allocate Omega Out
    Om_out = np.zeros((nmodes,nmodes), dtype=np.complex128)

    for ik, k1 in enumerate(kpts):

        G_p = np.zeros((nmodes,nmodes,nb,nb), dtype=np.complex128)
        G_m = np.zeros((nmodes,nmodes,nb,nb), dtype=np.complex128)

        ## Precompute combinations of energy conservation
        econ = get_Econ_Inv(en1F[ik], en2F[ik], omega_q) # Returns 1/(e1-e2-w) with index [e1,e2,w]
        econ_p = get_Econ_Inv(en1F[ik], en2F[ik], -omega_q) # Returns 1/(e1-e2+w) with index [e1,e2,w]
        econ_sum = econ[:,None,...,None] + econ[None,...,None,:] # precompute sum of inverses
        econ_p_sum = econ_p[:,None,...,None] + econ_p[None,...,None,:] # precompute sum of inverses [e1, e2, w_nu, w_mu]
        
        # Get single k-point
        ## einsum Reshape to atom, cartesian direction, band, band
        f_eph = f_ephF[ik]
        f_eph_m = f_ephF_m[ik]
        f_shape = f_eph.shape

        f_eph = f_eph.reshape(nat,3,nb,nb)
        f_eph_m = f_eph_m.reshape(nat,3,nb,nb)

        Km_q = np.einsum('IJK, kInb, kJmb, nmbvu -> kKnmvu',
                       eps, f_eph, f_eph.conj(), econ_sum)
        Kp_mq = np.einsum('IJK, kInb, kJmb, nmbuv -> kKnmuv',
                       eps, f_eph_m.conj(), f_eph_m, econ_p_sum)

        #Km_q = np.einsum('IJK, kInb, kJbm, nbvu -> kKnmvu',
        #               eps, f_eph, f_eph_m, econ_sum) +
        #       np.einsum('IJK, kInb, kJbm, mbvu -> kKnmvu',
        #               eps, f_eph, f_eph_m, econ_sum)

        #Kp_mq = np.einsum('IJK, kIbn, kJmb, mbvu -> kKnmvu',
        #               eps, f_eph, f_eph_m, econ_p_sum) +
        #        np.einsum('IJK, kIbn, kJmb, nbvu -> kKnmvu',
        #               eps, f_eph, f_eph_m, econ_p_sum)

        #print(omega_qm.shape, omega_q.shape, Wq.shape)
        A_q = compute_ph_amplitude_q(omega_q)
        A_qm = compute_ph_amplitude_q(omega_qm)
        for l1 in range(nmodes):
            Wql1 = Wq[:,l1].reshape(-1,3)
            Wqml1 = Wqm[:,l1].reshape(-1,3)
            for l2 in range(nmodes):
                #if (omega_qm[l2] > min_freq) and (omega_q[l1] > min_freq):
                Wql2 = Wq[:,l2].reshape(-1,3)
                Wqml2 = Wqm[:,l2].reshape(-1,3)
                #prefrac_p = -0.5j * np.sqrt(omega_q[l1] / omega_q[l2]) * A_q[l1] * A_q[l2]
                #prefrac_m = -0.5j * np.sqrt(omega_qm[l1] / omega_qm[l2]) * A_qm[l1] * A_qm[l2]
                prefrac_p = -0.5j * A_q[l1] * A_q[l2]
                prefrac_m = -0.5j * A_qm[l1] * A_qm[l2]
                for k in range(nat):
                    #Mph = -1j*np.sqrt(omega_qm[l2] / omega_q[l1])*np.cross(Wql1[k,:], Wqml2[k,:])
                    #Mph_m = -1j*np.sqrt(omega_q[l2] / omega_qm[l1])*np.cross(Wqml1[k,:], Wql2[k,:])
                    Mph = -1j*np.cross(Wql1[k,:], Wqml2[k,:])
                    Mph_m = -1j*np.cross(Wqml1[k,:], Wql2[k,:])
                    for xi in range(3):
                        G_p[l1, l2, ...] += prefrac_m * Kp_mq[k,xi,...,l1,l2] * Mph_m[xi,None,None]
                        G_m[l1, l2, ...] += prefrac_p * Km_q[k,xi,...,l1,l2] * Mph[xi,None,None]

        for l1 in range(nmodes):
            for l2 in range(nmodes):
                if l1 == l2:
                    Om_out[l1,l2] += omega_q[l1]
                Grho = np.einsum('nm, nm ->',G_p[l1,l2],rho[ik]) + np.einsum('nm, mn ->',G_m[l2,l1],rho[ik])
                Om_out[l1,l2] += Grho
    return Om_out/nk


print('Reading q Path')
## Load Q-Path
gen_grid = True
gamma = True
if gen_grid:
    nk = 100 #int(sys.argv[1])
    print(f'Generating grid {nk} x {nk} x 1')
    k1 = make_kgrid_2D(nk)
else:
    print('Reading k grid from kpoints.dat')
    k1 = np.loadtxt('kpoints.dat')

nk_use = k1.shape[0]
EkL = []
EkR = []
VkL = []
if gamma:
    print('Running Gamma Point',flush=True)
    q = np.array([0.0, 0.0, 0.0])
else:
    print('Running K Point',flush=True)
    q = np.array([1.0/3.0, 1.0/3.0, 0.0])
#B_scan = [0.0, 10.0, 1.0E4, 1.0E6]
B_scan = [0.0, 1.0E-1, 1.0E0, 1.0E1]
#B_scan = [0.0, 1.0E1]
for i,Bz in enumerate(B_scan):
    B = np.array([0.0,0.0,Bz])
    print(i,B)
    #phase = np.exp((2j*np.pi)*np.tensordot(q,cellMapPh.T, axes=1))
    #omegaSq, U = np.linalg.eigh(np.tensordot(phase, omegaSqR, axes=1))
    
    phase = np.exp((2j*np.pi)*np.tensordot(q,cellMapPh.T, axes=1))
    HkR = np.tensordot(phase, omegaSqR, axes=1)
    
    Hk = compute_Omega_q(q,k1, 9, B=B,min_freq=1.0E-3, reduce=True, T=4, Ef=-0.183439546988241) 
    
    if not scipy.linalg.ishermitian(Hk, atol=1e-9):
        h_err = np.max(np.abs(Hk - Hk.conj().T))
        print(f'Hermiticity Error {h_err:0.2e}')
        Hk = (Hk + Hk.conj().T) / 2
    
    #--- Diagonalize:
    Ek,Vk = np.linalg.eigh(Hk)
    Ek_R,Vk_R = np.linalg.eigh(HkR)
    #--- Extract required mode:
    omega = np.copysign(np.sqrt(np.abs(Ek_R)), Ek_R) #signed-sqrt
    EkR.append(omega)
    EkL.append(Ek)
    ionDisp = []
    for i in range(9):
        #ionDisp_t = np.reshape(Vk[:,i] * phononBasis, (-1,3))
        ionDisp_t = np.reshape(Vk[:,i], (-1,3))
        ionDisp_t *= 1./np.sqrt(np.max(np.sum(np.square(ionDisp_t),axis=1)))
        ionDisp.append(ionDisp_t)
    VkL.append(ionDisp)

ionDisp = np.array(VkL)
Ek = np.array(EkL)
EkR = np.array(EkR)
if gamma:
    np.save(f'lEkr-{nk_use}-phonon.npy',EkR)
    np.save(f'lEk-{nk_use}-phonon.npy',Ek)
else:
    np.save(f'K-lEkr-{nk_use}-phonon.npy',EkR)
    np.save(f'K-lEk-{nk_use}-phonon.npy',Ek)

fig, ax = plt.subplots(figsize=(6,4.5))
lin_x = np.arange(Ek.shape[0])
ylim = 0
meV = 1e-3/27.2114

cm1 = 8.065610 / meV

Bi = 0
for Bi in range(Ek.shape[1]):
    y = Ek[:,Bi] *cm1 #/ meV
    yR = EkR[:, Bi] *cm1 #/ meV
    delta = y-yR
    ax.plot(B_scan, delta, label= r'$\mu_'+ f'{Bi}=$' + f'{yR[0]:0.1f}')
    if ylim < np.abs(delta).max():
        ylim = np.abs(delta).max()

ax.legend(ncols=3)
ax.set_xlabel(r'B Field (T)',fontsize=16)
ax.set_ylabel(r'$\Delta \omega (cm^{-1})$',fontsize=16)

plt.tight_layout()
if gamma:
    plt.savefig(f'gamma-{nk_use}-phonon-change-Bfield.png',dpi=300)
else:
    plt.savefig(f'K-{nk_use}-phonon-change-Bfield.png',dpi=300)
plt.show()
