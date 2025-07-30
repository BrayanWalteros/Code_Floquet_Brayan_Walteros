import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from numba import njit, prange



# Matrices de Pauli 
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_0 = np.eye(2, dtype=complex)

# Constantes fisicas
e = 1.602e-19           # C
h_bar = 6.582776e-16    # eV.s 
cte = (e/(2*h_bar))**2
img = complex(0.0,1.0)

# Bicapa
a = 1                   # (Å)
a_0 = a / np.sqrt(3)
b = (4*np.pi)/(np.sqrt(3.0)*a)
a1 = a*np.array([0.5*np.sqrt(3.0), 0.5])
a2 = a*np.array([0.5*np.sqrt(3.0)/2, -0.5])
b1 = b*np.array([0.5, 0.5*np.sqrt(3.0)])
b2 = b*np.array([0.5,-0.5*np.sqrt(3.0)],dtype=np.float64)
b3 = -b1 - b2
K_BZ    = (b/np.sqrt(3.0))*np.array([0.5*np.sqrt(3.0),-0.5],dtype=np.float64)
Gamma = np.array([0.0,0.0],dtype=np.float64)-K_BZ
KK    = (b/np.sqrt(3.0))*np.array([0.5*np.sqrt(3.0),-0.5],dtype=np.float64)-K_BZ
KP    = (b/np.sqrt(3.0))*np.array([0.5*np.sqrt(3.0),0.5],dtype=np.float64)-K_BZ
MM    = (b/2)*np.array([1,0],dtype=np.float64)-K_BZ
L     = (np.pi/(5*a))*np.array([-1,-1],dtype=np.float64)
OO    = KK + L

gamma2 = 2.22        # MoS2: vel fermi upper [v_u=hbar.v_Fu] (eV.Å)
epsilon2 = 2.07              # masa upper (eV)
gamma1 = 2.59        # WS2:  vel fermi lower [v_l=hbar.v_Fl] (eV.Å)
epsilon1 = 2.03              # masa lower (eV)                    
epsilon0 = 0
Tplus0plus0  = 0.0067   # (eV)
Tminus2plus0 = 0.0033   # (eV)
Tplus0plus2  = 0.0033   # (eV)
Tminus2plus2 = 0.0100   # (eV)
m_plus = h_bar**2 / (np.abs(gamma2)**2 / epsilon2 + np.abs(gamma1)**2 / epsilon1)
m_minus = h_bar**2 / (np.abs(gamma2)**2 / epsilon2 - np.abs(gamma1)**2 / epsilon1)                                        
alpha_plus = h_bar**2 / (2 * m_plus)
alpha_minus = h_bar**2 / (2 * m_minus)
mu_plus = alpha_minus + alpha_plus
mu_minus = alpha_minus - alpha_plus
u = e / h_bar

# Canales t
def canales(varphi,Delta,config):
    if varphi == +1:
        stacking = "R"
        if config == 1:
            t_cc = 3*Tplus0plus0
            t_vv = 3*Tminus2plus2
            return [0, 0, 0.5*(-np.conj(t_cc)*gamma1*(Delta-2*epsilon1)/(epsilon1*(Delta-epsilon1))+np.conj(t_vv)*gamma2*(Delta-2*epsilon2)/(epsilon2*(Delta-epsilon2))), 0, 0], 1
        elif config == 2:
            t_vc = 3*Tminus2plus0
            return [np.conj(t_vc), 0, 0,(np.conj(t_vc)/2)*(abs(gamma2)**2/(epsilon2**2)+ abs(gamma1)**2 / (epsilon1**2)), 0], 2
        elif config== 3:
            t_cv = 3*Tplus0plus2
            return [0, 0, 0, 0, (gamma1*gamma2*np.conj(t_cv)/2)*((Delta * (epsilon1 + epsilon2) - 2 * epsilon1 * epsilon2)/(epsilon1 * epsilon2 * (Delta - epsilon1) * (Delta - epsilon2)))], 3
    elif varphi == -1:
        stacking = "H"
        if config == 1:
            t_vv = 3*Tminus2plus2          
            return [0, 0.5*gamma2*np.conj(t_vv)*(Delta-2*epsilon2)/(epsilon2*(Delta-epsilon2)), 0, 0, 0], 4
        elif config == 2:    
            t_cc = 3*Tplus0plus0         
            return [0, 0, -0.5*gamma1*np.conj(t_cc)*(Delta-2*epsilon1)/(epsilon1*(Delta-epsilon1)), 0, 0], 5      
        elif config == 3:
            t_cv = 3*Tplus0plus2
            t_vc = 3*Tminus2plus0
            return [np.conj(t_vc), 0, 0, gamma1*gamma2*np.conj(t_cv)/(epsilon1*epsilon2), 0], 6 

@njit(parallel=True)
def BZ_TB(N):
    vx = np.array([-2*L[0], 0]) / N
    vy = np.array([0, -2*L[0]]) / N
    BZ = np.empty((N, N, 2), dtype=np.float64)
    for jy in prange(N):
        for jx in range(N):
            BZ[jx, jy, :] = OO + vx * jx + vy * jy
    return BZ
    
# BZ discreta
def imagen(E_0, hw, Delta):
    # ---------------------------------------------------
    # Hamiltoniano de Floquet
    alfa = (a*e*E_0)/(hw)
    A0 = alfa * h_bar/(a*e)

    def hamiltonian(x,y,hw,Delta,A0,eta):
        q_plus, q_minus = x+img*y, x-img*y
        q_plus_eta, q_minus_eta = x+eta*img*y, x-eta*img*y
        q2 = x**2 + y**2
        z = 1 - img*eta
        epsilon_q = epsilon0 + (h_bar**2 * q2) / (2 * m_minus)
        Delta_q_medium = 0.5*Delta + (h_bar**2 * q2) / (2 * m_plus)
        t_q = t0 + gamma_plus*q_plus + gamma_minus*q_minus + gamma_I*q2 + gamma_II*q_minus**2 
        factor = (u * A0) / (2*img)
        # matriz hk
        hk = np.array([[epsilon_q+Delta_q_medium, t_q], [np.conj(t_q), epsilon_q-Delta_q_medium]], dtype=np.complex128)
        
        # H2,H1 and H0
        H011 = -4 * (factor**2) * mu_plus
        H022 = -4 * (factor**2) * mu_minus
        H012 = -4 * (factor**2) * (gamma_I + gamma_II)
        H021 = np.conj(H012)
        H0 = hk + np.array([[H011, H012],[H021, H022]], dtype=np.complex128)

        H111 = factor * 2 * mu_plus * q_minus_eta
        H122 = factor * 2 * mu_minus * q_minus_eta
        H112 = factor * (gamma_plus * z + gamma_minus * np.conj(z) + gamma_I * 2 * q_minus_eta + gamma_II * q_minus * np.conj(z))
        H121 = np.conj(H112)
        H1 = np.array([[H111, H112],[H121, H122]], dtype=np.complex128)


        H_plus_1 = H1
        H_minus_1= np.conjugate(np.transpose(H_plus_1))

        Basis = np.arange(-Nbasis, Nbasis + 1, dtype=np.int32)
        HF = np.zeros((2 * Basis.size, 2 * Basis.size), dtype=np.complex128)
        for ii, n in enumerate(Basis):
            HF[2*ii:2*ii+2, 2*ii:2*ii+2] = H0 + n * hw * sigma_0
        for ii in range(Basis.size - 1):
            HF[2*ii:2*ii+2, 2*(ii+1):2*(ii+1)+2] = H_minus_1  # H_{-1}
            HF[2*(ii+1):2*(ii+1)+2, 2*ii:2*ii+2] = H_plus_1   # H_{1}

        return HF
    

    @njit(parallel=True)
    def compute_U_F(BZ, Parr, mitad):
        N = BZ.shape[0]
        Uarr = np.empty((N, N, 2, 2, mitad), np.complex128)
        Farr = np.empty((N, N, 2, mitad), np.complex128)

        for i in prange(N):
            for j in range(N):
                ip = (i+1) % N
                jp = (j+1) % N
                for band in range(mitad):
                    for b in range(2):
                        a = Parr[i,j,:, b*mitad + band]
                        bvec = Parr[ip,j,:, b*mitad + band]
                        dum = np.vdot(a, bvec)
                        Uarr[i,j,0,b,band] = dum/abs(dum) if abs(dum)!=0 else 0
                        
                        cvec = Parr[i,jp,:, b*mitad + band]
                        dum = np.vdot(a, cvec)
                        Uarr[i,j,1,b,band] = dum/abs(dum) if abs(dum)!=0 else 0

        for i in prange(N):
            for j in range(N):
                ip = (i+1)%N 
                jp = (j+1)%N
                for b in range(2):
                    for band in range(mitad):
                        num = Uarr[i,j,0,b,band] * Uarr[ip,j,1,b,band]
                        den = Uarr[i,jp,0,b,band] * Uarr[i,j,1,b,band]
                        Farr[i,j,b,band] = np.log(num/den) if abs(den) > 1e-12 and abs(num) > 1e-12 else 0

                        
        return Uarr, Farr
              

    # Función para obtener eigenvalores y vectores
    get_eig = lambda kx, ky: np.linalg.eigh(hamiltonian(kx, ky, hw, Delta, A0, eta))
    eigG, vecG = get_eig(0, 0)
    idx = np.argsort(eigG)

    if Nbasis == 1:
        idx_v0, idx_vp1, idx_cm1, idx_c0 = idx[1], idx[2], idx[3], idx[4]
        mapping = {'v₀': (0, idx_v0), 'v₊₁': (0, idx_vp1), 'c₋₁': (1, idx_cm1-half), 'c₀': (1, idx_c0-half)}
    elif Nbasis == 0:
        idx_v0, idx_c0 = idx[0], idx[1]
        mapping = {'v₀': (0, idx_v0), 'c₀': (1, idx_c0-half)}


    Earr = np.zeros((NBZ, NBZ, 2*(2*Nbasis+1)))
    Parr = np.zeros((NBZ, NBZ, 2*(2*Nbasis+1), 2*(2*Nbasis+1)), dtype=np.complex128)
    for i in range(NBZ):
        for j in range(NBZ):
            kx, ky = BZ[i,j]
            vals, vecs = np.linalg.eigh(hamiltonian(kx, ky, hw, Delta, A0, eta))
            Earr[i,j] = vals
            Parr[i,j] = vecs

    Uarr, Farr = compute_U_F(BZ, Parr, half)


    chern_vals = [np.imag(np.sum(Farr[:,:,b,band]))/(2*np.pi) for (b,band) in [mapping[n] for n in bands]]
    return chern_vals

# --------------------- Parámetros ------------------
# Discretizacion de BZ
NBZ = 100       
# Tamaño de matriz de Floquet
Nbasis = 1
# Verifica Nbasis y define bandas según su valor
if Nbasis == 1:
    bands = ['v₀','v₊₁','c₋₁','c₀']
elif Nbasis == 0:
    bands = ['v₀','c₀']
else:
    raise ValueError("Nbasis debe ser 0 o 1")
# Gap entre cond_lower y val_upper (eV)
Delta_g = 1.45    
# Polarizacion (1:right, -1:left)
eta = 1
# Campo electrico E_0
d = 6.516 # A
 
# ---------------------------------------------------  
BZ = BZ_TB(NBZ)  
size_F = 2 * (2 * Nbasis + 1)
half   = size_F // 2  # número de bandas de valencia/conducción   

varphi = 1
config = 1


hw_values = np.linspace(0.5, 1.79, 30)         # Puedes ajustar la resolución
E_0_values = np.linspace(0.001, 0.01, 20)
banda_objetivo = "v₊₁"  # Cambia aquí la banda que quieras graficar (ej: "v₀", "c₀", "v₊₁", "c₋₁")
chern_map = np.zeros((len(E_0_values), len(hw_values)))

total_iteraciones = len(hw_values) * len(E_0_values)
contador = 0
for iy, E_0 in enumerate(E_0_values):
    for ix, hw in enumerate(hw_values):
        U = E_0 * d
        Delta = Delta_g - U   
        canal, apilamiento = canales(varphi, Delta, config)
        t0, gamma_plus, gamma_minus, gamma_I, gamma_II = canal
        chern_vals = imagen(E_0, hw, Delta)
        
        # Definir el orden de bandas según Nbasis
        if Nbasis == 1:
            bandas = ['v₀', 'v₊₁', 'c₋₁', 'c₀']
        elif Nbasis == 0:
            bandas = ['v₀', 'c₀']
        ind = bandas.index(banda_objetivo)
        chern_map[iy, ix] = chern_vals[ind]
        contador += 1
        progreso = 100 * contador / total_iteraciones
        print(f'Progreso global: {progreso:.2f}% | apilamiento = {apilamiento}, ℏω = {hw:.2f} eV, E₀ = {E_0:.3f}')

# --- Gráfica final ---
plt.figure(figsize=(8, 6))
extent = [hw_values[0], hw_values[-1], E_0_values[0], E_0_values[-1]]
im = plt.imshow(chern_map, aspect='auto', origin='lower', extent=extent,
                cmap='RdBu_r', vmin=-2, vmax=2)
plt.xlabel(r'$\hbar\omega$ [eV]')
plt.ylabel(r'$E_0$ [V/Å]')
plt.title(f'Número de Chern | Banda: {banda_objetivo} | Apilamiento: {apilamiento}')
cbar = plt.colorbar(im, label='Chern')
# Etiquetas solo enteros en barra de color
cbar.set_ticks(np.arange(-2, 3, 1))
fname = f"mapa_chern_apil{apilamiento}_banda_{banda_objetivo}.png"
plt.tight_layout()
plt.savefig(f"mapa_colores_apilamiento_{apilamiento}.png")
plt.close()
    

    
