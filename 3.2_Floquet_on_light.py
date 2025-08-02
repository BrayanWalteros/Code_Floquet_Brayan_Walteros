import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from numba import prange

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
a = 3.15                  # (Å)
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
Q_kp = 3*np.pi/5
L     = Q_kp*np.array([-1,-1],dtype=np.float64)
OO    = KK + L

gamma1 = 2.59       # WS2:  vel fermi upper [v_l=hbar.v_Fl] (eV.Å)
epsilon1 = 2.03            # masa upper (eV)   
gamma2 = 2.22       # MoS2: vel fermi lower [v_u=hbar.v_Fu] (eV.Å)
epsilon2 = 2.07            # masa lower (eV)                 
epsilon0 = 0
Tplus0plus0  = 0.0067   # (eV)
Tminus2plus0 = 0.0033   # (eV)
Tplus0plus2  = 0.0033   # (eV)
Tminus2plus2 = 0.0100   # (eV)

t_cc = 3*Tplus0plus0
t_vv = 3*Tminus2plus2
t_cv = 3*Tplus0plus2
t_vc = 3*Tminus2plus0
m_plus = h_bar**2 / (np.abs(gamma2)**2 / epsilon2 + np.abs(gamma1)**2 / epsilon1)
m_minus = h_bar**2 / (np.abs(gamma2)**2 / epsilon2 - np.abs(gamma1)**2 / epsilon1)                                        
alpha_plus = h_bar**2 / (2 * m_plus)
alpha_minus = h_bar**2 / (2 * m_minus)
mu_plus = alpha_minus + alpha_plus
mu_minus = alpha_minus - alpha_plus
u = e / h_bar

def canales(varphi, Delta, config):
    # Predefinición de los canales para cada caso, con formato [t_0, gamma_plus, gamma_minus, gamma_I, gamma_II]
    canales_dict = {
        (+1, 1): ([0, 0, gamma_minus, 0, 0], 1),
        (+1, 2): ([t_0, 0, 0, gamma_I, 0], 2),
        (+1, 3): ([0, 0, 0, 0, gamma_II], 3),
        (-1, 1): ([0, gamma_plus, 0, 0, 0], 4),
        (-1, 2): ([0, 0, gamma_minus, 0, 0], 5),
        (-1, 3): ([t_0, 0, 0, gamma_I, gamma_II], 6)
    }
    # Retorna la tupla correspondiente o lanza error si la combinación es inválida
    try:
        return canales_dict[(varphi, config)]
    except KeyError:
        raise ValueError(f"Combinación no válida: varphi={varphi}, config={config}")

def BZ_TB(N):
    vx = np.array([-2*L[0], 0]) / N
    vy = np.array([0, -2*L[0]]) / N
    BZ = np.empty((N, N, 2), dtype=np.float64)
    for jy in prange(N):
        for jx in range(N):
            BZ[jx, jy, :] = OO + vx * jx + vy * jy
    return BZ

# BZ discreta
def imagen(hw):
    # ---------------------------------------------------
    alfa = (a*e*E_0)/(hw)
    A0 = alfa * h_bar/(a*e)
    # Hamiltoniano de Floquet
    def hamiltonian(x,y,hw,Delta,A0,eta):
        x=x/a
        y=y/a
        q_plus, q_minus = x+img*y, x-img*y
        q_plus_eta, q_minus_eta = x+eta*img*y, x-eta*img*y
        q2 = x**2 + y**2
        z = 1 - img*eta
        epsilon_q = epsilon0 + (h_bar**2 * q2) / (2 * m_minus)
        Delta_q_medium = 0.5*Delta + (h_bar**2 * q2) / (2 * m_plus)
        t_q = t0 + gamma_plus*q_plus + gamma_minus*q_minus + gamma_I*q2 + gamma_II*q_minus**2 
        
        # matriz hk
        hk = np.array([[epsilon_q+Delta_q_medium, t_q], [np.conj(t_q), epsilon_q-Delta_q_medium]], dtype=np.complex128)
        
        # H2,H1 and H0
        H011 = u**2 * A0**2 * mu_plus
        H022 = u**2 * A0**2 * mu_minus
        H012 = u**2 * A0**2 * (gamma_I + gamma_II)
        H021 = np.conj(H012)
        H0 = hk  + np.array([[H011, H012],[H021, H022]], dtype=np.complex128)

        H111 = ((u * A0)/(2*img)) * 2 * mu_plus * q_minus_eta
        H122 = ((u * A0)/(2*img)) * 2 * mu_minus * q_minus_eta
        H112 = ((u * A0)/(2*img)) * (-(gamma_plus * z + gamma_minus * np.conj(z) + gamma_I * 2 * q_minus_eta + gamma_II * q_minus * np.conj(z)))
        H121 = np.conj(H112)
        H1 = np.array([[H111, H112],[H121, H122]], dtype=np.complex128)

        H211 = 0
        H222 = 0
        H212 = -((u**2 * A0**2)/(2*img))* gamma_II * eta
        H221 = np.conj(H212)
        H2 = np.array([[H211, H212],[H221, H222]], dtype=np.complex128)

        H_plus_1 = H1
        H_minus_1= np.conjugate(np.transpose(H_plus_1))
        H_plus_2 = H2
        H_minus_2= np.conjugate(np.transpose(H_plus_2))

        Basis = np.arange(-Nbasis, Nbasis + 1, dtype=np.int32)
        HF = np.zeros((2 * Basis.size, 2 * Basis.size), dtype=np.complex128)
        for ii, n in enumerate(Basis):
            HF[2*ii:2*ii+2, 2*ii:2*ii+2] = H0 + n * hw * sigma_0
        for ii in range(Basis.size - 1):
            HF[2*ii:2*ii+2, 2*(ii+1):2*(ii+1)+2] = H_minus_1  # H_{-1}
            HF[2*(ii+1):2*(ii+1)+2, 2*ii:2*ii+2] = H_plus_1   # H_{1}
        for ii in range(Basis.size - 2):
            HF[2*ii:2*ii+2, 2*(ii+2):2*(ii+2)+2] = H_minus_2  # H_{-2}
            HF[2*(ii+2):2*(ii+2)+2, 2*ii:2*ii+2] = H_plus_2   # H_{2}

        return HF

    def compute_U_F(BZ, Parr, mitad):
        N = BZ.shape[0]
        Uarr = np.empty((N, N, 2, 2, mitad), np.complex128)
        Farr = np.empty((N, N, 2, mitad), np.complex128)

            # separar eigenestados
        for i in prange(N):
            for j in range(N):
                pass  # Parr ya contiene v[:,band] en último eje

        # Calcular Uarr
        for i in prange(N):
            for j in range(N):
                for band in range(mitad):
                    # índices vecinos
                    ip = (i+1) % N
                    jp = (j+1) % N

                    # valence y conduction en bandas 0 y 1
                    for b in range(2):
                        # avance en kx
                        a = Parr[i,j,:,b*mitad+band]
                        bvec = Parr[ip,j,:,b*mitad+band]
                        dum = np.vdot(a, bvec)
                        Uarr[i,j,0,b,band] = dum/abs(dum) if abs(dum)!=0 else 0
                        # avance en ky
                        cvec = Parr[i,jp,:,b*mitad+band]
                        dum = np.vdot(a, cvec)
                        Uarr[i,j,1,b,band] = dum/abs(dum) if abs(dum)!=0 else 0

        # Calcular Farr
        for i in prange(N):
            for j in range(N):
                ip = (i+1)%N; jp = (j+1)%N
                for b in range(2):
                    for band in range(mitad):
                        num = Uarr[i,j,0,b,band] * Uarr[ip,j,1,b,band]
                        den = Uarr[i,jp,0,b,band] * Uarr[i,j,1,b,band]
                        Farr[i,j,b,band] = np.log(num/den) if abs(den)!=0 else 0
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

    # Línea central en la dirección k_y
    iy = NBZ // 2
    #kx_arr = np.linspace(-np.pi/5,np.pi/5,1000)
    kx_arr = BZ[:, iy, 0]
    # Espectro para bandas seleccionadas
    if Nbasis == 1:
        E_line = [Earr[:, iy, idx] for idx in [idx_v0, idx_vp1, idx_cm1, idx_c0]]
    elif Nbasis == 0:
        E_line = [Earr[:, iy, idx] for idx in [idx_v0, idx_c0]]


    # Cálculo de números de Chern
    chern_vals = [np.imag(np.sum(Farr[:,:,b,band]))/(2*np.pi) for (b,band) in [mapping[n] for n in bands]]
    chern_str = ",  ".join(f"{b}={v:+.2f}" for b, v in zip(bands, chern_vals))

    # Plot del espectro
    fig, ax = plt.subplots(figsize=(8,5))
    for y, band in zip(E_line, bands):
        ax.plot(kx_arr, y, label=band, color='blue')

    ax.set_title(f"ℏω = {hw:.2f} eV\nChern: {chern_str}", fontsize=16, fontweight='bold')
    #ax.set_title(f"Chern: {chern_str}", fontsize=16, fontweight='bold')
    ax.set_xlabel('$aq_x$',fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_ylabel('Energía (eV)',fontsize=18)
    plt.tight_layout()
    plt.savefig(f"3.{apilamiento}1.png")
    plt.show()
    plt.close()

    # Plot de curvatura de Berry
    if Nbasis == 1:
        fig, axes = plt.subplots(2, 2, figsize=(12,10), sharex=True, sharey=True)
        ax_list = axes.flatten()
    elif Nbasis == 0:
        fig, axes = plt.subplots(1, 2, figsize=(12,5), sharex=True, sharey=True)
        ax_list = axes.flatten()

    for ax, name in zip(ax_list, bands):
        b, band = mapping[name]
        Z = Farr[:,:,b,band].imag
        ctf = ax.contourf(BZ[:,:,0], BZ[:,:,1], Z, levels=100, cmap='RdBu_r')
        fig.colorbar(ctf, ax=ax)
        ax.set_title(f'{name}',fontsize=18)
        ax.set_xlabel('$aq_x$',fontsize=16)
        ax.set_ylabel('$aq_y$',fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        

    # Título global
    fig.suptitle(f'ℏω = {hw:.2f} eV\nChern: {chern_str}', fontsize=20, fontweight='bold')
    #fig.suptitle(f'Chern: {chern_str}', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(f"3.{apilamiento}2.png")
    plt.show()
    plt.close()



# --------------------- Parámetros ------------------
# Discretizacion de BZ
NBZ = 500      
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
d = 6.516
E_0 = 0.01 # V/A
U = E_0 * d
Delta = Delta_g - U    
# ---------------------------------------------------  
t_0 = np.conjugate(t_vc)
gamma_plus = 0.5 * (gamma2 * np.conjugate(t_vv) * (Delta - 2 * epsilon2)) / (epsilon2 * (Delta - epsilon2))
gamma_minus = -0.5 * (gamma1 * np.conjugate(t_cc) * (Delta - 2 * epsilon1)) / (epsilon1 * (Delta - epsilon1))
gamma_I = 0.5 * np.conjugate(t_vc) * (
    (np.abs(gamma2)**2) / (epsilon2 * (Delta - epsilon2)) +
    (np.abs(gamma1)**2) / (epsilon1 * (Delta - epsilon1))
)
gamma_II = (gamma1 * gamma2 * np.conjugate(t_cv) * (Delta * (epsilon1 + epsilon2) - 2 * epsilon1 * epsilon2)) / (
    2 * epsilon1 * epsilon2 * (Delta - epsilon1) * (Delta - epsilon2)
)
# ---------------------------------------------------  
BZ = BZ_TB(NBZ)  
size_F = 2 * (2 * Nbasis + 1)
half   = size_F // 2  # número de bandas de valencia/conducción   

varphi = -1
config = 3

canal, apilamiento = canales(varphi, Delta, config)[0], canales(varphi, Delta, config)[1]
t0, gamma_plus, gamma_minus, gamma_I, gamma_II = canal

hw = 1.6
imagen(hw)


