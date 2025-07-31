import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import numba
from numba import prange


@numba.njit
def Rot(th):
    return np.array([[np.cos(th), -np.sin(th)],
                     [np.sin(th),  np.cos(th)]], dtype=np.float64)

@numba.njit
def hamilton2x2(x, y, Delta, B, D, epsilon, t1, t2, t3, t4):
    q_x, q_y = (x/a,y/a)
    q_plus, q_minus, qcuadrado= q_x + 1j * q_y, q_x - 1j * q_y, q_x**2 + q_y**2
    q_plus_epsilon, q_minus_epsilon = (q_plus, q_minus) if epsilon == +1 else (q_minus, q_plus)
    HH=np.zeros((2,2),dtype=np.complex128)
    HH[0,0] = 0.5*D*qcuadrado+(0.5*Delta+0.5*B*qcuadrado)
    HH[1,1] = 0.5*D*qcuadrado-(0.5*Delta+0.5*B*qcuadrado)
    HH[0,1] = np.conjugate(t1) + np.conjugate(t2)*q_minus_epsilon + np.conjugate(t3)*q_minus + np.conjugate(t4)*q_minus_epsilon*q_minus
    HH[1,0] = t1 + t2*q_plus_epsilon + t3*q_plus + t4*q_plus_epsilon*q_plus
    return HH

@numba.njit(parallel=True)
def build_BZ(b1, b2, N):
    vx, vy = np.array([-2*L[0],0],dtype=np.float64)/N, np.array([0,-2*L[0]],dtype=np.float64)/N
    BZ = np.zeros((N,N,2),dtype=np.float64)
    for jy in range(N):
        for jx in range(N):
            BZ[jx, jy] = OO + vx * jx + vy * jy
    return BZ

@numba.njit(parallel=True)
def compute_eigens(BZ, KK, Delta, B, D, epsilon,
                   t1, t2, t3, t4):
    N = BZ.shape[0]
    Earr = np.empty((N, N, 2), np.float64)
    Parr = np.empty((N, N, 2, 2), np.complex128)
    for i in prange(N):
        for j in range(N):
            kx = BZ[i,j,0] - KK[0]
            ky = BZ[i,j,1] - KK[1]
            H = hamilton2x2(kx, ky, Delta, B, D, epsilon, t1, t2, t3, t4)
            w, v = np.linalg.eigh(H)
            Earr[i,j,0] = w[0]; Earr[i,j,1] = w[1]
            Parr[i,j,:,:] = v
    return Earr, Parr

@numba.njit(parallel=True)
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

# ---------------------------------------------------
# --- Constantes y parámetros ---
img = complex(0.0,1.0)
a = 3.15                 # (Å)
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
Q_kp = np.pi / (5)
L     = Q_kp*np.array([-1,-1],dtype=np.float64)
OO    = KK + L

v_u = 2.59; M_u = 2.03
v_l = 2.22; M_l = 2.07
Tplus0plus0  = 0.0067; Tminus2plus0 = 0.0033; Tplus0plus2 = 0.0033
Tminus2plus2 = 0.0100; Tplus2plus2 = 0.0100
B = (v_l**2/M_l) + (v_u**2/M_u)
D = (v_l**2/M_l) - (v_u**2/M_u)

def canales_por_apilamiento(idx):
    # idx de 1 a 6
    # Devuelve t_cc, t_vv, t_cv, t_vc, stacking_upper, stacking_lower, stacking_str
    if idx == 1:
        return [3*Tplus0plus0, 3*Tminus2plus2, 0, 0, "M", "M", r"$R^{M}_{M}$"]
    elif idx == 2:
        return [0, 0, 0, 3*Tminus2plus0, "M", "X", r"$R^{M}_{X}$"]
    elif idx == 3:
        return [0, 0, 3*Tplus0plus2, 0, "X", "M", r"$R^{X}_{M}$"]
    elif idx == 4:
        return [0, 3*Tminus2plus2, 0, 0, "M", "X", r"$H^{M}_{X}$"]
    elif idx == 5:
        return [3*Tplus0plus0, 0, 0, 0, "M", "M", r"$H^{M}_{M}$"]
    elif idx == 6:
        return [0, 0, 3*Tplus0plus2, 3*Tminus2plus0, "X", "X", r"$H^{X}_{X}$"]

# Prepara parámetros comunes
NBZ = 100
BZ = build_BZ(b1, b2, NBZ)
Delta_vals_por_apilamiento = []
chern_valencia_por_apilamiento = []
leyendas = []

for apilamiento in range(1,7):
    t_cc, t_vv, t_cv, t_vc, stacking_upper, stacking_lower, stacking_str = canales_por_apilamiento(apilamiento)
    t1 = t_vc
    t2 = (v_l / M_l) * t_vv
    t3 = -(v_u / M_u) * t_cc
    t4 = ((v_l * v_u) / (M_l * M_u)) * t_cv
    # Criterio para Delta_kp
    if apilamiento == 1:
        t = t2+t3; A_coeff = B
        B_coeff = t**2 + B**2 * Q_kp**2
        C_coeff = B * t**2 * Q_kp**2
    elif apilamiento == 2:
        t = 0; A_coeff = B
        B_coeff = t**2 + B**2 * Q_kp**2
        C_coeff = B * t**2 * Q_kp**2
    elif apilamiento == 3:
        t = t4; A_coeff = 1
        B_coeff = - B * Q_kp**2
        C_coeff = - (8*t**2+2*B**2)*Q_kp**4
    elif apilamiento == 4:
        t = t2; A_coeff = B
        B_coeff = t**2 + B**2 * Q_kp**2
        C_coeff = B * t**2 * Q_kp**2
    elif apilamiento == 5:
        t = t3; A_coeff = B
        B_coeff = t**2 + B**2 * Q_kp**2
        C_coeff = B * t**2 * Q_kp**2
    elif apilamiento == 6:
        t = 0; A_coeff = B
        B_coeff = t**2 + B**2 * Q_kp**2
        C_coeff = B * t**2 * Q_kp**2
    # Solución cuadrática
    discriminante = B_coeff**2 - 4 * A_coeff * C_coeff
    if discriminante >= 0:
        delta1 = (-B_coeff + np.sqrt(discriminante)) / (2 * A_coeff)
        delta2 = (-B_coeff - np.sqrt(discriminante)) / (2 * A_coeff)
        Delta_kp = min(delta1,delta2)/a**2
    else:
        Delta_kp = 0.1 # valor por defecto seguro
    
    f_delta = 0.9
    Delta_max = f_delta * Delta_kp
    Delta_values = np.linspace(0, Delta_max, 80)
    chern_val = []

    for Delta in Delta_values:
        Earr, Parr = compute_eigens(BZ, KK, Delta, B, D, +1 if apilamiento<=3 else -1, t1, t2, t3, t4)
        mitad = 1
        Uarr, Farr = compute_U_F(BZ, Parr, mitad)
        Vv = 0.0
        for i in range(NBZ):
            for j in range(NBZ):
                Vv += np.imag(Farr[i,j,0,0])
        ChVv = Vv/(2*np.pi)
        chern_val.append(ChVv)

    Delta_vals_por_apilamiento.append(Delta_values*1000) # meV
    chern_valencia_por_apilamiento.append(chern_val)
    leyendas.append(stacking_str)

# --------- GRAFICAR TODO -----------
plt.figure(figsize=(12,7))
colores = ['tab:red','tab:brown','tab:orange','tab:blue','tab:red','tab:brown']
for i in range(6):
    plt.plot(Delta_vals_por_apilamiento[i], chern_valencia_por_apilamiento[i],
             marker='o', label=leyendas[i], color=colores[i])
plt.axhline(0, linestyle='--', alpha=0.5, color='k')
plt.axhline(1, linestyle='--', alpha=0.3)
plt.axhline(-1, linestyle='--', alpha=0.3)
plt.xlabel('Δ (meV)')
plt.ylabel('Número de Chern (valencia)')
plt.legend()
plt.gca().yaxis.set_major_locator(MultipleLocator(1))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('chern_vs_delta_apilamientos.png', dpi=300)
plt.show()
