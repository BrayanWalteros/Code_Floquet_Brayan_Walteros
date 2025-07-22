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
    q_x, q_y = (x,y)
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
# PARTE PRINCIPAL
# ---------------------------------------------------

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

epsilon = int(input("\nStacking:\n +1. R-stacking\n -1. H-stacking\n Ingrese el número de stacking: "))             
v_u = 2.22        # MoS2: vel fermi upper [v_u=hbar.v_Fu] (eV.Å)
M_u = 2.07              # masa upper (eV)
v_l = 2.59        # WS2:  vel fermi lower [v_l=hbar.v_Fl] (eV.Å)
M_l = 2.03              # masa lower (eV)
Tplus0plus0  = 0.0067   # (eV)
Tminus2plus0 = 0.0033   # (eV)
Tplus0plus2  = 0.0033   # (eV)
Tminus2plus2 = 0.0100   # (eV)
B = (v_u**2/M_u) + (v_l**2/M_l)
D = (v_u**2/M_u) - (v_l**2/M_l)

# Canales t
def canales(epsilon):
    if epsilon == +1:
        stacking = "R"
        configuracion = int(input("\nElegiste R-stacking. Elige una configuración:\n 1. MM \n 2. MX \n 3. XM \nIngrese el número: "))             
        if configuracion == 1:
            return [3*Tplus0plus0, 3*Tminus2plus2, 0, 0, "M", "M", stacking], 1
        elif configuracion == 2:
            return [0, 0, 0, 3*Tminus2plus0, "M", "X", stacking], 2
        elif configuracion == 3:
            return [0, 0, 3*Tplus0plus2, 0, "X", "M", stacking], 3
    elif epsilon == -1:
        stacking = "H"
        configuracion = int(input("\nElegiste H-stacking. Elige una configuración:\n 1. MX \n 2. MM \n 3. XX \nIngrese el número: "))             
        if configuracion == 1:           
            return [0, 3*Tminus2plus2, 0, 0, "M", "X", stacking], 4
        elif configuracion == 2:           
            return [3*Tplus0plus0, 0, 0, 0, "M", "M", stacking], 5
        elif configuracion == 3:
            return [0, 0, 3*Tplus0plus2, 3*Tminus2plus0, "X", "X", stacking], 6

canales = canales(epsilon)
canal, apilamiento= canales[0], canales[1]
t_cc, t_vv, t_cv, t_vc, stacking_upper, stacking_lower, stacking = canal
t1 = t_vc
t2 = (v_l / M_l) * t_vv
t3 = -(v_u / M_u) * t_cc
t4 = ((v_l * v_u) / (M_l * M_u)) * t_cv

# Fronteras
Q_kp = np.pi / (5 *a)
Q_max = (2 * np.pi) / (3 * a)
if apilamiento == 1:
    t = t2+t3
    A_coeff = B
    B_coeff = t**2 + B**2 * Q_kp**2
    C_coeff = B * t**2 * Q_kp**2
elif apilamiento == 2:
    t = 0
    A_coeff = B
    B_coeff = t**2 + B**2 * Q_kp**2
    C_coeff = B * t**2 * Q_kp**2
elif apilamiento == 3:
    t = t4
    A_coeff = 1
    B_coeff = - B * Q_kp**2
    C_coeff = - (8*t**2+2*B**2)*Q_kp**4
elif apilamiento == 4:           
    t = t2
    A_coeff = B
    B_coeff = t**2 + B**2 * Q_kp**2
    C_coeff = B * t**2 * Q_kp**2
elif apilamiento == 5:           
    t = t3
    A_coeff = B
    B_coeff = t**2 + B**2 * Q_kp**2
    C_coeff = B * t**2 * Q_kp**2
elif apilamiento == 6:
    t = 0
    A_coeff = B
    B_coeff = t**2 + B**2 * Q_kp**2
    C_coeff = B * t**2 * Q_kp**2
# Soluciones de la cuadrática
discriminante = B_coeff**2 - 4 * A_coeff * C_coeff

if discriminante >= 0:
    delta1 = (-B_coeff + np.sqrt(discriminante)) / (2 * A_coeff)
    delta2 = (-B_coeff - np.sqrt(discriminante)) / (2 * A_coeff)
    soluciones = [delta1, delta2]
    Delta_kp = min(delta1,delta2)

# mesh y barrido en Delta
NBZ = 100
f_delta = 0.95
Delta_max = f_delta * Delta_kp
Delta_values = np.linspace(0, Delta_max, 500)
chern_convergence = {'conduction': [], 'valence': []}

# construir BZ una sola vez
BZ = build_BZ(b1, b2, NBZ)

for Delta in Delta_values:
    # calcular todos los autovalores y vectores propios
    Earr, Parr = compute_eigens(BZ, KK, Delta, B, D, epsilon, t1, t2, t3, t4)

    # calcular conexiones de enlace y curvatura
    mitad = 1  # ya que dim=2, mitad=1
    Uarr, Farr = compute_U_F(BZ, Parr, mitad)

    # sumar curvaturas
    Cv = 0.0; Vv = 0.0
    for i in range(NBZ):
        for j in range(NBZ):
            Cv += np.imag(Farr[i,j,1,0])
            Vv += np.imag(Farr[i,j,0,0])
    ChCv = Cv/(2*np.pi)
    ChVv = Vv/(2*np.pi)

    chern_convergence['conduction'].append(ChCv)
    chern_convergence['valence'].append(ChVv)
    print(f"Δ = {Delta*1000:6.2f} meV → Chern(C)={ChCv:.4f}, Chern(V)={ChVv:.4f}")

# ---------------------------------------------------
# GRAFICAR RESULTADOS
# ---------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(Delta_values, chern_convergence['conduction'], 'o-', label='Conducción')
plt.plot(Delta_values, chern_convergence['valence'],     's-', label='Valencia')
plt.axhline(0, linestyle='--', alpha=0.3)
plt.axhline(1, linestyle='--', alpha=0.3)
plt.axhline(-1, linestyle='--', alpha=0.3)
plt.xlabel('Δ (eV)')
plt.ylabel('Número de Chern')
plt.legend(); plt.grid(alpha=0.3)
plt.gca().yaxis.set_major_locator(MultipleLocator(1))
plt.tight_layout()
plt.savefig(f'stacking_{stacking}_{stacking_lower}^{stacking_upper}.png', dpi=300)
plt.show()
