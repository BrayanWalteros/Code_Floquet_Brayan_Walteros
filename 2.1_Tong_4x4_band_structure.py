import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

# Mathematical functions
def Rot(th): 
    RR = np.array([[np.cos(th), -np.sin(th)], 
                   [np.sin(th), np.cos(th)]], dtype=np.float64)
    return RR

# Bicapa
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
                                  
epsilon = int(input("\nStacking:\n +1. R-stacking\n -1. H-stacking\n Ingrese el número de stacking: "))             
v_u = 2.59       # WS2:  vel fermi upper [v_l=hbar.v_Fl] (eV.Å)
M_u = 2.03            # masa upper (eV)   
v_l = 2.22       # MoS2: vel fermi lower [v_u=hbar.v_Fu] (eV.Å)
M_l = 2.07            # masa lower (eV)  
Tplus0plus0  = 0.0067   # (eV)
Tminus2plus0 = 0.0033   # (eV)
Tplus0plus2  = 0.0033   # (eV)
Tminus2plus2 = 0.0100   # (eV)
Tplus2plus2 = 0.0100    # (eV)
B = (v_l**2/M_l) + (v_u**2/M_u)
D = (v_l**2/M_l) - (v_u**2/M_u)

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

# Criterio de Delta
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
    Delta_kp = min(delta1,delta2)/a**2

# Solucion Delta          
puntos = 3000
f_delta = 0.5
Delta = f_delta*Delta_kp
Q = Q_kp
kmin, kmax = -Q, Q    

# ---------------------------Calculos-----------------------------------
def hamilton4x4(x,y):
    q_x, q_y = (x/a, y/a)
    HH=np.zeros((4,4),dtype=np.complex128)
    HH[0,0] = -0.5*Delta+M_u
    HH[0,1] = v_u*(q_x+1j*q_y)
    HH[0,2] = np.conjugate(t_cc)
    HH[0,3] = np.conjugate(t_cv)
    HH[1,0] = v_u*(q_x-1j*q_y)
    HH[1,1] = -0.5*Delta
    HH[1,2] = np.conjugate(t_vc)
    HH[1,3] = np.conjugate(t_vv)
    HH[2,0] = t_cc
    HH[2,1] = t_vc
    HH[2,2] = 0.5*Delta
    HH[2,3] = v_l*(q_x+1j*epsilon*q_y)
    HH[3,0] = t_cv
    HH[3,1] = t_vv
    HH[3,2] = v_l*(q_x-1j*epsilon*q_y)
    HH[3,3] = 0.5*Delta-M_l
    return HH

# Crear figura
fig, ax = plt.subplots(figsize=(8, 6))
factor = 1
kx_values = np.linspace(factor*kmin, factor*kmax, puntos)
line_colors = ['b', 'r', 'b', 'r']  # Colores para las bandas
lines = [ax.plot([], [], color=color, marker='.',markersize=1)[0] for color in line_colors]
ax.set_xlim(kmin, kmax)
ax.set_xlabel('$aq_x$')
ax.set_ylabel('Energía (eV)')
ax.axhline(0, color='black', marker='.', linestyle='None', linewidth=0.7)
title = ax.set_title('')

energy_bands = []
for kx in kx_values:
    HH = hamilton4x4(kx, 0)
    energies = np.linalg.eigvalsh(HH)
    energy_bands.append(energies)
energy_bands = np.array(energy_bands)
ax.set_ylim(np.min(energy_bands)*1.2, np.max(energy_bands)*1.2)
for band, line in enumerate(lines):
    line.set_data(kx_values, energy_bands[:, band])
title.set_text(f'Espectro de Energías (Δ = {Delta*1000:.2f} meV)')  # Actualizar título dinámicamente

# Calcular diferencias entre bandas consecutivas
diferencias = np.diff(energy_bands, axis=1)  # (N_kx, N_bandas - 1)
min_diferencias = np.min(diferencias, axis=0)  # mínimo para cada separación entre bandas
min_diferencia_total = np.min(min_diferencias)  # mínimo global

# Imprimir resultados
for i, val in enumerate(min_diferencias):
    print(f"Diferencia mínima entre banda {i+1} y banda {i+2}: {val*1000:.2f} meV")

print(f"\nDiferencia mínima total entre todas las bandas: {min_diferencia_total*1000:.2f} meV")


# Guardar figura
plt.savefig('band_structure_4x4_puntos'+str(apilamiento)+'.png', dpi=300)
plt.show()
