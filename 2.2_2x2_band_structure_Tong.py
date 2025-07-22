import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

# Mathematical functions
def Rot(th): 
    RR = np.array([[np.cos(th), -np.sin(th)], 
                   [np.sin(th), np.cos(th)]], dtype=np.float64)
    return RR

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

# Parámetros          
puntos = 3000
f_delta = 0.50
Delta = f_delta * Delta_kp
Q = Q_kp
kmin, kmax = -Q, Q    

# Hamilton tipo Dirac 2x2
def hamilton2x2(x,y):
    q_x, q_y = (x,y)
    q_plus, q_minus, qcuadrado= q_x + 1j * q_y, q_x - 1j * q_y, q_x**2 + q_y**2
    q_plus_epsilon, q_minus_epsilon = (q_plus, q_minus) if epsilon == +1 else (q_minus, q_plus)
    HH=np.zeros((2,2),dtype=np.complex128)
    HH[0,0] = 0.5*D*qcuadrado+(0.5*Delta+0.5*B*qcuadrado)
    HH[1,1] = 0.5*D*qcuadrado-(0.5*Delta+0.5*B*qcuadrado)
    HH[0,1] = np.conjugate(t1) + np.conjugate(t2)*q_minus_epsilon + np.conjugate(t3)*q_minus + np.conjugate(t4)*q_minus_epsilon*q_minus
    HH[1,0] = t1 + t2*q_plus_epsilon + t3*q_plus + t4*q_plus_epsilon*q_plus
    return HH

# Crear figura
factor = 1
fig, ax = plt.subplots(figsize=(8, 6))
kx_values = np.linspace(factor*kmin, factor*kmax, puntos)
Enarr = np.zeros((2,kx_values.size),dtype=np.float64)
# Calcular eigenestados
ik=0
while(ik<kx_values.size):
     kx = kx_values[ik]
     HH = hamilton2x2(kx,0)
     w,v = np.linalg.eigh(HH)
     Enarr[:,ik] = w
     ik+=1

# Calcular diferencia entre las dos bandas
diferencias = np.abs(Enarr[1] - Enarr[0])  # |E2 - E1| para cada kx
min_diferencia = np.min(diferencias)       # valor mínimo
# Imprimir resultado
print(f"\nDiferencia mínima entre las dos bandas: {min_diferencia*1000:.2f} meV")


# Graficar puntos
ax.plot(kx_values, Enarr[0]*1000, 'r-', markersize=1)
ax.plot(kx_values, Enarr[1]*1000, 'b-', markersize=1)

# Personalizar gráfica
ax.set_xlabel('$aq_x$')
ax.set_ylabel('Energía (meV)')
ax.set_title(f'Espectro de Energías (Δ = {Delta:.2f} eV)')
ax.set_xlim(kmin, kmax)

# Guardar figura
plt.savefig('band_structure_2x2_puntos'+str(apilamiento)+'.png', dpi=300)
plt.show()
