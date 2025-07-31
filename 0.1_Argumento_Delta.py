import numpy as np
import matplotlib.pyplot as plt

epsilon = 1          # Tipo de apilamiento (+1 para R, -1 para H)
configuracion = 3     # configuracion (1,2,3: ver tabla)

a = 3.15                   # (Å)
a_0 = a / np.sqrt(3.0)
b = (4*np.pi)/(np.sqrt(3.0)*a)

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
def canales(epsilon,configuracion):
    if epsilon == +1:
        stacking = "R"
        if configuracion == 1:
            return [3*Tplus0plus0, 3*Tminus2plus2, 0, 0, "M", "M", stacking]
        elif configuracion == 2:
            return [0, 0, 0, 3*Tminus2plus0, "M", "X", stacking]
        elif configuracion == 3:
            return [0, 0, 3*Tplus0plus2, 0, "X", "M", stacking]
    elif epsilon == -1:
        stacking = "H"
        if configuracion == 1:
            return [0, 3*Tminus2plus2, 0, 0, "M", "X", stacking]
        elif configuracion == 2:
            return [3*Tplus0plus0, 0, 0, 0, "M", "M", stacking]
        elif configuracion == 3:
            return [0, 0, 3*Tplus0plus2, 3*Tminus2plus0, "X", "X", stacking]
canal = canales(epsilon,configuracion)
t_cc, t_vv, t_cv, t_vc, stacking_upper, stacking_lower, stacking = canal

t1 = t_vc
t2 = (v_l / M_l) * t_vv
t3 = -(v_u / M_u) * t_cc
t4 = ((v_l * v_u) / (M_l * M_u)) * t_cv

if epsilon == +1:
  if configuracion == 1:
    t=t1+t2
    apilamiento=1
  elif configuracion == 2:
    t=0
    apilamiento=2
  elif configuracion == 3:
    t=t4
    apilamiento=3
elif epsilon == -1:
  if configuracion == 1:
    t=t2
    apilamiento=4
  elif configuracion == 2:
    t=t3
    apilamiento=5
  elif configuracion == 3:
    t=0
    apilamiento=6

q_max = np.pi / (5 * a)
# Coeficientes de la cuadrática en Δ
if apilamiento==1:
  A_coeff = B
  B_coeff = t**2 + B**2 * q_max**2
  C_coeff = B * t**2 * q_max**2
elif apilamiento==2:
  A_coeff = 0
  B_coeff = 0
  C_coeff = 0
elif apilamiento==3:
  A_coeff = 1
  B_coeff = - B * q_max**2
  C_coeff = - (8*t**2+2*B**2)*q_max**4
elif apilamiento==4:
  A_coeff = B
  B_coeff = t**2 + B**2 * q_max**2
  C_coeff = B * t**2 * q_max**2
elif apilamiento==5:
  A_coeff = B
  B_coeff = t**2 + B**2 * q_max**2
  C_coeff = B * t**2 * q_max**2
elif apilamiento==6:
  A_coeff = 0
  B_coeff = 0
  C_coeff = 0

# Soluciones de la cuadrática
discriminante = B_coeff**2 - 4 * A_coeff * C_coeff

if discriminante >= 0:
    delta1 = (-B_coeff + np.sqrt(discriminante)) / (2 * A_coeff)
    delta2 = (-B_coeff - np.sqrt(discriminante)) / (2 * A_coeff)
    soluciones = [delta1, delta2]
    Delta = min(delta1,delta2)
    print(f"Soluciones reales: Δ₁ = {delta1:.4f}, Δ₂ = {delta2:.4f}")
else:
    soluciones = []
    print("No hay soluciones reales para Δ")

# Graficar la función cuadrática
Δ = np.linspace(-5, 5, 500)
y = A_coeff * Δ**2 + B_coeff * Δ + C_coeff

plt.figure(figsize=(8,5))
plt.plot(Δ, y, label='Ecuación cuadrática', color='blue')
plt.axhline(0, color='gray', linestyle='--')

for sol in soluciones:
    plt.plot(sol, 0, 'ro')
    plt.text(sol, 0.05, f'{sol:.4f}', ha='center', color='red')

plt.title('Soluciones de la ecuación cuadrática')
plt.xlabel('Δ (eV)')
plt.ylabel('Valor de la ecuación')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Domain for qx and qy
q_range = np.linspace(-q_max, q_max, 500)
qx, qy = np.meshgrid(q_range, q_range)
q2 = qx**2 + qy**2

# Berry curvature
if apilamiento==1:
  numerator = -t**2 * (Delta - B * q2)
  denominator = 4 * (t**2 * q2 + 0.25 * (Delta + B * q2)**2 )**(1.5)
  omega = numerator / denominator
elif apilamiento==2:
  numerator = 0
  denominator = (2 * t**2 + 0.25 * (Delta + B * q2)**2)**1.5
  omega = numerator / denominator
elif apilamiento==3:
  numerator = -Delta * t**2 * q2
  denominator = (t**2 * q2**2 + 0.25 * (Delta + B * q2)**2)**1.5
  omega = numerator / denominator
elif apilamiento==4:
  numerator = -t**2 * (Delta - B * q_max**2)
  denominator = 4 * (t**2 * q2 + 0.25 * (Delta + B * q2)**2 )**(1.5)
  omega = numerator / denominator
elif apilamiento==5:
  numerator = t**2 * (Delta - B * q2)
  denominator = 4 * (t**2 * q2 + 0.25 * (Delta + B * q2)**2 )**(1.5)
  omega = numerator / denominator
elif apilamiento==6:
  numerator = 0
  denominator = (2 * t**2 + 0.25 * (Delta + B * q2)**2)**1.5
  omega = numerator / denominator


# Plotting
plt.figure(figsize=(6, 5))
plt.pcolormesh(qx, qy, omega, shading='auto')
plt.colorbar(label=r'$a^2\Omega(q)$')
plt.xlabel(r'$aq_x$')
plt.ylabel(r'$aq_y$')
plt.title(r'$a^2\Omega(q)$')
plt.tight_layout()
plt.show()
