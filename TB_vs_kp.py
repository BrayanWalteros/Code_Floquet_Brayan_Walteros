import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

# Funciones matematicas
def Rot(th): # theta en radianes
    RR=np.zeros((2,2),dtype=np.float64)
    RR[0,0]= np.cos(th)
    RR[1,1]= np.cos(th)
    RR[0,1]=-np.sin(th)
    RR[1,0]= np.sin(th)
    return RR
def Phi(x,y):
    kk = np.array([x,y],dtype=np.float64)
    return np.exp(img*np.dot(kk,d1)) + np.exp(img*np.dot(kk,d2)) + np.exp(img*np.dot(kk,d3)) 

# Constantes fisicas
img = complex(0.0,1.0)
e = 1.602e-19
hbar = 6.583e-16        # (eV.s)
cc = 3e8*1e10           # (Å/s)

# Parametros de grafeno
a = 1                  # (Å)
t = -10.1                # (eV)
v_F = cc/300.0          # (Å/s)
a0 = a/np.sqrt(3.0)
b = (4*np.pi)/(np.sqrt(3.0)*a)
a1 = a*np.array([0.5*np.sqrt(3.0),0.5],dtype=np.float64)                     
a2 = a*np.array([0.5*np.sqrt(3.0)/2,-0.5],dtype=np.float64)      
b1 = b*np.array([0.5,0.5*np.sqrt(3.0)],dtype=np.float64)    
b2 = b*np.array([0.5,-0.5*np.sqrt(3.0)],dtype=np.float64)                 
b3 = -b1-b2                                                     
d1 = a0*np.array([1,0],dtype=np.float64)
d2 = np.matmul(Rot((2.0*np.pi)/3.0),d1)
d3 = np.matmul(Rot((4.0*np.pi)/3.0),d1)
Gamma = np.array([0.0,0.0],dtype=np.float64)
KK    = (b/np.sqrt(3.0))*np.array([0.5*np.sqrt(3.0),-0.5],dtype=np.float64)
KP    = (b/np.sqrt(3.0))*np.array([0.5*np.sqrt(3.0),0.5],dtype=np.float64)
MM    = (b/2)*np.array([1,0],dtype=np.float64)

# Parametros
N = 100         # Numero de puntos q en grafica
mass = 0.1        # Medio Gap (eV) 
valle = +1      # Valle elegido (+ o -)
vref = Gamma

# Tight Binding
def Gfull(x,y,mass,valle):
    HH=np.zeros((2,2),dtype=np.complex128)
    phi = Phi(x,y)
    HH[0,0] = mass
    HH[1,1] = -mass
    if(valle==1):
        HH[0,1] = t*np.conjugate(phi)
        HH[1,0] = t*phi
    else:
        HH[0,1] = t*phi
        HH[1,0] = t*np.conjugate(phi)
    return HH

# k.p
def Gkp(x,y,mass,valle): 
    HH=np.zeros((2,2),dtype=np.complex128)
    HH[0,0] = mass
    HH[1,1] = -mass
    dist = np.array([x, y]) 
    dist_KK = np.linalg.norm(dist - KK)
    dist_KP = np.linalg.norm(dist - KP)
    if dist_KK <= dist_KP:
        HH[0,1] = hbar*v_F*((x-KK[0])+valle*img*(y-KK[1]))
        HH[1,0] = hbar*v_F*((x-KK[0])-valle*img*(y-KK[1]))
    elif dist_KP < dist_KK:
        HH[0,1] = hbar*v_F*((x-KP[0])+valle*img*(y-KP[1]))
        HH[1,0] = hbar*v_F*((x-KP[0])-valle*img*(y-KP[1]))
    return HH

# Recorrido entre puntos
def recorrido(ki, kf):
    global kpathx, kpathy, klen, klenref, kpos, klabels  # Declarar las variables globales
    # Número de puntos de muestreo de acuerdo a origen elegido
    if np.array_equal(ki,Gamma) and np.array_equal(kf, KK):     # Gamma - KK
        NN = N
    if np.array_equal(ki, KK) and np.array_equal(kf, MM):       # KK - MM
        NN = N / 2
    if np.array_equal(ki, MM) and np.array_equal(kf, KP):       # MM - KP
        NN = N / 2
    if np.array_equal(ki, KP) and np.array_equal(kf, Gamma):    # KP - Gamma
        NN = N
    # Camino en la BZ
    dk = (kf - ki) / NN
    dl = np.sqrt(np.dot(dk, dk))
    ik = 0
    while(ik<NN):
        kk = ki + ik * dk
        kpathx = np.append(kpathx, kk[0])
        kpathy = np.append(kpathy, kk[1])
        klen = np.append(klen, klenref + ik * dl)
        ik+=1
    # Actualiza el largo total del recorrido
    klenref = klen.max()+dl
    # Actualiza las posiciones y etiquetas
    kpos = np.append(kpos, klenref) 
    return None

# Recorrido completo
def recorrido_completo():
    if np.array_equal(vref, Gamma):
        recorrido(Gamma, KK)
        recorrido(KK, MM)
        recorrido(MM, KP)
        recorrido(KP, Gamma)
    elif np.array_equal(vref, KK):
        recorrido(KK, MM)
        recorrido(MM, KP)
        recorrido(KP, Gamma)
        recorrido(Gamma, KK)
    elif np.array_equal(vref, MM):
        recorrido(MM, KP)
        recorrido(KP, Gamma)
        recorrido(Gamma, KK)
        recorrido(KK, MM)
    elif np.array_equal(vref, KP):
        recorrido(KP, Gamma)
        recorrido(Gamma, KK)
        recorrido(KK, MM)
        recorrido(MM, KP)
    return None

# Obtener puntos k
kpathx = np.array([],dtype=np.float64)
kpathy = np.array([],dtype=np.float64)
klen   = np.array([],dtype=np.float64)
klenref = 0.0
kpos    = [0.0]
klabels = []
if np.array_equal(vref, Gamma):
    klabels = [r'$\Gamma$', r'$K$', r'$M$', r'$K^\prime$', r'$\Gamma$']
elif np.array_equal(vref, KK):
    klabels = [r'$K$', r'$M$', r'$K^\prime$', r'$\Gamma$', r'$K$']
elif np.array_equal(vref, MM):
    klabels = [r'$M$', r'$K^\prime$', r'$\Gamma$', r'$K$', r'$M$']
elif np.array_equal(vref, KP):
    klabels = [r'$K^\prime$', r'$\Gamma$', r'$K$', r'$M$', r'$K^\prime$']

recorrido_completo()

# Obtener energias
Enarr_TB = np.zeros((kpathx.size,2),dtype=np.float64)
Enarr_kp = np.zeros((kpathx.size,2),dtype=np.float64)
ik=0
while(ik < kpathx.size):
    HH_TB = Gfull(kpathx[ik],kpathy[ik],mass,valle)
    Enarr_TB[ik], eigenvector = np.linalg.eigh(HH_TB)
    HH_kp = Gkp(kpathx[ik],kpathy[ik],mass,valle)
    Enarr_kp[ik], eigenvector = np.linalg.eigh(HH_kp)
    ik+=1

# Gráfica de las bandas con etiquetas de simetría
f, ax = plt.subplots(figsize=(5, 3))
ax.set_ylabel(r'$E\,(\mathrm{eV})$')
ax.set_xlabel(r'$q\,(\mathrm{\AA^{-1}})$')
ax.set_xticks(kpos)
ax.set_xticklabels(klabels)
# Gráficas para el modelo Tight Binding
ax.plot(klen, Enarr_TB[:, 1], 'b-', label="TB")
ax.plot(klen, Enarr_TB[:, 0], 'b-')
# Gráficas para el modelo k.p
ax.plot(klen, Enarr_kp[:, 1], 'r-', label="k·p")
ax.plot(klen, Enarr_kp[:, 0], 'r-')

ax.legend(loc='upper right')

# Líneas verticales en puntos de alta simetría
for kp in kpos:
    ax.axvline(kp, color='black', ls='--', lw=0.5)


# Guardar grafica
if np.array_equal(vref,Gamma):
    ref="Gamma"
elif np.array_equal(vref,KK):
    ref="KK"
elif np.array_equal(vref,KP):
    ref="KP"
elif np.array_equal(vref,MM):
    ref="MM"
if valle==+1:
    punto_simetria="+"
elif valle==-1:
    punto_simetria="-"
plt.savefig('./graphene_TB_kp_' + ref + '_' + punto_simetria + '.png', dpi=300)
plt.show()
plt.close()
