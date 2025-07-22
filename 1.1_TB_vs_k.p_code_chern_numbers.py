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
a0 = 1.42               # (Å)
t = 3.2                 # (eV)
v_F = cc/300.0          # (Å/s)
a = np.sqrt(3.0)*a0
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
NBZ = 100       # Discretizacion de la BZ
mass = 1        # Medio Gap (eV)  
valle = +1      # valle elegido (+ o -)
ModeloHamilton=int(input("Modelos:\n 1. Tigth Binding \n 2. k.p theory\n Ingrese el numero de modelo: "))

# Tight Binding
def Gfull(x,y,mass,valle): 
    HH=np.zeros((2,2),dtype=np.complex128)
    phi = Phi(x,y)
    HH[0,0] = mass
    HH[1,1] = -mass
    if(valle==1):
        HH[0,1] = t*phi
        HH[1,0] = t*np.conjugate(phi)
    else:
        HH[0,1] = t*np.conjugate(phi)
        HH[1,0] = t*phi
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
        HH[0,1] = hbar*v_F*((x-KK[0])-img*(y-KK[1]))
        HH[1,0] = hbar*v_F*((x-KK[0])+img*(y-KK[1]))
    elif dist_KP < dist_KK:
        HH[0,1] = hbar*v_F*((x-KP[0])+img*(y-KP[1]))
        HH[1,0] = hbar*v_F*((x-KP[0])-img*(y-KP[1]))
    return HH

# BZ discreta
def BZ_TB(N):
    vx = b1/N               
    vy = b2/N                
    BZ = np.zeros((N,N,2),dtype=np.float64)
    jy = 0
    while(jy<N):
        jx = 0
        while(jx<N):
            kk = vx*jx + vy*jy
            BZ[jx,jy] = kk
            jx+=1
        jy+=1
    return BZ
BZ=BZ_TB(NBZ)


# Eigenproblem (banda{v:0,c:1}, dimension{x:0,y:1})

# Earr(kx,ky,banda): Eigenvalores
Earr = np.zeros((BZ.shape[0],BZ.shape[1],2),dtype=np.float64)
# Parr(kx,ky,dimension,banda): Eigenestados      
Parr = np.zeros((BZ.shape[0],BZ.shape[1],2,2),dtype=np.complex128)
# Uarr(kx,ky,dimension,banda): Funcion U   
Uarr = np.zeros((BZ.shape[0],BZ.shape[1],2,2),dtype=np.complex128)
# Farr(kx,ky,banda): Curvatura de Berry
Farr = np.zeros((BZ.shape[0],BZ.shape[1],2),dtype=np.complex128) 

# Eigenvalores
ik1 = 0
while(ik1<BZ.shape[0]): # filas
    ik2 = 0
    while(ik2<BZ.shape[1]): # columnas
        kk = BZ[ik1,ik2]
        if ModeloHamilton==1:
            HH = Gfull(kk[0],kk[1], mass, valle)
        elif ModeloHamilton==2:
            HH = Gkp(kk[0],kk[1], mass, valle)
        Earr[ik1,ik2], Parr[ik1,ik2] = np.linalg.eigh(HH)
        ik2+=1
    ik1+=1
Eigen_v=Parr[:,:,:,0]
Eigen_c=Parr[:,:,:,1]

# Funcion U   
ky = 0
while(ky<BZ.shape[1]): # filas
    kx = 0
    while(kx<BZ.shape[0]): # columnas
        # valencia, kx
        if (kx+1 < BZ.shape[0]):
            dum = np.dot(np.conjugate(np.transpose(Eigen_v[kx,ky,:])), Eigen_v[kx+1,ky,:])
        else:
            dum = np.dot(np.conjugate(np.transpose(Eigen_v[kx,ky,:])), Eigen_v[0,ky,:])
        Uarr[kx,ky,0,0] = dum/np.absolute(dum)
        # valencia, ky
        if (ky+1 < BZ.shape[1]):
            dum = np.dot(np.conjugate(np.transpose(Eigen_v[kx,ky,:])), Eigen_v[kx,ky+1,:])
        else:
            dum = np.dot(np.conjugate(np.transpose(Eigen_v[kx,ky,:])), Eigen_v[kx,0,:]) 
        Uarr[kx,ky,1,0] = dum/np.absolute(dum)
        # conduccion, kx
        if (kx+1 < BZ.shape[0]):
            dum = np.dot(np.conjugate(np.transpose(Eigen_c[kx,ky,:])), Eigen_c[kx+1,ky,:])
        else:
            dum = np.dot(np.conjugate(np.transpose(Eigen_c[kx,ky,:])), Eigen_c[0,ky,:])
        Uarr[kx,ky,0,1] = dum/np.absolute(dum)
        # conduccion, ky
        if (ky+1 < BZ.shape[1]):
            dum = np.dot(np.conjugate(np.transpose(Eigen_c[kx,ky,:])), Eigen_c[kx,ky+1,:])
        else:
            dum = np.dot(np.conjugate(np.transpose(Eigen_c[kx,ky,:])), Eigen_c[kx,0,:])
        Uarr[kx,ky,1,1] = dum/np.absolute(dum)
        kx+=1
    ky+=1

del(Earr,Parr)

# Curvatura de Berry
ky = 0
while(ky<BZ.shape[0]): # filas
    if (ky+1 < BZ.shape[1]):
        kyplus1 = ky+1
    else:
        kyplus1 = 0  
    kx = 0
    while(kx<BZ.shape[0]): # columnas  
        if (kx+1 < BZ.shape[0]):
            kxplus1 = kx+1
        else:
            kxplus1 = 0
        Farr[kx,ky,0] = np.emath.log((Uarr[kx,ky,0,0] * Uarr[kxplus1,ky,1,0]) / (Uarr[kx,kyplus1,0,0] * Uarr[kx,ky,1,0]))
        Farr[kx,ky,1] = np.emath.log((Uarr[kx,ky,0,1] * Uarr[kxplus1,ky,1,1]) / (Uarr[kx,kyplus1,0,1] * Uarr[kx,ky,1,1]))
        kx+=1
    ky+=1

del(Uarr)

# Plot de Curvatura de Berry
kxarr = np.array([],dtype=np.float64)
kyarr = np.array([],dtype=np.float64)
BCurvC = np.array([],dtype=np.float64)
BCurvV = np.array([],dtype=np.float64)
ky = 0
while(ky<BZ.shape[1]): # filas
    kx = 0
    while(kx<BZ.shape[0]): # columnas    
        kxarr  = np.append(kxarr, BZ[kx,ky,0])
        kyarr  = np.append(kyarr, BZ[kx,ky,1])
        BCurvV = np.append(BCurvV, Farr[kx,ky,0])
        BCurvC = np.append(BCurvC, Farr[kx,ky,1])
        kx+=1
    ky+=1

del(Farr)

# Calculos de numeros de Chern
ChC = np.sum(BCurvC)/(2.0*np.pi*img)
ChV = np.sum(BCurvV)/(2.0*np.pi*img)
# Calculos de numeros de Chern en valle especifico
ChCv =-np.sign(valle)*np.sum(np.absolute(np.imag(BCurvC)))/(2.0*np.pi)
ChVv = np.sign(valle)*np.sum(np.absolute(np.imag(BCurvV)))/(2.0*np.pi)

# Imprimir numeros de Chern
print('Chern numbers:')
print('>>> Conduction:', ChC)
print('>>> Valence   :', ChV)
if(valle==+1):
    print('Chern number at K_+ valley:')
    print('>>> Conduction:', ChCv)
    print('>>> Valence   :', ChVv)
elif(valle==-1):
    print('Chern number at K_- valley:')
    print('>>> Conduction:', ChCv)
    print('>>> Valence   :', ChVv)

# Crear figura y subplots
fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10, 9))
# Títulos de cada subplot
titles = ["Real (Berry_CurvC)", "Imaginary (Berry_CurvC)", "Real (Berry_CurvV)", "Imaginary (Berry_CurvV)"]
for i, a in enumerate(ax.flatten()):
    a.set_title(titles[i])
# Etiquetas
ax[1, 0].set_xlabel(r"$q_x (\mathrm{\AA^{-1}})$")
ax[1, 1].set_xlabel(r"$q_x (\mathrm{\AA^{-1}})$")
ax[0, 0].set_ylabel(r"$q_y (\mathrm{\AA^{-1}})$")
ax[1, 0].set_ylabel(r"$q_y (\mathrm{\AA^{-1}})$")
# Contorno
bar10 = ax[1, 0].tricontourf(kxarr, kyarr, np.real(BCurvV), levels=100, cmap='RdBu')
bar11 = ax[1, 1].tricontourf(kxarr, kyarr, np.imag(BCurvV), levels=100, cmap='RdBu')
bar00 = ax[0, 0].tricontourf(kxarr, kyarr, np.real(BCurvC), levels=100, cmap='RdBu')
bar01 = ax[0, 1].tricontourf(kxarr, kyarr, np.imag(BCurvC), levels=100, cmap='RdBu')
# Colores
fig.colorbar(bar00, ax=ax[0, 0], shrink=0.8)
fig.colorbar(bar01, ax=ax[0, 1], shrink=0.8)
fig.colorbar(bar10, ax=ax[1, 0], shrink=0.8)
fig.colorbar(bar11, ax=ax[1, 1], shrink=0.8)
# Texto para los números de Chern
chern_text = fig.text(0.5, 0.005, "", ha='center', fontsize=12, bbox=dict(facecolor='beige', alpha=0.7))
chern_text.set_text(f"Chern Numbers:\n C_C = {ChCv:.2f}, C_V = {ChVv:.2f}\n m = {mass:.2f} eV")
# Enmarcar puntos K y K'
for i in range(2):
        for j in range(2):
            ax[i, j].scatter(KK[0], KK[1], color='yellow', label='$K$', marker='.', s=100, edgecolors='white')
            ax[i, j].scatter(KP[0], KP[1], color='green', label='$K\'$', marker='.', s=100, edgecolors='white')
            ax[i, j].legend()
# Guardar grafica
if valle==+1:
    punto_simetria="+"
elif valle==-1:
    punto_simetria="-"
if ModeloHamilton==1:
    modelo="TB"
elif ModeloHamilton==2:
    modelo="kp"
plt.savefig('./BerryCurv_'+modelo+'_'+punto_simetria+'.png',dpi=300)
plt.show()
plt.close()


