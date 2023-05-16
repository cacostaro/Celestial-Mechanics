import numpy as np
import pandas as pd

Kcte = 0.01720209908

def DetAlpha(time):
    hours, minutes, seconds = int(time.split('h')[0]), int(time.split('h')[1].split('m')[0]), float(time.split('m')[1].split('s')[0])
    time = (hours + (minutes/60) + (seconds/3600))*15
    return time

def DetDelta(delta):
    grados, minutos, segundos = delta.split("°")[0], delta.split("'")[0].split(" ")[-1], delta.split(" ")[-1].split("''")[0]
    delta = - (int(grados) + (int(minutos)/60) + (float(segundos)/3600))
    return delta

#####################################               Modify               ######################################

#t1 = 02 de mayo 0h 0m 0s TU 2023,
t1 = 2460066.5 #JDT para el 2 de mayo
Alpha_t1 = DetAlpha("21h26m2.69s")
Delta_t1 = DetDelta("19° 57' 59.94''")  # Tener cuidado con el - **TRUE
Rho_t1 = 1.536587515

# Componentes de los vectores posicion del sol con respecto a la tierra http://vo.imcce.fr/webservices/miriade/?forms
SolarXPRIMA_t1 = 0.7600151866455
SolarYPRIMA_t1 = 0.6069067436060
SolarZPRIMA_t1 = 0.2630859744948


#t2 = 10 de mayo 6h 0m 0s TU 2023,
t2 = 2460074.75 #JDT para el 10 de mayo
Alpha_t2 = DetAlpha("21h37m06.07s")
Delta_t2 = DetDelta("18° 43' 13.75''") # - **TRUE
Rho_t2 = 1.464105182

# Componentes de los vectores posicion del sol con respecto a la tierra http://vo.imcce.fr/webservices/miriade/?forms
SolarXPRIMA_t2 = 0.6620237347199
SolarYPRIMA_t2 = 0.6993361573355
SolarZPRIMA_t2 = 0.3031490225833


#t3 = 08 de junio 13h 6m 56s TU 2023.
t3 = 2460104.04648 #JDT para el 8 de junio
Alpha_t3 = DetAlpha("22h07m13.29s")
Delta_t3 = DetDelta("14° 07' 40.47''") # - **TRUE
Rho_t3 = 1.175670944

# Componentes de los vectores posicion del sol con respecto a la tierra http://vo.imcce.fr/webservices/miriade/?forms
SolarXPRIMA_t3 = 0.2251235431244
SolarYPRIMA_t3 = 0.9080172090125
SolarZPRIMA_t3 = 0.3936129657371


#####################################             NO  Modify               ######################################
print("__________________________________________________________________\n")
print("...Determinacion de Elementos Orbitales...\n")

##################### Se hallan las componentes del vector geocentrico del objeto #####################

# Calculo de las ξ Prima "Xi"

Xiprim_t1 = Rho_t1*np.cos(np.radians(Alpha_t1))*np.cos(np.radians(Delta_t1))
Xiprim_t2 = Rho_t2*np.cos(np.radians(Alpha_t2))*np.cos(np.radians(Delta_t2))
Xiprim_t3 = Rho_t3*np.cos(np.radians(Alpha_t3))*np.cos(np.radians(Delta_t3))


# Calculo de las η Prima "Eta"

Etaprim_t1 = Rho_t1*np.sin(np.radians(Alpha_t1))*np.cos(np.radians(Delta_t1))
Etaprim_t2 = Rho_t2*np.sin(np.radians(Alpha_t2))*np.cos(np.radians(Delta_t2))
Etaprim_t3 = Rho_t3*np.sin(np.radians(Alpha_t3))*np.cos(np.radians(Delta_t3))


# Calculo las ζ Prima "Zeta"

Zetaprim_t1 = Rho_t1*np.sin(np.radians(Delta_t1))
Zetaprim_t2 = Rho_t2*np.sin(np.radians(Delta_t2))
Zetaprim_t3 = Rho_t3*np.sin(np.radians(Delta_t3))

pd.set_option('display.float_format', '{:.10f}'.format)  # Mostrar 12 decimales
data = {
    'ξ\'': [Xiprim_t1, Xiprim_t2, Xiprim_t3],
    'η\'': [Etaprim_t1, Etaprim_t2, Etaprim_t3],
    'ζ\'': [Zetaprim_t1, Zetaprim_t2, Zetaprim_t3]
}
df = pd.DataFrame(data, index=['t1', 't2', 't3'])
print(" COMPONENTES DEL VECTOR GEOCENTRICO DEL OBJETO")
print(df.to_string(col_space=4, sparsify=True))


##################### Pasamos a coordenadas ecuatoriales heliocentricas #####################

# Calculo de las x prima

Xprim_t1 = Xiprim_t1 - SolarXPRIMA_t1
Xprim_t2 = Xiprim_t2 - SolarXPRIMA_t2
Xprim_t3 = Xiprim_t3 - SolarXPRIMA_t3


# Calculo de las y prima

Yprim_t1 = Etaprim_t1 - SolarYPRIMA_t1
Yprim_t2 = Etaprim_t2 - SolarYPRIMA_t2
Yprim_t3 = Etaprim_t3 - SolarYPRIMA_t3


# Calculo de las z prima

Zprim_t1 = Zetaprim_t1 - SolarZPRIMA_t1
Zprim_t2 = Zetaprim_t2 - SolarZPRIMA_t2
Zprim_t3 = Zetaprim_t3 - SolarZPRIMA_t3

pd.set_option('display.float_format', '{:.10f}'.format)  # Mostrar 10 decimales
data = {
    'x\'': [Xprim_t1, Xprim_t2, Xprim_t3],
    'y\'': [Yprim_t1, Yprim_t2, Yprim_t3],
    'z\'': [Zprim_t1, Zprim_t2, Zprim_t3]
}
df = pd.DataFrame(data, index=['t1', 't2', 't3'])
print("\n=============================================================\n COORDENADAS ECUATORIALES HELIOCENTRICAS")
print(df.to_string(col_space=4, sparsify=True))


##################### Pasamos a coordenadas eclipticas heliocentricas #####################
ePsilon = DetDelta("23° 26' 21.406") * -1
ePsilon = np.radians(ePsilon)
#print(ePsilon)
# Calculamos las x

X_t1 = Xprim_t1
X_t2 = Xprim_t2
X_t3 = Xprim_t3

# Calculamso las y

Y_t1 = Yprim_t1 * np.cos((ePsilon)) + Zprim_t1*np.sin((ePsilon))
Y_t2 = Yprim_t2 * np.cos((ePsilon)) + Zprim_t2*np.sin((ePsilon))
Y_t3 = Yprim_t3 * np.cos((ePsilon)) + Zprim_t3*np.sin((ePsilon))

# Calculamos las z

Z_t1 = Zprim_t1 * np.cos((ePsilon)) - Yprim_t1*np.sin((ePsilon))
Z_t2 = Zprim_t2 * np.cos((ePsilon)) - Yprim_t2*np.sin((ePsilon))
Z_t3 = Zprim_t3 * np.cos((ePsilon)) - Yprim_t3*np.sin((ePsilon))

# Calculamos las r

r_t1 = np.sqrt(X_t1**2+Y_t1**2+Z_t1**2)
r_t2 = np.sqrt(X_t2**2+Y_t2**2+Z_t2**2)
r_t3 = np.sqrt(X_t3**2+Y_t3**2+Z_t3**2)


pd.set_option('display.float_format', '{:.10f}'.format) # Mostrar 10 decimales
data = {
    'x': [X_t1, X_t2, X_t3],
    'y': [Y_t1, Y_t2, Y_t3],
    'z': [Z_t1, Z_t2, Z_t3],
    'r': [r_t1, r_t2, r_t3]
}
df = pd.DataFrame(data, index=['t1', 't2', 't3'])
print("\n=============================================================\n COORDENADAS ECLIPTICAS HELIOCENTRICAS")
print(df.to_string(col_space=4, sparsify=True))


print("\n=============================================================\n PRODUCTO CRUZ r1 Y r2 Y SU NORMA\n")

r1xr2_i = (Y_t1*Z_t2) - (Y_t2*Z_t1)
r1xr2_j = (Z_t1*X_t2) - (Z_t2*X_t1)
r1xr2_k = (X_t1*Y_t2) - (X_t2*Y_t1)
r1xr2 = np.sqrt((r1xr2_i**2) + (r1xr2_j**2) + (r1xr2_k**2))

print(f"r1 X r2 = {r1xr2_i:.10f} i  -  {r1xr2_j:.10f} j  -  {r1xr2_k:.10f} k\n")
print(f"|r1 X r2| = {r1xr2:.10f}\n")


print("=============================================================\n VECTOR MOMENTO ANGULAR (h gorro)\n")

# h = r1 x r2 / |r1 x r2|

h_i = (r1xr2_i)/(r1xr2)
h_j = (r1xr2_j)/(r1xr2)
h_k = (r1xr2_k)/(r1xr2)

B1 = h_i
B2 = h_j
B3 = h_k
print(f"h = {B1:.10f} i  -  {B2:.10f} j  -  {B3:.10f} k\n")

print("=============================================================\n INCLINACION Y LONGITUD DEL NODO ASCENDENTE\n")

i = np.degrees(np.arccos((B3)))
OmegaMa = np.degrees(np.arctan((B1/-B2)))

if i > 90:
    print("La direccion es contraria a los planetas")
else:
    print("La direccion es hacia los planetas,  en contra de las manecillas de reloj")

print(f"i = {i:.10f}    Ω = {OmegaMa:.10f}\n")

print("=============================================================\n VECTORES  _m_  _n_ \n")

n_i = np.cos(np.radians(OmegaMa))
n_j = np.sin(np.radians(OmegaMa))
n_k = 0
print(f"n = {n_i:.10f} i  +  {n_j:.10f} j  +  {n_k:.10f} k\n")

m_i = -np.sin(np.radians(OmegaMa)) * np.cos(np.radians(i))
m_j = np.cos(np.radians(OmegaMa)) * np.cos(np.radians(i))
m_k = np.sin(np.radians(i))
print(f"m = {m_i:.10f} i  +  {m_j:.10f} j  +  {m_k:.10f} k\n")

print("=============================================================\n SISTEMA LINEAL 2X2 PARA LOS VECTORES: r1 - r2  -----  r1 - r3 \n")

r1LESSr2_i = X_t1 - X_t2
r1LESSr2_j = Y_t1 - Y_t2  
r1LESSr2_k = Z_t1 - Z_t2 
print(f"r1 - r2 = {r1LESSr2_i:.10f} i  -  {r1LESSr2_j:.10f} j  +  {r1LESSr2_k:.10f} k\n")

r1LESSr3_i = X_t1 - X_t3
r1LESSr3_j = Y_t1 - Y_t3
r1LESSr3_k = Z_t1 - Z_t3
print(f"r1 - r3 = {r1LESSr3_i:.10f} i  -  {r1LESSr3_j:.10f} j  +  {r1LESSr3_k:.10f} k\n")


print("=============================================================\n OPERACION DE ESCALARES: r2 - r1 ; r3 - r1  \n")

ESr2LESSr1 = r_t2 - r_t1
print(f" r2 - r1 = {ESr2LESSr1:.10f}")

ESr3LESSr1 = r_t3 - r_t1
print(f" r3 - r1 = {ESr3LESSr1:.10f}\n")


print("=============================================================\n COEFICIENTES \n")

r1LESSr2PTO_n = r1LESSr2_i*n_i + r1LESSr2_j*n_j + r1LESSr2_k*n_k
r1LESSr2PTO_m = r1LESSr2_i*m_i + r1LESSr2_j*m_j + r1LESSr2_k*m_k

r1LESSr3PTO_n = r1LESSr3_i*n_i + r1LESSr3_j*n_j + r1LESSr3_k*n_k
r1LESSr3PTO_m = r1LESSr3_i*m_i + r1LESSr3_j*m_j + r1LESSr3_k*m_k

print(f"(r1-r2)·n = {r1LESSr2PTO_n:.10f}")
print(f"(r1-r2)·m = {r1LESSr2PTO_m:.10f}\n%%%%%%%%%%%%%%%%%%%%%%%%%")
print(f"(r1-r3)·n = {r1LESSr3PTO_n:.10f}")
print(f"(r1-r3)·m = {r1LESSr3PTO_m:.10f}")

print("=============================================================\n RESOLVEMOS LAS ECUACIONES DE 2X2 PARA H y K \n") # Variables Auxiliares
# Usamos H=e*cos(ω) y K=e*sen(ω)

matrix1 = np.array([[r1LESSr2PTO_n, r1LESSr2PTO_m], [r1LESSr3PTO_n, r1LESSr3PTO_m]])
matrix2 = np.array([ESr2LESSr1, ESr3LESSr1])

Solv_Matr = np.linalg.solve(matrix1, matrix2) 

H = Solv_Matr[0]
K = Solv_Matr[1]

print(f"H = {H:.10f}\nK = {K:.10F}\n")

print("=============================================================\n ANGULO DE PERIAPSIS ω Y EXCENTRICIDAD e\n")

Ang_Verd = np.degrees(np.arctan2(K, H))
e = (H**2+K**2)**(1/2)

print(f"ω = {Ang_Verd:.10f}\ne = {e:.10f}\n")


print("=============================================================\n SEMIEJE MAYOR a \n")

r1·n = (X_t1*n_i)+(Y_t1*n_j)+(Z_t1*n_k)
r1·m = (X_t1*m_i)+(Y_t1*m_j)+(Z_t1*m_k)
a = (r_t1+(H*r1·n)+(K*r1·m))/(1-(e**2))

print(f"a = {a:.10f} u.a.\n")

print("=============================================================\n ANOMALIA EXCENTRICA Y ANOMALIA MEDIA \n")

E1 = np.degrees(np.arccos((a-r_t1)/(a*e))) # Revisar el criterio de cuadrante
M1 = E1 - e*np.degrees(np.sin(np.radians(E1)))

print(f"E1 = {E1:.10f}\nM1 = {M1:.10f}")

print("=============================================================\n TIEMPO DEL PASO POR EL PERIHELIO \n")

n = (180*Kcte)/(np.pi*a**(3/2))
t0 = t1 - (M1/n)

print(f"t0 = {t0:.11} JDT")