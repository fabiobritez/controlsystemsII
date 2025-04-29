import numpy as np
import matplotlib.pyplot as plt


# ======== PARÁMETROS DEL CIRCUITO RLC =========

R = 220           # Resistencia en ohmios
L = 0.5           # Inductancia en henrios
C = 2.2e-6        # Capacitancia en faradios


# MATRICES DEL SISTEMA EN VARIABLES DE ESTADO

# x1 = corriente por el inductor
# x2 = tensión en el capacitor

A = np.array([[-R/L,   -1/L],
              [1/C,     0  ]])

b = np.array([[1/L],
              [0  ]])

cT = np.array([[R, 0]])  # Salida como caída de tensión en la resistencia


# CÁLCULO AUTOMÁTICO DEL TIEMPO DE INTEGRACIÓN dt

# Se hace un analisis de la dinamica del sistema (polos)

# Paso 1: calcular los polos del sistema (autovalores de A)
polos = np.linalg.eigvals(A)
print("Polos del sistema:", polos)
# Da como resultado polos complejos conjugados.


# Paso 2: frecuencia de oscilación (rad/s y Hz)
# Extraemos la parte imaginaria, su valor absoluto. 
omega_d = np.max(np.abs(np.imag(polos)))  # Parte imaginaria del polo dominante
f_d = omega_d / (2 * np.pi)               # Frecuencia en Hz
T = 1 / f_d                               # Período de oscilación

print(f"Frecuencia dominante: {f_d:.2f} Hz")
print(f"Período aproximado: {T*1000:.2f} ms")

# Paso 3: se elige un paso de integración 
dt = T / 50
print(f"Paso de integración dt = {dt*1e6:.2f} µs")


# SIMULACIÓN CON MÉTODO DE EULER


T_total = 0.05           # Tiempo total de simulación (s)
N = int(T_total / dt)   # Número de pasos

# Inicialización de estados y variables
x = np.zeros((2, N))    # x[0,:] = i(t), x[1,:] = v_C(t)
y = np.zeros(N)         # Salida del sistema
u_values = np.zeros(N)  # Entrada en cada instante
t = np.linspace(0, T_total, N)  # Vector de tiempo

# Función de entrada escalón alternante cada 10 ms
def entrada_u(tiempo):
    return 12 * (-1)**int(tiempo // 0.01)

# Bucle de integración con método de Euler
for k in range(N - 1):
    u = entrada_u(t[k])
    u_values[k] = u
    dx = A @ x[:, k] + b.flatten() * u
    x[:, k + 1] = x[:, k] + dx * dt
    y[k] = (cT @ x[:, k]).item()  # salida y(t) = R * i(t)

# Últimos valores
u_values[-1] = entrada_u(t[-1])
y[-1] = (cT @ x[:, -1]).item()


# GRÁFICAS DE RESULTADOS


plt.figure(figsize=(12, 8))

# Gráfico de la entrada
plt.subplot(3, 1, 1)
plt.plot(t * 1000, u_values, label='Entrada $u(t)$', color='black')
plt.ylabel('Voltaje (V)')
plt.title('Entrada escalón alternante')
plt.grid(True)
plt.legend()

# Gráfico de la salida (tensión en la resistencia)
plt.subplot(3, 1, 2)
plt.plot(t * 1000, y, label='Salida $y(t) = R i(t)$', color='black')
plt.ylabel('Voltaje (V)')
plt.title('Tensión de salida en la resistencia')
plt.grid(True)
plt.legend()

# Gráfico de la tensión del capacitor (x[1,:])
plt.subplot(3, 1, 3)
plt.plot(t * 1000, x[1, :], label='Tensión en el capacitor $v_C(t)$', color='black')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Voltaje (V)')
plt.title('Tensión en el capacitor')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

