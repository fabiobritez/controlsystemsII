import numpy as np
import matplotlib.pyplot as plt

# ========================
# Parámetros del motor DC
# ========================
L_AA = 366e-6     # Inductancia del inducido (H)
J = 5e-9          # Momento de inercia (kg·m²)
R_A = 55.6        # Resistencia del inducido (Ohm)
B = 0             # Fricción viscosa (Nm·s)
K_i = 6.49e-3     # Constante de torque (Nm/A)
K_m = 6.53e-3     # Constante de FEM (V·s/rad)

# ========================
# Matrices del sistema
# ========================
A = np.array([
    [-R_A/L_AA, -K_m/L_AA, 0],
    [K_i/J, -B/J, 0],
    [0, 1, 0]
])

B_mat = np.array([
    [1/L_AA, 0],
    [0, -1/J],
    [0, 0]
])

# ========================
# Configuración de simulación
# ========================
dt = 1e-7               # Paso de tiempo
t_final = 0.4           # Tiempo total (s)
N = int(t_final / dt)   # Número de pasos
time = np.linspace(0, t_final, N)
v_a = 12                # Voltaje aplicado (V)
T_L = np.zeros(N)       # Torque de carga

# Aplicar torque de carga igual al torque máximo desde t = 0.25 s
torque_carga = 0.001392  # Nm
T_L_start_index = int(0.25 / dt)
T_L[T_L_start_index:] = torque_carga

# ========================
# Estado inicial
# ========================
x = np.zeros((3, N))  # [i_a, ω, θ]

# ========================
# Simulación
# ========================
for k in range(N - 1):
    u = np.array([v_a, T_L[k]])  # entrada [voltaje, torque carga]
    dx = A @ x[:, k] + B_mat @ u
    x[:, k + 1] = x[:, k] + dx * dt

# ========================
# Resultados
# ========================
ia = x[0, :]
omega = x[1, :]
torque = K_i * ia

# ========================
# Gráficas
# ========================
plt.figure()
plt.plot(time, ia)
plt.xlabel("Tiempo (s)")
plt.ylabel("Corriente $i_a$ (A)")
plt.title("Corriente del inducido con carga a partir de 0.25 s")
plt.grid()

plt.figure()
plt.plot(time, omega)
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad angular $\\omega$ (rad/s)")
plt.title("Velocidad angular con carga a partir de 0.25 s")
plt.grid()

plt.figure()
plt.plot(time, torque)
plt.axhline(y=torque_carga, color='r', linestyle='--', label='Torque carga')
plt.xlabel("Tiempo (s)")
plt.ylabel("Torque (Nm)")
plt.title("Torque electromagnético")
plt.legend()
plt.grid()

plt.show()

