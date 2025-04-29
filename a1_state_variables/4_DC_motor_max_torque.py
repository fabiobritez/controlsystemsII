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
t_final = 0.1           # Tiempo total (s)
N = int(t_final / dt)   # Número de pasos
time = np.linspace(0, t_final, N)
v_a = 12                # Voltaje aplicado (V)
T_L = np.zeros(N)       # Torque de carga

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
theta = x[2, :]
torque = K_i * ia
torque_max = np.max(torque)

fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Velocidad angular
axs[0].plot(time, omega, color='black')
axs[0].set_ylabel("Velocidad $\\omega$ (rad/s)")
axs[0].set_title("Velocidad angular del motor")
axs[0].grid()

# Posición angular
axs[1].plot(time, theta, color='black')
axs[1].set_ylabel("Posición $\\theta$ (rad)")
axs[1].set_title("Posición angular del motor")
axs[1].grid()

# Torque electromagnético
axs[2].plot(time, torque, color='black')
axs[2].set_xlabel("Tiempo (s)")
axs[2].set_ylabel("Torque (Nm)")
axs[2].set_title("Torque electromagnético")
axs[2].grid()

# Marcar el torque máximo en rojo
torque_max_idx = np.argmax(torque)  # índice del torque máximo
axs[2].plot(time[torque_max_idx], torque_max, 'ro')  # punto rojo
axs[2].annotate(f'{torque_max:.2e} Nm',
                xy=(time[torque_max_idx], torque_max),
                xytext=(time[torque_max_idx] + 0.01, torque_max),
                arrowprops=dict(arrowstyle="->", color='red'),
                color='red')

plt.tight_layout()
plt.show()

# ========================
# Resultado final
# ========================
print(f"Torque máximo alcanzado: {torque_max:.6f} Nm")

