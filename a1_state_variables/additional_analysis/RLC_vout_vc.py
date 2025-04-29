import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# FUNCIÓN DE SIMULACIÓN CON RUNGE-KUTTA 4° ORDEN
# ==========================================================
def simular_vc(R, L, C, dt, T_total):
    # Matrices del sistema (salida = voltaje del capacitor)
    A = np.array([[-R/L, -1/L],
                  [1/C,   0  ]])
    b = np.array([[1/L],
                  [0  ]])
    cT = np.array([[0, 1]])  # Elegimos como salida el voltaje en el capacitor

    def entrada_u(t): return 12  # Escalón constante

    N = int(T_total / dt)
    t = np.linspace(0, T_total, N)
    x = np.zeros((2, N))
    y = np.zeros(N)

    for k in range(N - 1):
        def f(x_vec, t_local):
            return A @ x_vec + b.flatten() * entrada_u(t_local)

        k1 = f(x[:, k], t[k])
        k2 = f(x[:, k] + 0.5 * dt * k1, t[k] + 0.5 * dt)
        k3 = f(x[:, k] + 0.5 * dt * k2, t[k] + 0.5 * dt)
        k4 = f(x[:, k] + dt * k3, t[k] + dt)

        x[:, k + 1] = x[:, k] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        y[k] = (cT @ x[:, k]).item()

    y[-1] = (cT @ x[:, -1]).item()
    return t, y

# ==========================================================
# PARÁMETROS BASE Y CONFIGURACIÓN
# ==========================================================
R0 = 220        # Ohmios
L0 = 0.5        # Henrios
C0 = 2.2e-6     # Faradios
T_total = 0.05  # 50 ms
dt = 1e-5       # Paso fijo

# Valores para cada parámetro
valores_R = np.linspace(100, 1000, 6)
valores_L = np.linspace(0.1, 1.0, 6)
valores_C = np.linspace(0.5e-6, 5e-6, 6)

# ==========================================================
# FIGURA CON SUBPLOTS - SALIDA: TENSIÓN EN EL CAPACITOR
# ==========================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Colormaps
colores_R = plt.cm.Reds(np.linspace(0.3, 1, len(valores_R)))
colores_L = plt.cm.Blues(np.linspace(0.3, 1, len(valores_L)))
colores_C = plt.cm.Greens(np.linspace(0.3, 1, len(valores_C)))

# --------------------------
# Variación de R
# --------------------------
for i, R in enumerate(valores_R):
    t, vc = simular_vc(R, L0, C0, dt, T_total)
    axes[0].plot(t * 1000, vc, label=f'R = {int(R)} Ω', color=colores_R[i])
axes[0].set_title('Tensión en el capacitor $V_C(t)$ al variar R')
axes[0].set_ylabel('Voltaje (V)')
axes[0].grid(True)
axes[0].legend()

# --------------------------
# Variación de L
# --------------------------
for i, L in enumerate(valores_L):
    t, vc = simular_vc(R0, L, C0, dt, T_total)
    axes[1].plot(t * 1000, vc, label=f'L = {L:.2f} H', color=colores_L[i])
axes[1].set_title('Tensión en el capacitor $V_C(t)$ al variar L')
axes[1].set_ylabel('Voltaje (V)')
axes[1].grid(True)
axes[1].legend()

# --------------------------
# Variación de C
# --------------------------
for i, C in enumerate(valores_C):
    t, vc = simular_vc(R0, L0, C, dt, T_total)
    axes[2].plot(t * 1000, vc, label=f'C = {C*1e6:.2f} µF', color=colores_C[i])
axes[2].set_title('Tensión en el capacitor $V_C(t)$ al variar C')
axes[2].set_ylabel('Voltaje (V)')
axes[2].set_xlabel('Tiempo (ms)')
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()
plt.show()




# Variando R: A mayor R -> más amortiguado → el capacitor carga suavemente.

# Variando L: A mayor L -> el capacitor carga más lento.

# Variando C: A mayor C → el capacitor tarda más en alcanzar el valor final, y la frecuencia natural baja.
