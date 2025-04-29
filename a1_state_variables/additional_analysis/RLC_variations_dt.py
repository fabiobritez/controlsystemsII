import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# PARÁMETROS DEL CIRCUITO RLC
# ==========================================================
R = 220           # Ohmios
L = 0.5           # Henrios
C = 2.2e-6        # Faradios

# MATRICES DEL SISTEMA
A = np.array([[-R/L, -1/L],
              [1/C,   0  ]])
b = np.array([[1/L],
              [0  ]])
cT = np.array([[R, 0]])

# ==========================================================
# ANÁLISIS DE LOS POLOS PARA DETERMINAR DINÁMICA
# ==========================================================
polos = np.linalg.eigvals(A)
omega_d = np.max(np.abs(np.imag(polos)))     # rad/s
f_d = omega_d / (2 * np.pi)                  # Hz
T = 1 / f_d                                  # Período de oscilación
print(f"Polos: {polos}")
print(f"Frecuencia dominante: {f_d:.2f} Hz | Período: {T*1000:.2f} ms")

# ==========================================================
# PARÁMETROS DE SIMULACIÓN
# ==========================================================
T_total = 0.03  # 30 ms de simulación
divisiones = np.linspace(10, 100, 6, dtype=int)
colores = plt.cm.Blues(np.linspace(0.4, 1, len(divisiones)))

# ==========================================================
# ENTRADA ESCALÓN CAMBIANTE
# ==========================================================
def entrada_u(tiempo):
    return 12 * (-1)**int(tiempo // 0.01)

# ==========================================================
# SIMULACIÓN CON EULER (RETORNA TAMBIÉN u(t))
# ==========================================================
def simular(dt):
    N = int(T_total / dt)
    t = np.linspace(0, T_total, N)
    x = np.zeros((2, N))
    y = np.zeros(N)
    u_vals = np.zeros(N)

    for k in range(N - 1):
        u = entrada_u(t[k])
        u_vals[k] = u
        dx = A @ x[:, k] + b.flatten() * u
        x[:, k + 1] = x[:, k] + dx * dt
        y[k] = (cT @ x[:, k]).item()

    u_vals[-1] = entrada_u(t[-1])
    y[-1] = (cT @ x[:, -1]).item()
    return t, y, u_vals

# ==========================================================
# GRÁFICA DE LA SALIDA Y(T) + ENTRADA U(T)
# ==========================================================
plt.figure(figsize=(10, 6))

for i, div in enumerate(divisiones):
    dt_i = T / div
    t_i, y_i, u_i = simular(dt_i)
    label = f'dt = T/{div}'
    plt.plot(t_i * 1000, y_i, label=label, color=colores[i], alpha=0.9)

# Dibujar una sola curva de entrada u(t) (punta negra)
plt.plot(t_i * 1000, u_i, 'k--', label='Entrada $u(t)$', linewidth=1)

plt.title('Comparación de la salida $y(t)$ con diferentes pasos de integración')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Voltaje (V)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

