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
# ANÁLISIS DE LOS POLOS PARA DEFINIR DINÁMICA Y dt
# ==========================================================
polos = np.linalg.eigvals(A)
omega_d = np.max(np.abs(np.imag(polos)))
f_d = omega_d / (2 * np.pi)
T = 1 / f_d
dt = T / 20
T_total = 0.05  # 50 ms de simulación

print(f"Usando dt = {dt*1e6:.2f} µs")

# ==========================================================
# FUNCIÓN DE ENTRADA ESCALÓN ALTERNANTE
# ==========================================================
def entrada_u(tiempo):
    return 12 * (-1)**int(tiempo // 0.01)

# ==========================================================
# SIMULACIÓN CON EULER
# ==========================================================
def simular_euler(dt):
    N = int(T_total / dt)
    t = np.linspace(0, T_total, N)
    x = np.zeros((2, N))
    y = np.zeros(N)

    for k in range(N - 1):
        u = entrada_u(t[k])
        dx = A @ x[:, k] + b.flatten() * u
        x[:, k + 1] = x[:, k] + dx * dt
        y[k] = (cT @ x[:, k]).item()

    y[-1] = (cT @ x[:, -1]).item()
    return t, y

# ==========================================================
# SIMULACIÓN CON RUNGE-KUTTA 4° ORDEN
# ==========================================================
def simular_rk4(dt):
    N = int(T_total / dt)
    t = np.linspace(0, T_total, N)
    x = np.zeros((2, N))
    y = np.zeros(N)

    for k in range(N - 1):
        u = entrada_u(t[k])
        
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
# COMPARACIÓN DE MÉTODOS: GRAFICAR
# ==========================================================
t_euler, y_euler = simular_euler(dt)
t_rk4, y_rk4 = simular_rk4(dt)

plt.figure(figsize=(10, 6))
plt.plot(t_euler * 1000, y_euler, label='Método de Euler', linestyle='--', color='red')
plt.plot(t_rk4 * 1000, y_rk4, label='Runge-Kutta 4° orden', color='blue')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Voltaje de salida $y(t)$ (V)')
plt.title('Comparación de métodos: Euler vs Runge-Kutta para dt=T/20')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

