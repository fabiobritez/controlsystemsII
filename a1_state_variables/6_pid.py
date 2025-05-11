import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tf2ss, cont2discrete

# === Parámetros del modelo identificado ===
K = 3.8181 # Va
T1 = 0.006584 # Va
T2 = 0.046201 #Va

# === Planta continua incluyendo la integración (ángulo) ===
num = [K]
den = np.convolve([T1, 1], [T2, 1])
den = np.convolve(den, [1, 0])  # Integrador

A_c, B_c, C_c, D_c = tf2ss(num, den)
dt = 0.005
(A, B, C, D, _) = cont2discrete((A_c, B_c, C_c, D_c), dt)

# === Función PID discreto incremental ===
def pid_discreto_incremental(Kp, Ki, Kd, dt):
    a0 = Kp + Ki * dt + Kd / dt
    a1 = -Kp - 2 * Kd / dt
    a2 = Kd / dt
    return a0, a1, a2


def generar_torque(t, torque_start=10.0, torque_end=45.0, torque_amp=0.42):
    torque = np.zeros_like(t)
    idx_start = int(torque_start / dt)
    idx_end = int(torque_end / dt)
    torque[idx_start:idx_end] = torque_amp
    return torque


# === Simulador en lazo cerrado ===
def simular_pid(Kp, Ki, Kd, torque, referencia=1.0, t_sim=60.0):
    N = int(t_sim/dt)
    t = np.linspace(0, t_sim, N)

    u = np.zeros(N)
    y = np.zeros(N)
    e = np.zeros(N)

    a0, a1, a2 = pid_discreto_incremental(Kp, Ki, Kd, dt)

    x = np.zeros(A.shape[0])

    e_k1 = 0
    e_k2 = 0
    u_k1 = 0

    for k in range(N-1):
        e[k] = referencia - y[k]

        delta_u = a0 * e[k] + a1 * e_k1 + a2 * e_k2
        u[k] = u_k1 + delta_u
        u[k] = np.clip(u[k], -100, 100)

        entrada = u[k] - torque[k]

        x = A @ x + B.flatten() * entrada
        y[k+1] = C.flatten() @ x + D.item() * entrada

        e_k2 = e_k1
        e_k1 = e[k]
        u_k1 = u[k]

    return t, y, u, e


# === Cálculo del tiempo de establecimiento ===
def tiempo_establecimiento(t, y, tolerancia=0.02, referencia=1.0):
    margen = tolerancia * referencia
    for i in range(len(y)-1, -1, -1):
        if np.abs(y[i] - referencia) > margen:
            if i + 1 < len(t):
                return t[i+1]
            else:
                return t[-1]
    return t[0]

# === Cálculo de métricas ===
def calcular_metricas(t, y, e, referencia=1.0, tolerancia=0.02):
    overshoot = (np.max(y) - referencia) / referencia * 100
    t_est = tiempo_establecimiento(t, y, tolerancia, referencia)
    error_final = np.abs(referencia - y[-1])
    itae = np.sum(t * np.abs(e))
    return overshoot, t_est, error_final, itae

# === Definición de PIDs ===

# PID Optimizado por diferencial evolutiva
# Accion de control menor a 50V
# Sobrepaso menor a 15%
# Minimizacion del tiempo
# Sin error en estado estable

# Kp_opt = 3.0432
# Ki_opt = 0.4039 # Tienen muy buena respuesta al escalon, hasta 50v de u(t) y frente al torque tiene una respuesta con sobrepaso (<20%), lento
# Kd_opt = 0.2351

# Kp_opt = 2.48
# Ki_opt = 0.48  # Estos tienen menos accion de control
# Kd_opt = 0.13

# Kp_opt = 1.03
# Ki_opt = 0.5  # Estos tienen un sobrepaso (al corregir la perturbacion un 20% aprox)
# Kd_opt = 0.12

Kp_opt = 2.9827
Ki_opt = 0.4788
Kd_opt = 0.2408
#

# PID por Ziegler-Nichols
Ku = 1.5
Tu = 0.8
Kp_zn = 0.6 * Ku
Ki_zn = 2 * Kp_zn / Tu
Kd_zn = Kp_zn * Tu / 8

# PID Método clásico tipo Cohen-Coon modificado
Kp_cc = 0.33 * Ku
Ti_cc = 0.5 * Tu
Td_cc = 0.33 * Tu

Ki_cc = Kp_cc / Ti_cc
Kd_cc = Kp_cc * Td_cc


# === Simulaciones ===
t_sim = 60.0
N = int(t_sim/dt)
t = np.linspace(0, t_sim, N)
torque = generar_torque(t)


t, y_opt, u_opt, e_opt = simular_pid(Kp_opt, Ki_opt, Kd_opt, torque)
t, y_zn, u_zn, e_zn = simular_pid(Kp_zn, Ki_zn, Kd_zn, torque)
t, y_cc, u_cc, e_cc = simular_pid(Kp_cc, Ki_cc, Kd_cc, torque)

# === Cálculo de métricas ===
metricas_opt = calcular_metricas(t, y_opt, e_opt)
metricas_zn = calcular_metricas(t, y_zn, e_zn)
metricas_cc = calcular_metricas(t, y_cc, e_cc)

# === Mostrar métricas ===
print("\n=== Comparativa de Desempeño ===")
print("Controlador\tOvershoot [%]\tT. Establecimiento [s]\tError Final\tITAE")
# print(f"Cohen-Coon\t\t{metricas_cc[0]:.2f}\t\t{metricas_cc[1]:.2f}\t\t{metricas_cc[2]:.4f}\t{metricas_cc[3]:.2f}")
print(f"Opt. Diff. Evo.\t{metricas_opt[0]:.2f}\t\t{metricas_opt[1]:.2f}\t\t{metricas_opt[2]:.4f}\t{metricas_opt[3]:.2f}")
print(f"Z-N\t\t{metricas_zn[0]:.2f}\t\t{metricas_zn[1]:.2f}\t\t{metricas_zn[2]:.4f}\t{metricas_zn[3]:.2f}")

# === Gráficas ===
plt.figure(figsize=(10, 9))

# Respuesta del sistema
plt.subplot(3, 1, 1)
# plt.plot(t, y_cc, label=f"PID Cohen-Coon")
plt.plot(t, y_opt, label=f"PID Optimizado")
plt.plot(t, y_zn, label=f"PID Z-N")
plt.axhline(1.0, color='k', linestyle='--', label='Referencia')
plt.ylabel('Ángulo [rad]')
plt.title('Comparativa: Respuesta del Sistema')
plt.legend()
plt.grid()

# Señal de control
plt.subplot(3, 1, 2)
#plt.plot(t, u_cc, label=f"u Cohen-Coon")
plt.plot(t, u_opt, label=f"u Optimizado")
plt.plot(t, u_zn, label=f"u Z-N")
plt.ylabel('Señal de Control u(t)')
plt.title('Comparativa: Señal de Control')
plt.legend()
plt.grid()

# Torque aplicado
plt.subplot(3, 1, 3)
plt.plot(t, torque, label="Torque aplicado", color='r')
plt.ylabel('Torque [Nm]')
plt.xlabel('Tiempo [s]')
plt.title('Torque Aplicado')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

