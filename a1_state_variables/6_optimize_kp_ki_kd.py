import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tf2ss, cont2discrete
from scipy.optimize import differential_evolution
import time

# ========================
# 1. Definición de la funcion de transferencia w_r/Va (velocidad angular)
# ========================
 
K = 3.8181  # Ganancia del sistema
T1 = 0.006584   
T2 = 0.046201   

# Como tenemos la velocidad angular, y queremos controlar el angulo, necesitamos integrarlo para obtener su respectiva TF.
# Esto se hace multiplicando la TF w_r/Va por 1/s.  
num = [K]  # Numerador del sistema (ganancia)
den = np.convolve([T1, 1], [T2, 1])  # Denominador: producto de (T1*s + 1)(T2*s + 1)
den = np.convolve(den, [1, 0])  # Multiplicamos por 's' para agregar un integrador.

# Convertimos el sistema en espacio de estados:
A_c, B_c, C_c, D_c = tf2ss(num, den)

# Definimos el tiempo de muestreo para discretizar el sistema (5 ms por muestra):
dt = 0.005
(A, B, C, D, _) = cont2discrete((A_c, B_c, C_c, D_c), dt)

# ========================
# 2. Función PID discreto incremental
# ========================

def pid_discreto_incremental(Kp, Ki, Kd, dt):
    """
    Calcula los coeficientes del PID incremental discreto.
    """
    a0 = Kp + Ki * dt + Kd / dt
    a1 = -Kp - 2 * Kd / dt
    a2 = Kd / dt
    return a0, a1, a2

# ========================
# 3. Simulador del sistema en lazo cerrado
# ========================

# Se toma en cuenta el torque de la imagen 1-3, pero se podria reducir los tiempos y optimizar
# la ejecución.
def simular_pid(Kp, Ki, Kd, referencia=1.0, t_sim=3500.0, 
                torque_start=950.0, torque_end=2150.0, torque_amp=0.12):
    """
    Simula la respuesta del sistema bajo control PID discreto incremental.
    """
    N = int(t_sim/dt)  # Número total de muestras
    t = np.linspace(0, t_sim, N)  # Vector de tiempo

    # Inicializamos las señales:
    u = np.zeros(N)  # Señal de control
    y = np.zeros(N)  # Salida del sistema (ángulo)
    e = np.zeros(N)  # Error entre referencia y salida
    torque = np.zeros(N)  # Perturbación externa (torque)

    # Se define una perturbación (torque) en un intervalo de tiempo:
    idx_start = int(torque_start/dt)
    idx_end = int(torque_end/dt)
    torque[idx_start:idx_end] = torque_amp

    # Calculamos los coeficientes del PID discreto:
    a0, a1, a2 = pid_discreto_incremental(Kp, Ki, Kd, dt)

    # Inicializamos el estado del sistema:
    x = np.zeros(A.shape[0])

    # Inicializamos las variables del PID:
    e_k1 = 0  # Error anterior
    e_k2 = 0  # Error dos pasos atrás
    u_k1 = 0  # Control anterior

    # Simulación paso a paso:
    for k in range(N-1):
        # Calculamos el error actual:
        e[k] = referencia - y[k]

        # Actualizamos el control (incremental):
        delta_u = a0 * e[k] + a1 * e_k1 + a2 * e_k2
        u[k] = u_k1 + delta_u

        # Limitamos la señal de control:
        u[k] = np.clip(u[k], -100, 100)

        # Entrada al sistema = control - perturbación
        entrada = u[k] - torque[k]

        # Actualizamos el estado del sistema:
        x = A @ x + B.flatten() * entrada
        y[k+1] = C.flatten() @ x + D.item() * entrada

        # Actualizamos las variables del PID:
        e_k2 = e_k1
        e_k1 = e[k]
        u_k1 = u[k]

    return t, y, u, e

# ========================
# 4. Función para calcular el tiempo de establecimiento
# ========================

def tiempo_establecimiento(t, y, tolerancia=0.02, referencia=1.0):
    """
    Calcula el tiempo en que la salida se mantiene dentro de un margen
    del 2% alrededor de la referencia.
    """
    margen = tolerancia * referencia
    for i in range(len(y)-1, -1, -1):
        if np.abs(y[i] - referencia) > margen:
            return t[i+1]
    return t[0]

# ========================
# 5. Función de costo para la optimización
# ========================

def indice_custom(KpKiKd):
    """
    Esta funcion de costo mide qué tan buena es una combinación de Kp, Ki y Kd.
    Penaliza saturaciones (de la acción de control, mayores a 50V), sobrepasos (mayores a 15%), errores grandes y tiempos lentos.
    """
    Kp, Ki, Kd = KpKiKd
    t, y, u, e = simular_pid(Kp, Ki, Kd)

    # Penalizar si hay valores numéricos inválidos:
    if np.isnan(np.sum(e)) or np.isinf(np.sum(e)):
        return 1e10

    # Penalizar sobrepaso grande (>15% de la referencia):
    if np.max(y) > 1.15:
        return 1e6 + np.max(y)

    # Penalizar saturación de control (>50V):
    if np.max(np.abs(u)) > 50:
        return 1e5 + np.max(np.abs(u))

    # Índices de de inicio y fin de torque:
    idx_torque_start = int(950.0 / dt)
    idx_torque_end = int(2150.0 / dt)

    # Error ponderado antes, durante y después de la perturbación:
    # ITAE = tamaño del error en el tiempo (tiempo x error)
    # Primero solo consideraba el ITAE ante la entrada y me quedaba mal el control frente al torque

    itae_inicio = np.sum(t[:idx_torque_start] * np.abs(e[:idx_torque_start])) # ante la entrada del escalon
    itae_perturbacion = np.sum(t[idx_torque_start:idx_torque_end] * np.abs(e[idx_torque_start:idx_torque_end])) # ante el torque
    itae_despues = np.sum(t[idx_torque_end:] * np.abs(e[idx_torque_end:])) # luego del torque

    # Tiempo de establecimiento:
    t_est = tiempo_establecimiento(t, y)

    # Cálculo del costo final (ponderaciones elegidas):
    costo = 1.0 * itae_inicio + 1.5 * itae_perturbacion + 0.5 * itae_despues + 0.5 * t_est

    # Se muestra el progreso:
    print(f"Progreso: Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f}, Costo={costo:.2f}")

    return costo

# ========================
# 6. Optimización por Evolución Diferencial
# ========================

# Definimos los rangos de búsqueda para cada parámetro:
bounds = [(0.01, 5), (0.001, 0.5), (0.01, 2)]

# Ejecutamos la optimización:
result = differential_evolution(
    indice_custom, 
    bounds, 
    strategy='best1bin', 
    popsize=15, 
    maxiter=1000, 
    workers=-1  # Para usar todos los nucleos del CPU
)

# Guardamos el mejor conjunto de parámetros encontrados:
Kp_opt, Ki_opt, Kd_opt = result.x
print(f"\n--- PID Óptimo encontrado ---")
print(f"Kp = {Kp_opt:.4f}, Ki = {Ki_opt:.4f}, Kd = {Kd_opt:.4f}")

# ========================
# 7. Comparativa con otros métodos de ajuste
# ========================

# PID inicial manual:
# Kp_ini = 0.1
# Ki_ini = 0.01  # muy malo, genera muchos problemas al visualizar
# Kd_ini = 5

# PID por método clásico de Ziegler-Nichols:
Ku = 1.5  # Ganancia última
Tu = 0.8  # Período de oscilación
Kp_zn = 0.6 * Ku
Ki_zn = 2 * Kp_zn / Tu
Kd_zn = Kp_zn * Tu / 8

# Simulaciones para cada PID:
# t, y_ini, u_ini, e_ini = simular_pid(Kp_ini, Ki_ini, Kd_ini)
t, y_opt, u_opt, e_opt = simular_pid(Kp_opt, Ki_opt, Kd_opt)
t, y_zn, u_zn, e_zn = simular_pid(Kp_zn, Ki_zn, Kd_zn)

# ========================
# 8. Graficación de Resultados
# ========================

plt.figure(figsize=(12,6))
#plt.plot(t, y_ini, label=f"PID Inicial (Kp={Kp_ini}, Ki={Ki_ini}, Kd={Kd_ini})")
plt.plot(t, y_opt, label=f"PID Optimizado (Kp={Kp_opt:.3f}, Ki={Ki_opt:.3f}, Kd={Kd_opt:.3f})")
plt.plot(t, y_zn, label=f"PID Z-N (Kp={Kp_zn:.3f}, Ki={Ki_zn:.3f}, Kd={Kd_zn:.3f})")
plt.axhline(1.0, color='k', linestyle='--', label='Referencia 1 rad')
plt.xlabel('Tiempo [s]')
plt.ylabel('Ángulo [rad]')
plt.title('Comparativa Respuesta en Lazo Cerrado')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

