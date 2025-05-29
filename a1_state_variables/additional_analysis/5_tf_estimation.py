import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lsim, TransferFunction

# === Funciones auxiliares ===

def encontrar_y(t_relativo, retardo, tiempos, salidas):
    """
    Encuentra el valor de salida más cercano a un tiempo relativo más retardo.
    """
    idx = np.argmin(np.abs((t_relativo + retardo) - tiempos))
    return salidas[idx], tiempos[idx] - retardo

def obtener_parametros_por_metodo_chen(tiempos_relativos, retardo, tiempos, salidas_preparadas, amplitud_escalon, y_final_est):
    """
    Método generalizado de Chen para estimar k, T1, T2 y T3.

    IMPORTANTE: La señal 'salidas_preparadas' debe estar tratada adecuadamente afuera. 
    Ya que el metodo admite respuestas a escalones unitarios (positivos) y sin retardo.
    """
    valores, tiempos_ajustados = [], []
    for t in tiempos_relativos:
        y_val, t_ajustado = encontrar_y(t, retardo, tiempos, salidas_preparadas)
        valores.append(y_val)
        tiempos_ajustados.append(t_ajustado)
    
    y1, y2, y3 = valores
    t1, t2, t3 = tiempos_ajustados
    print(f"\n--- Datos de identificación ---")
    print(f"Retardo considerado: {retardo:.6f} s")
    print(f"y1 = {y1:.6f} (en t1 = {t1:.6f} s)")
    print(f"y2 = {y2:.6f} (en t2 = {t2:.6f} s)")
    print(f"y3 = {y3:.6f} (en t3 = {t3:.6f} s)")

    k = y_final_est / amplitud_escalon

    k1 = (y1 / (amplitud_escalon * k)) - 1
    k2 = (y2 / (amplitud_escalon * k)) - 1
    k3 = (y3 / (amplitud_escalon * k)) - 1

    print(f"k1 = {k1:.6f}")
    print(f"k2 = {k2:.6f}")
    print(f"k3 = {k3:.6f}")

    b = 4 * k1**3 * k3 - 3 * k1**2 * k2**2 - 4 * k2**3 + k3**2 + 6 * k1 * k2 * k3
    print(f"Discriminante b = {b:.6f}")

    if b < 0:
        print(f"Revisar los tiempos t1, t2, t3. Los polos pueden ser complejos")
    sqrt_b = np.sqrt(b)

    alfa1 = (k1 * k2 + k3 - sqrt_b) / (2 * (k1**2 + k2))
    alfa2 = (k1 * k2 + k3 + sqrt_b) / (2 * (k1**2 + k2))
    beta = (2 * k1**3 + 3 * k1 * k2 + k3 - sqrt_b) / sqrt_b
    print(f"alfa1 = {alfa1:.6f}")
    print(f"alfa2 = {alfa2:.6f}")
    print(f"beta  = {beta:.6f}")
    T1 = -t1 / np.log(abs(alfa1))
    T2 = -t1 / np.log(alfa2)
    T1, T2 = np.real(T1), np.real(T2)
    T3 = np.real(beta * (T1 - T2) + T1)

    return k, T1, T2, T3

# === Cargar datos medidos ===

file_path = "./data/Curvas_Medidas_Motor_2025.csv"
df = pd.read_csv(file_path)

df.columns = [col.strip() for col in df.columns]

tiempos = df["Tiempo [Seg.]"].values
salidas = df["Velocidad angular [rad /seg]"].values

# === Parámetros generales ===

retardo = 160          # Retardo tensión-velocidad [s]
retardo_torque = 161   # Retardo torque-velocidad [s]
step_amplitude = 2     # Escalón de tensión [V]
torque_amplitude = 0.12 # Escalón de torque [Nm]
velocidad_offset = 7.63   # Velocidad estacionaria previa al cambio de torque

tiempos_relativos = [0.040, 0.080, 0.120]  # Tiempos para tensión

# TIEMPOS OPTIMIZADOS (ver archivo de optimizacion de la carpeta)
# tiempos_relativos_torque = [40.3, 156.375, 199.166] # v1 , con nelder-mead simplex
tiempos_relativos_torque = [50.6211,108.3162, 163.9544] # optimizado con algoritmo evolutivo (differential evolution)

# === Tratamiento de señales para adaptar al metodo chen ===

# Para tensión: usamos directamente la salida medida
salidas_tension = salidas

# Para torque: ajuste de la salida para representar la contribución del torque
salidas_torque = -(salidas - velocidad_offset)

# Estimación de estado estacionario para tensión
mask_estacionario = (tiempos >= 3000) & (tiempos <= 3100)
y_final_est_tension = salidas[mask_estacionario].mean()

# Estimación del estado estacionario para torque
mask_estacionario = (tiempos >=1250) & (tiempos <= 1300)
y_final_est_torque = -(salidas[mask_estacionario].mean() - y_final_est_tension)
print(f" Y final estimado del torque = {y_final_est_torque}")
# === Identificación de parámetros ===

# Tensión -> Velocidad
k_va, T1_va, T2_va, T3_va = obtener_parametros_por_metodo_chen(
    tiempos_relativos, retardo, tiempos, salidas_tension, step_amplitude, y_final_est_tension
)

# Torque -> Velocidad
k_torque, T1_torque, T2_torque, T3_torque = obtener_parametros_por_metodo_chen(
    tiempos_relativos_torque, retardo_torque, tiempos, salidas_torque, torque_amplitude, y_final_est_torque
)

# === Mostrar resultados ===

print("\n--- PARÁMETROS IDENTIFICADOS ---")
print(f"Tensión -> Velocidad:")
print(f"  K_va = {k_va:.4f}")
print(f"  T1_va = {T1_va:.6f} s")
print(f"  T2_va = {T2_va:.6f} s")
print(f"  T3_va = {T3_va:.6f} s")

print(f"\nTorque -> Velocidad:")
print(f"  K_torque = {k_torque:.4f}")
print(f"  T1_torque = {T1_torque:.6f} s")
print(f"  T2_torque = {T2_torque:.6f} s")
print(f"  T3_torque = {T3_torque:.6f} s")

# === Construccion de las funciones de transferencia ===

# Velocidad por tensión
sistema_va = TransferFunction([k_va], np.convolve([T1_va, 1], [T2_va, 1]))

# Velocidad por torque
sistema_torque = TransferFunction(k_torque, np.convolve([T1_torque, 1], [T2_torque, 1]))

# === Simulación ===

# Configuración de tiempo de simulación
dt = 0.005
t_sim = np.arange(0, 3500, dt)

# Entradas
entrada_tension = np.zeros_like(t_sim)
entrada_torque = np.zeros_like(t_sim)

# Escalones aplicados
idx_retardo = int(retardo / dt)
idx_inicio_torque = int(retardo_torque / dt)  # Torque empieza en este indice 
idx_fin_torque = int(1361 / dt) # Finaliza en este indice, el num se obtiene observando


entrada_tension[idx_retardo:] = 2  # Escalón de tensión

entrada_torque[idx_inicio_torque:idx_fin_torque] = torque_amplitude  # Escalón de torque

# Simulación de respuestas
#y_tension, _, _ = lsim(sistema_va, U=entrada_tension, T=t_sim)
#y_torque, _, _ = lsim(sistema_torque, U=entrada_torque, T=t_sim)

t_out, y_tension, _ = lsim(sistema_va,    U=entrada_tension, T=t_sim)
t_out, y_torque,  _ = lsim(sistema_torque, U=entrada_torque,  T=t_sim)

y_total = y_tension - y_torque
# === Gráficas ===

plt.figure(figsize=(12, 8))
plt.plot(tiempos, salidas, label="Medición Real", linewidth=2)
plt.plot(t_sim, y_total, label="Modelo Identificado (Chen)", linewidth=2)
plt.title("Comparación Final: Modelo vs Medición")
plt.xlabel("Tiempo [s]")
plt.ylabel("Velocidad Angular [rad/s]")
plt.legend()
plt.grid(True)
#plt.xlim(159, 175)
plt.tight_layout()
plt.show()

