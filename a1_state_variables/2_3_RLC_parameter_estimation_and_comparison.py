import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, lsim

# === 1. Cargar datos reales medidos ===
file_path = "./data/Curvas_Medidas_RLC_2025.csv"
df = pd.read_csv(file_path)

tiempo = df["Tiempo [Seg.]"]
entrada = df["Tensión de entrada [V]"]
salida = df["Tensión en el capacitor [V]"]
corriente = df["Corriente [A]"]
# === Parámetros para visualización ===
tiempo_minimo = 0.008
tiempo_maximo = 0.014
valor_maximo = 13
valor_minimo = -1
puntos_seleccionados = [0.01060, 0.01120, 0.01180]
valores_puntos = [np.interp(t, tiempo, salida) for t in puntos_seleccionados]

# === 2. Gráficas: Curva real y puntos seleccionados ===
fig, axs = plt.subplots(3, 1, figsize=(12, 10))

axs[0].plot(tiempo, salida, label="Tensión en el capacitor (Salida)", linewidth=2, color='black')
axs[0].scatter(puntos_seleccionados, valores_puntos, color='red', zorder=5, label="Puntos seleccionados")

for t, y in zip(puntos_seleccionados, valores_puntos):
    axs[0].annotate(f"({t:.5f}, {y:.2f} V)", xy=(t, y), xytext=(t + 0.0003, y + 0.5),
                    arrowprops=dict(arrowstyle="->", lw=1), fontsize=9)

axs[0].set_title("Respuesta al Escalón - Tensión en el Capacitor")
axs[0].set_xlabel("Tiempo [s]")
axs[0].set_ylabel("Tensión [V]")
axs[0].legend()
axs[0].grid(True)
axs[0].set_xlim(tiempo_minimo, tiempo_maximo)  # Ajuste del rango visible del tiempo
axs[0].set_ylim(valor_minimo, valor_maximo)

# === 3. Ajuste de tiempos por retardo de 10ms ===
retardo = 0.01
t1 = puntos_seleccionados[0] - retardo
t2 = puntos_seleccionados[1] - retardo
t3 = puntos_seleccionados[2] - retardo

y1, y2, y3 = valores_puntos

# Promedio de la salida entre 25 y 30 ms que consideramos el estado estacionario por inspección visual
y_final_estimado = salida[(tiempo >= 0.025) & (tiempo <= 0.03)].mean()
K = y_final_estimado

# === 4. Estimación de parámetros por el método de polos distintos ===
def estimar_parametros_polos_distintos(t1, y1, y2, y3, K):
    k1 = (y1 / K) - 1
    k2 = (y2 / K) - 1
    k3 = (y3 / K) - 1

    num_alpha12 = k3 + k1 * k2
    den_alpha12 = k1**2 + k2

    # Calculamos alfa1 + alfa2
    alpha1_plus_alpha2 = num_alpha12 / den_alpha12
    # Calculamos alfa1 * alfa2
    alpha1_times_alpha2 = (k2**2 - k1 * k3) / den_alpha12


# En este punto lo que hacemos es determinar si los polos son reales o complejos
# Visualmente, observando la dinamica del sistema podemos ver que los polos son reales, pero los calculamos como forma de validación
# Si el discriminante es negativo es porque los polos son complejos conjugados  (pues la raíz cuadrada de un número negativo no es real)
     
    discriminante = alpha1_plus_alpha2**2 - 4 * alpha1_times_alpha2
    if discriminante < 0:
        raise ValueError("Discriminante negativo, no se puede resolver.")

    sqrt_b = np.sqrt(discriminante)
    alpha1 = (alpha1_plus_alpha2 - sqrt_b) / 2
    alpha2 = (alpha1_plus_alpha2 + sqrt_b) / 2
    beta = (k1 + alpha2) / (alpha1 - alpha2)

    T1 = -t1 / np.log(alpha1)
    T2 = -t1 / np.log(alpha2)
    T3 = beta * (T1 - T2) + T1

    return K, T1, T2, T3

K_est, T1_est, T2_est, T3_est = estimar_parametros_polos_distintos(t1, y1, y2, y3, K)

# Mostrar resultados por consola
print("\n--- RESULTADOS DEL MODELO IDENTIFICADO ---")
print(f"Ganancia estacionaria estimada K: {K_est:.4f}")
print(f"Constante de tiempo T1: {T1_est:.6f} s")
print(f"Constante de tiempo T2: {T2_est:.6f} s")
print(f"Constante auxiliar T3: {T3_est:.6f} s")


# === Estimación de C a partir del segundo punto seleccionado y la siguiente muestra ===

# Obtener índice del segundo punto seleccionado (t2)
idx_central = np.argmin(np.abs(tiempo - puntos_seleccionados[1]))

# Asegurar que haya una muestra siguiente
if idx_central < len(tiempo) - 1:
    t_central = tiempo.iloc[idx_central]
    t_siguiente = tiempo.iloc[idx_central + 1]

    vc_central = salida.iloc[idx_central]
    vc_siguiente = salida.iloc[idx_central + 1]

    vin_central = entrada.iloc[idx_central]

    # Derivada por diferencia hacia adelante
    dvc_dt = (vc_siguiente - vc_central) / (t_siguiente - t_central)

    i_aprox = corriente.iloc[idx_central+1]

    # Capacitancia estimada
    C_estimado = i_aprox / dvc_dt

    print("\n--- ESTIMACIÓN DE C (desde t2 y t2+1) ---")
    print(f"Instante central: {t_central:.6f} s")
    print(f"Derivada de Vc: {dvc_dt:.2f} V/s")
    print(f"Corriente estimada: {i_aprox:.4f} A")
    print(f"Capacitancia estimada: {C_estimado*1e6:.2f} µF")

    C_est = C_estimado
else:
    print("\n No se pudo estimar C")


# === Estimación de R, L y C para sistema RLC serie ===
# Asumimos: FT = K * (T3 s + 1) / (T1*T2 s^2 + (T1+T2)s + 1)
# FT RLC serie normalizada: 1 / (L*C s^2 + R*C s + 1)

# Identificamos: T1*T2 = L*C, (T1+T2) = R*C
#C_est = 220e-6  # 220 uF
L_est = T1_est * T2_est / (C_est*K_est)
R_est = (T1_est + T2_est) / (C_est*K_est)

print("\n--- ESTIMACIÓN DE PARÁMETROS FÍSICOS ---")
print(f"C (asumido): {C_est*1e6:.0f} µF")
print(f"R estimado: {R_est:.2f} ohm")
print(f"L estimado: {L_est*1e6:.2f} µH")

# === 5. Crear función de transferencia ===
num = [1]
den = [T1_est * T2_est, T1_est + T2_est, 1]
sistema_estimado = TransferFunction(num, den)

den_str = f"{den[0]:.9f}·s² + {den[1]:.6f}·s + {den[2]:.6f}"

print("\n--- MODELO ESTIMADO (FORMA SIMBÓLICA) ---")
print(f"                1")
print(f"G(s) = ---------------------")
print(f"          {den_str}")


# === 6. Entrada alternante ===
t_total = 0.2
vin = 12
dt = 0.0001
t_sim = np.arange(0, t_total, dt)
u_sim = np.zeros_like(t_sim)

for i, t in enumerate(t_sim):
    if t < 0.01:
        u_sim[i] = 0
    elif t < 0.05:
        u_sim[i] = vin
    else:
        period_index = int(t // 0.05)
        u_sim[i] = vin if period_index % 2 == 0 else -vin

# === 7. Simular sistema ===
t_out, y_out, _ = lsim(sistema_estimado, U=u_sim, T=t_sim)

# === 8. Comparación con datos reales ===
mask_real = (tiempo >= 0) & (tiempo <= t_total)
tiempo_real = tiempo[mask_real]
salida_real = salida[mask_real]
entrada_real = entrada[mask_real]

# axs[1].plot(t_out, u_sim, label="Entrada simulada (alternante)", linestyle='--')
# axs[1].plot(tiempo_real, entrada_real, label="Entrada real medida", linestyle=':')
axs[1].plot(tiempo_real, salida_real, label="Respuesta real medida", linewidth=2)
axs[1].plot(t_out, y_out, label="Respuesta simulada (modelo identificado)", linewidth=2)
axs[1].set_title("Comparación: Sistema estimado vs Medición real")
axs[1].set_xlabel("Tiempo [s]")
axs[1].set_ylabel("Tensión [V]")
axs[1].set_xlim(tiempo_minimo, tiempo_maximo)  
axs[1].set_ylim(valor_minimo, valor_maximo)
axs[1].legend()
axs[1].grid(True)


axs[2].plot(tiempo_real, salida_real, label="Respuesta real (completa)", linewidth=2)
axs[2].plot(t_out, y_out, label="Respuesta simulada (completa)", linewidth=2)
axs[2].set_title("Comparación completa")
axs[2].set_xlabel("Tiempo [s]")
axs[2].set_ylabel("Tensión [V]")
axs[2].legend()
axs[2].grid(True)


plt.tight_layout()
plt.show()

