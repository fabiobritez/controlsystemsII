import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lsim, TransferFunction
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# === Cargar datos ===
file_path = "./data/Curvas_Medidas_Motor_2025.csv"
df = pd.read_csv(file_path)
df.columns = [col.strip() for col in df.columns]

# === Datos ===
tiempos = df["Tiempo [Seg.]"].values
salidas = df["Velocidad angular [rad /seg]"].values

# === Parámetros ===
retardo_torque = 161  # [s]
torque_amplitude = 0.12
step_amplitude = 2
velocidad_offset = 7.63

# Tratamiento de señales
salidas_torque = -(salidas - velocidad_offset)

# Estimación estado estacionario
mask_estacionario = (tiempos >=1250) & (tiempos <= 1300)
y_final_est_torque = -(salidas[mask_estacionario].mean() - salidas[(tiempos >= 3000) & (tiempos <= 3100)].mean())

# === Funciones auxiliares ===
def encontrar_y(t_relativo, retardo, tiempos, salidas):
    idx = np.argmin(np.abs((t_relativo + retardo) - tiempos))
    return salidas[idx], tiempos[idx] - retardo

def obtener_parametros_por_metodo_chen(tiempos_relativos, retardo, tiempos, salidas_preparadas, amplitud_escalon, y_final_est):
    valores, tiempos_ajustados = [], []
    for t in tiempos_relativos:
        y_val, t_ajustado = encontrar_y(t, retardo, tiempos, salidas_preparadas)
        valores.append(y_val)
        tiempos_ajustados.append(t_ajustado)

    y1, y2, y3 = valores
    t1, t2, t3 = tiempos_ajustados

    k = y_final_est / amplitud_escalon

    k1 = (y1 / (amplitud_escalon * k)) - 1
    k2 = (y2 / (amplitud_escalon * k)) - 1
    k3 = (y3 / (amplitud_escalon * k)) - 1

    b = 4 * k1**3 * k3 - 3 * k1**2 * k2**2 - 4 * k2**3 + k3**2 + 6 * k1 * k2 * k3
    if b < 0:
        return np.nan, np.nan, np.nan, np.nan

    sqrt_b = np.sqrt(b)

    alfa1 = (k1 * k2 + k3 - sqrt_b) / (2 * (k1**2 + k2))
    alfa2 = (k1 * k2 + k3 + sqrt_b) / (2 * (k1**2 + k2))
    beta = (2 * k1**3 + 3 * k1 * k2 + k3 - sqrt_b) / sqrt_b

    T1 = -t1 / np.log(abs(alfa1))
    T2 = -t1 / np.log(alfa2)
    T1, T2 = np.real(T1), np.real(T2)
    T3 = np.real(beta * (T1 - T2) + T1)

    return k, T1, T2, T3

def error_de_modelo(tiempos_relativos):
    if np.any(np.diff(tiempos_relativos) <= 0):
        return 1e6  # Penalizar tiempos mal ordenados

    k, T1, T2, T3 = obtener_parametros_por_metodo_chen(
        tiempos_relativos, retardo_torque, tiempos, salidas_torque, torque_amplitude, y_final_est_torque)
    
    if np.isnan(k) or np.isnan(T1) or np.isnan(T2):
        return 1e6  # Penalizar si Chen falla

    sistema_torque = TransferFunction(k, np.convolve([T1, 1], [T2, 1]))

    dt = 0.005
    t_sim = np.arange(0, 3500, dt)
    entrada = np.zeros_like(t_sim)
    idx_retardo = int(retardo_torque / dt)
    entrada[idx_retardo:] = torque_amplitude

    _, y_simulado, _ = lsim(sistema_torque, U=entrada, T=t_sim)

    # Interpolar datos reales sobre el mismo tiempo simulado
    from scipy.interpolate import interp1d
    interp_real = interp1d(tiempos, salidas_torque, fill_value="extrapolate")
    salidas_reales_interp = interp_real(t_sim)

    mse = np.mean((y_simulado - salidas_reales_interp)**2)
    return mse

# === Optimizador ===

# Valores iniciales aproximados (basados en análisis anterior)
x0 = [39, 139, 239]

res = minimize(error_de_modelo, x0, method='Nelder-Mead', options={'maxiter': 200})

# === Mostrar resultados ===
print("\n--- Resultados de la optimización ---")
print(f"Tiempos optimizados: {res.x}")
print(f"Error final (MSE): {res.fun}")

# Puedes ahora usar esos tiempos óptimos en tu método Chen normal.

