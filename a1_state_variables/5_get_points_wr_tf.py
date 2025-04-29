import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos desde el CSV
df = pd.read_csv("data/Curvas_Medidas_Motor_2025.csv")

# Limpiar nombres de columnas
df.columns = [col.strip() for col in df.columns]

tiempo = df["Tiempo [Seg.]"].values
salida = df["Velocidad angular [rad /seg]"].values
# Definir la ventana de tiempo a visualizar
tiempo_min = 0 #159.6
tiempo_max = 3500#161

# Filtrar el DataFrame para incluir solo los datos dentro del rango deseado
df_filtrado = df[(df['Tiempo [Seg.]'] >= tiempo_min) & (df['Tiempo [Seg.]'] <= tiempo_max)]

# Tiempos a marcar
puntos_seleccionados = [160.030, 160.060, 160.090]
valores_puntos = [np.interp(t, tiempo, salida) for t in puntos_seleccionados]

# Crear gráficos
plt.figure(figsize=(12, 10))

# Gráfico 1: Velocidad angular vs Tiempo
plt.subplot(4, 1, 1)
plt.plot(df_filtrado['Tiempo [Seg.]'], df_filtrado['Velocidad angular [rad /seg]'], color='black')
# Marcar puntos en rojo
for t in puntos_seleccionados:
    idx = df_filtrado['Tiempo [Seg.]'].sub(t).abs().idxmin()
    valor = df_filtrado.loc[idx, 'Velocidad angular [rad /seg]']
    plt.plot(t, valor, 'ro')
    # Añadir texto apuntando al punto
    plt.annotate(f'{t:.3f}s\n{valor:.2f} rad/s', 
                 xy=(t, valor), 
                 xytext=(t+0.02, valor-1),  # Posición TEXTO
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=8)

plt.title(f'Velocidad angular ({tiempo_min}s - {tiempo_max}s)')
plt.ylabel('Velocidad [rad/s]')
plt.grid(True)

# Gráfico 2: Corriente en armadura vs Tiempo
plt.subplot(4, 1, 2)
plt.plot(df_filtrado['Tiempo [Seg.]'], df_filtrado['Corriente en armadura [A]'], color='black')
plt.title(f'Corriente en armadura ({tiempo_min}s - {tiempo_max}s)')
plt.ylabel('Corriente [A]')
plt.grid(True)

# Gráfico 3: Tensión vs Tiempo
plt.subplot(4, 1, 3)
plt.plot(df_filtrado['Tiempo [Seg.]'], df_filtrado['Tensión [V]'], color='black')
plt.title(f'Tensión ({tiempo_min}s - {tiempo_max}s)')
plt.ylabel('Tensión [V]')
plt.grid(True)

# Gráfico 4: Torque vs Tiempo
plt.subplot(4, 1, 4)
plt.plot(df_filtrado['Tiempo [Seg.]'], df_filtrado['Torque'], color='black')
plt.title(f'Torque ({tiempo_min}s - {tiempo_max}s)')
plt.xlabel('Tiempo [s]')
plt.ylabel('Torque')
plt.grid(True)

plt.tight_layout()
plt.show()

