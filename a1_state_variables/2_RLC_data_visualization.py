import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV
archivo_csv = 'data/Curvas_Medidas_RLC_2025.csv'
df = pd.read_csv(archivo_csv)

# Asegurar que los nombres de las columnas están bien
df.columns = [col.strip() for col in df.columns]

# Crear subplots: 4 filas, 1 columna
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# 1. Tensión de entrada
axs[0].plot(df['Tiempo [Seg.]'], df['Tensión de entrada [V]'], color='black')
axs[0].set_title('Tensión de entrada vs Tiempo')
axs[0].set_ylabel('Tensión [V]')
axs[0].grid(True)

# 2. Corriente
axs[1].plot(df['Tiempo [Seg.]'], df['Corriente [A]'], color='black')
axs[1].set_title('Corriente vs Tiempo')
axs[1].set_ylabel('Corriente [A]')
axs[1].grid(True)

# 3. Tensión en el capacitor
axs[2].plot(df['Tiempo [Seg.]'], df['Tensión en el capacitor [V]'], color='black')
axs[2].set_title('Tensión en el capacitor vs Tiempo')
axs[2].set_ylabel('Tensión [V]')
axs[2].grid(True)

# 4. Tensión de salida
axs[3].plot(df['Tiempo [Seg.]'], df['Tensión de salida [V]'], color='black')
axs[3].set_title('Tensión de salida vs Tiempo')
axs[3].set_ylabel('Tensión [V]')
axs[3].set_xlabel('Tiempo [Seg.]')
axs[3].grid(True)

# Ajustar espacio entre gráficos
plt.tight_layout()
plt.show()

