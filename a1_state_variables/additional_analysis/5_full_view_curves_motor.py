import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos desde el CSV
df = pd.read_csv("data/Curvas_Medidas_Motor_2025.csv")

# Convertir columnas a los nombres correctos si es necesario
df.columns = [col.strip() for col in df.columns]  # eliminar espacios

# Crear gráficos
plt.figure(figsize=(12, 10))

# Gráfico 1: Velocidad angular vs Tiempo
plt.subplot(4, 1, 1)
plt.plot(df['Tiempo [Seg.]'], df['Velocidad angular [rad /seg]'], color='black')
plt.title('Velocidad angular')
plt.ylabel('Velocidad [rad/s]')
plt.grid(True)

# Gráfico 2: Corriente en armadura vs Tiempo
plt.subplot(4, 1, 2)
plt.plot(df['Tiempo [Seg.]'], df['Corriente en armadura [A]'], color='black')
plt.title('Corriente en armadura')
plt.ylabel('Corriente [A]')
plt.grid(True)

# Gráfico 3: Tensión vs Tiempo
plt.subplot(4, 1, 3)
plt.plot(df['Tiempo [Seg.]'], df['Tensión [V]'], color='black')
plt.title('Tensión')
plt.ylabel('Tensión [V]')
plt.grid(True)

# Gráfico 4: Torque vs Tiempo
plt.subplot(4, 1, 4)
plt.plot(df['Tiempo [Seg.]'], df['Torque'], color='black')
plt.title('Torque aplicado')
plt.xlabel('Tiempo [s]')
plt.ylabel('Torque')
plt.grid(True)

plt.tight_layout()
plt.show()

