import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('iris.csv')

plt.figure(figsize=(10, 8))

plt.scatter(df['petal_length'], df['petal_width'], alpha=0.6)

plt.title('Distribución de Iris según la forma del pétalo', fontsize=12)
plt.xlabel('Longitud del Pétalo')
plt.ylabel('Ancho del Pétalo')
plt.grid(True)

plt.annotate('Posible Grupo 1', 
             xy=(1.5, 0.3), 
             xytext=(2, 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate('Posible Grupo 2', 
             xy=(4, 1.3), 
             xytext=(4.5, 1.0),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate('Posible Grupo 3', 
             xy=(5, 1.8), 
             xytext=(5.5, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()