import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('iris.csv')

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.scatter(df['sepal_length'], df['sepal_width'], alpha=0.5)
plt.title('Datos Originales: Forma del Sépalo')
plt.xlabel('Longitud del Sépalo')
plt.ylabel('Ancho del Sépalo')
plt.grid(True)


# Ajustar el layout
plt.tight_layout()

# Mostrar el gráfico
plt.show()