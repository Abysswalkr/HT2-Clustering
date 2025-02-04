import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')

X = df[['sepal_length', 'sepal_width']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K, inertias, 'bx-')
plt.xlabel('k (número de clusters)')
plt.ylabel('Inercia')
plt.title('Método del Codo para determinar k óptimo')
plt.grid(True)

cambios = np.diff(inertias)
tasa_cambio = np.diff(cambios)
print("\nTasa de cambio de la inercia entre k consecutivos:")
for k, cambio in enumerate(cambios, start=1):
    print(f"De k={k} a k={k+1}: {cambio:.2f}")

print("\nTasa de cambio de segundo orden (cambio en la pendiente):")
for k, tasa in enumerate(tasa_cambio, start=1):
    print(f"En k={k+1}: {tasa:.2f}")

plt.annotate('Zona de alto\ncambio', xy=(2, inertias[1]), 
             xytext=(3, inertias[1]+50),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate('Zona de\nestabilización', xy=(4, inertias[3]), 
             xytext=(5, inertias[3]+50),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()

def calcular_r2(x, y):
    correlacion = np.corrcoef(x, y)[0,1]
    return correlacion**2