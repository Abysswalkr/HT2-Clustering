import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')
X = df[['petal_length', 'petal_width']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

cambios = np.diff(inertias)
tasa_cambio = np.diff(cambios)

plt.figure(figsize=(12, 6))

plt.plot(K, inertias, 'bx-', label='Inercia')
plt.xlabel('k (número de clusters)')
plt.ylabel('Inercia')
plt.title('Método del Codo para determinar k óptimo usando características del pétalo')
plt.grid(True)

plt.annotate('Primer codo', 
             xy=(2, inertias[1]), 
             xytext=(2.5, inertias[1]+50),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate('Segundo codo', 
             xy=(3, inertias[2]), 
             xytext=(3.5, inertias[2]+50),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()

print("Análisis de la inercia y sus cambios:")
print("\nInercia para cada k:")
for k, inercia in zip(K, inertias):
    print(f"k={k}: {inercia:.2f}")

print("\nTasa de cambio (primera derivada):")
for k, cambio in zip(range(2, 11), cambios):
    print(f"De k={k-1} a k={k}: {abs(cambio):.2f}")

print("\nTasa de cambio de segundo orden (segunda derivada):")
for k, tasa in zip(range(2, 10), tasa_cambio):
    print(f"En k={k}: {abs(tasa):.2f}")