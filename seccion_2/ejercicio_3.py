import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')
X = df[['petal_length', 'petal_width']].values

plt.figure(figsize=(15, 12))

kmeans_original = KMeans(n_clusters=2, random_state=42)
clusters_original = kmeans_original.fit_predict(X)

plt.subplot(2, 2, 1)
for i in range(2):
    mask = clusters_original == i
    plt.scatter(X[mask, 0], X[mask, 1], 
                label=f'Cluster {i+1}', alpha=0.6)
plt.scatter(kmeans_original.cluster_centers_[:, 0], 
            kmeans_original.cluster_centers_[:, 1], 
            c='red', marker='x', s=200, linewidths=3, 
            label='Centroides')
plt.title('Clustering con Datos Originales')
plt.xlabel('Longitud del Pétalo')
plt.ylabel('Ancho del Pétalo')
plt.legend()
plt.grid(True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans_scaled = KMeans(n_clusters=2, random_state=42)
clusters_scaled = kmeans_scaled.fit_predict(X_scaled)

plt.subplot(2, 2, 2)
for i in range(2):
    mask = clusters_scaled == i
    plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1], 
                label=f'Cluster {i+1}', alpha=0.6)
plt.scatter(kmeans_scaled.cluster_centers_[:, 0], 
            kmeans_scaled.cluster_centers_[:, 1], 
            c='red', marker='x', s=200, linewidths=3, 
            label='Centroides')
plt.title('Clustering con Datos Estandarizados\n(Espacio Estandarizado)')
plt.xlabel('Longitud del Pétalo (estandarizada)')
plt.ylabel('Ancho del Pétalo (estandarizado)')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
for i in range(2):
    mask = clusters_scaled == i
    plt.scatter(X[mask, 0], X[mask, 1], 
                label=f'Cluster {i+1}', alpha=0.6)
centroids_original = scaler.inverse_transform(kmeans_scaled.cluster_centers_)
plt.scatter(centroids_original[:, 0], centroids_original[:, 1], 
            c='red', marker='x', s=200, linewidths=3, 
            label='Centroides')
plt.title('Clustering con Datos Estandarizados\n(Espacio Original)')
plt.xlabel('Longitud del Pétalo')
plt.ylabel('Ancho del Pétalo')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
diferencias = clusters_original != clusters_scaled
plt.scatter(X[~diferencias, 0], X[~diferencias, 1], 
           label='Igual Asignación', alpha=0.6, c='green')
plt.scatter(X[diferencias, 0], X[diferencias, 1], 
           label='Cambió Asignación', alpha=0.6, c='red')
plt.title('Comparación de Asignaciones\nVerde: Igual, Rojo: Cambió')
plt.xlabel('Longitud del Pétalo')
plt.ylabel('Ancho del Pétalo')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

n_diferentes = np.sum(diferencias)
porcentaje_diferentes = (n_diferentes / len(X)) * 100

print("\nAnálisis de diferencias entre clustering original y estandarizado:")
print(f"Número de puntos con diferente asignación: {n_diferentes}")
print(f"Porcentaje de puntos que cambiaron: {porcentaje_diferentes:.2f}%")

print("\nComparación de inercias:")
print(f"Inercia con datos originales: {kmeans_original.inertia_:.2f}")
print(f"Inercia con datos estandarizados: {kmeans_scaled.inertia_:.2f}")

print("\nEstadísticas de los clusters:")
print("\n1. Clustering con datos originales:")
for i in range(2):
    cluster_data = df[clusters_original == i]
    print(f"\nCluster {i+1}:")
    print("Número de muestras:", len(cluster_data))
    print(cluster_data[['petal_length', 'petal_width']].describe().round(2))

print("\n2. Clustering con datos estandarizados:")
for i in range(2):
    cluster_data = df[clusters_scaled == i]
    print(f"\nCluster {i+1}:")
    print("Número de muestras:", len(cluster_data))
    print(cluster_data[['petal_length', 'petal_width']].describe().round(2))