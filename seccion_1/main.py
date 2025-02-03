import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Cargar los datos
df = pd.read_csv('iris.csv')

plt.figure(figsize=(15, 12))

# 1. Datos Originales sin estandarizar
X_original = df[['sepal_length', 'sepal_width']].values
kmeans_original = KMeans(n_clusters=2, random_state=42)
clusters_original = kmeans_original.fit_predict(X_original)

# Graficar datos originales
plt.subplot(2, 2, 1)
for i in range(2):
    mask = clusters_original == i
    plt.scatter(X_original[mask, 0], X_original[mask, 1], 
                label=f'Cluster {i+1}', alpha=0.5)
plt.scatter(kmeans_original.cluster_centers_[:, 0], 
            kmeans_original.cluster_centers_[:, 1], 
            c='red', marker='x', s=200, linewidths=3, 
            label='Centroides')
plt.title('Clustering con Datos Originales')
plt.xlabel('Longitud del Sépalo')
plt.ylabel('Ancho del Sépalo')
plt.legend()
plt.grid(True)

# 2. Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_original)
kmeans_scaled = KMeans(n_clusters=2, random_state=42)
clusters_scaled = kmeans_scaled.fit_predict(X_scaled)

# Graficar datos estandarizados
plt.subplot(2, 2, 2)
for i in range(2):
    mask = clusters_scaled == i
    plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1], 
                label=f'Cluster {i+1}', alpha=0.5)
plt.scatter(kmeans_scaled.cluster_centers_[:, 0], 
            kmeans_scaled.cluster_centers_[:, 1], 
            c='red', marker='x', s=200, linewidths=3, 
            label='Centroides')
plt.title('Clustering con Datos Estandarizados')
plt.xlabel('Longitud del Sépalo (estandarizada)')
plt.ylabel('Ancho del Sépalo (estandarizado)')
plt.legend()
plt.grid(True)

# 3. Comparar asignaciones de clusters
plt.subplot(2, 2, 3)
plt.scatter(X_original[:, 0], X_original[:, 1], 
           c=clusters_original, cmap='viridis', 
           alpha=0.5)
plt.title('Asignación de Clusters (Datos Originales)')
plt.xlabel('Longitud del Sépalo')
plt.ylabel('Ancho del Sépalo')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.scatter(X_original[:, 0], X_original[:, 1], 
           c=clusters_scaled, cmap='viridis', 
           alpha=0.5)
plt.title('Asignación de Clusters (Datos Estandarizados)')
plt.xlabel('Longitud del Sépalo')
plt.ylabel('Ancho del Sépalo')
plt.grid(True)

plt.tight_layout()
plt.show()

# Análisis de los clusters
print("\nComparación de los Clusters:")

print("\n1. Datos Originales:")
for i in range(2):
    cluster_data = df[clusters_original == i]
    print(f"\nCluster {i+1}:")
    print("Número de muestras:", len(cluster_data))
    print("Centro del cluster (sin estandarizar):", kmeans_original.cluster_centers_[i])
    print("Estadísticas descriptivas:")
    print(cluster_data[['sepal_length', 'sepal_width']].describe())

print("\n2. Datos Estandarizados:")
for i in range(2):
    cluster_data = df[clusters_scaled == i]
    print(f"\nCluster {i+1}:")
    print("Número de muestras:", len(cluster_data))
    print("Centro del cluster (estandarizado):", kmeans_scaled.cluster_centers_[i])
    centro_original = scaler.inverse_transform([kmeans_scaled.cluster_centers_[i]])
    print("Centro del cluster (des-estandarizado):", centro_original[0])
    print("Estadísticas descriptivas:")
    print(cluster_data[['sepal_length', 'sepal_width']].describe())

cambios = np.sum(clusters_original != clusters_scaled)
print(f"\nNúmero de puntos que cambiaron de cluster: {cambios}")
print(f"Porcentaje de cambio: {(cambios/len(df))*100:.2f}%")

# Calcular la inercia
print(f"\nInercia con datos originales: {kmeans_original.inertia_:.2f}")
print(f"Inercia con datos estandarizados: {kmeans_scaled.inertia_:.2f}")