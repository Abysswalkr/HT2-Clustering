import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')

X = df[['petal_length', 'petal_width']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.scatter(df['petal_length'], df['petal_width'], alpha=0.5)
plt.title('Datos Originales: Forma del Pétalo')
plt.xlabel('Longitud del Pétalo')
plt.ylabel('Ancho del Pétalo')
plt.grid(True)

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

plt.subplot(1, 2, 2)

colors = ['#FF9999', '#66B2FF']
for i in range(2):
    mask = clusters == i
    plt.scatter(X[mask, 0], X[mask, 1], 
                c=colors[i],
                label=f'Cluster {i+1}', 
                alpha=0.6)

centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids_original[:, 0], centroids_original[:, 1], 
            c='red', marker='x', s=200, linewidths=3, 
            label='Centroides')

plt.title('K-means Clustering (k=2)')
plt.xlabel('Longitud del Pétalo')
plt.ylabel('Ancho del Pétalo')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nInformación de los Clusters:")
for i in range(2):
    cluster_data = df[clusters == i]
    print(f"\nCluster {i+1}:")
    print("Número de muestras:", len(cluster_data))
    print("\nEstadísticas del cluster:")
    print(cluster_data[['petal_length', 'petal_width']].describe().round(2))
    
print(f"\nInercia del modelo: {kmeans.inertia_:.2f}")

print("\nCoordenadas de los centroides (después de des-estandarizar):")
for i, centroid in enumerate(centroids_original):
    print(f"Centroide {i+1}: Longitud = {centroid[0]:.2f}, Ancho = {centroid[1]:.2f}")