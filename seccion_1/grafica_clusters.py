import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Cargar y preparar los datos
df = pd.read_csv('iris.csv')
X = df[['sepal_length', 'sepal_width']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

fig, axs = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle('Comparación de Diferentes Números de Clusters (K)', fontsize=16)

# Colores para los diferentes clusters
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']

# Función para crear gráfico de clusters
def plot_clusters(X_scaled, n_clusters, ax, title):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    X_original = scaler.inverse_transform(X_scaled)
    
    # Graficar cada cluster
    for i in range(n_clusters):
        mask = clusters == i
        ax.scatter(X_original[mask, 0], X_original[mask, 1], 
                  c=colors[i], label=f'Cluster {i+1}', alpha=0.6)
    
    centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(centroids_original[:, 0], centroids_original[:, 1], 
              c='red', marker='x', s=200, linewidths=3, 
              label='Centroides')
    
    ax.set_title(title)
    ax.set_xlabel('Longitud del Sépalo')
    ax.set_ylabel('Ancho del Sépalo')
    ax.legend()
    ax.grid(True)
    
    return kmeans.inertia_

# Graficar para diferentes valores de k
inertias = []
for i, k in enumerate([2, 3, 4]):
    row = i // 2
    col = i % 2
    inertia = plot_clusters(X_scaled, k, axs[row, col], f'K-means con {k} Clusters')
    inertias.append(inertia)

k_range = range(1, 11)
inertias_full = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias_full.append(kmeans.inertia_)

axs[1, 1].plot(k_range, inertias_full, 'bx-')
axs[1, 1].set_xlabel('k (número de clusters)')
axs[1, 1].set_ylabel('Inercia')
axs[1, 1].set_title('Método del Codo')
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

print("\nAnálisis de las diferentes configuraciones de clusters:")
for k, inertia in zip([2, 3, 4], inertias):
    print(f"\nPara k={k}:")
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Calcular estadísticas por cluster
    for i in range(k):
        cluster_data = df[clusters == i]
        print(f"\nCluster {i+1}:")
        print(f"Número de muestras: {len(cluster_data)}")
        print("Estadísticas del cluster:")
        print(cluster_data[['sepal_length', 'sepal_width']].describe().round(2))
    
    print(f"\nInercia total: {inertia:.2f}")