import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, silhouette_score

df_real = pd.read_csv('iris-con-respuestas.csv')
X = df_real[['sepal_length', 'sepal_width']].values
species = df_real['species'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
for i, specie in enumerate(np.unique(species)):
    mask = species == specie
    plt.scatter(X[mask, 0], X[mask, 1], label=specie, alpha=0.6)
plt.title('Especies Reales')
plt.xlabel('Longitud del Sépalo')
plt.ylabel('Ancho del Sépalo')
plt.legend()
plt.grid(True)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

plt.subplot(2, 2, 2)
for i in range(3):
    mask = clusters == i
    plt.scatter(X[mask, 0], X[mask, 1], label=f'Cluster {i+1}', alpha=0.6)
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids_original[:, 0], centroids_original[:, 1], 
           c='red', marker='x', s=200, linewidths=3, label='Centroides')
plt.title('Resultados del Clustering (k=3)')
plt.xlabel('Longitud del Sépalo')
plt.ylabel('Ancho del Sépalo')
plt.legend()
plt.grid(True)

correspondence = pd.crosstab(clusters, species)

plt.subplot(2, 2, 3)
plt.imshow(correspondence, cmap='YlOrRd')
plt.colorbar()
plt.xticks(range(len(np.unique(species))), np.unique(species), rotation=45)
plt.yticks(range(3), [f'Cluster {i+1}' for i in range(3)])
plt.title('Matriz de Correspondencia\nClusters vs Especies')

ari = adjusted_rand_score(species, clusters) 
silhouette = silhouette_score(X_scaled, clusters)

plt.subplot(2, 2, 4)
plt.text(0.5, 0.8, f'Métricas de Evaluación:', 
         horizontalalignment='center', fontsize=12)
plt.text(0.5, 0.6, f'Índice Rand Ajustado: {ari:.3f}', 
         horizontalalignment='center')
plt.text(0.5, 0.4, f'Coeficiente de Silueta: {silhouette:.3f}', 
         horizontalalignment='center')
plt.axis('off')

plt.tight_layout()
plt.show()
