import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('iris.csv')

plt.figure(figsize=(15, 6))

X = df[['sepal_length', 'sepal_width']].values
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

plt.subplot(1, 2, 2)

for i in range(2):
    cluster_data = df[df['cluster'] == i]
    plt.scatter(cluster_data['sepal_length'], 
                cluster_data['sepal_width'], 
                label=f'Cluster {i+1}', 
                alpha=0.5)

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], 
            c='red', marker='x', s=200, linewidths=3, 
            label='Centroides')

plt.title('K-means Clustering (k=2)')
plt.xlabel('Longitud del Sépalo')
plt.ylabel('Ancho del Sépalo')
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()

print("\nInformación de los Clusters:")
for i in range(2):
    cluster_data = df[df['cluster'] == i]
    print(f"\nCluster {i+1}:")
    print("Número de muestras:", len(cluster_data))
    print("Centro del cluster:", centroids[i])
    print("Estadísticas descriptivas:")
    print(cluster_data[['sepal_length', 'sepal_width']].describe())