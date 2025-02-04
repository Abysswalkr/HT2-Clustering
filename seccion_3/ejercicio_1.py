import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from kneed import KneeLocator

# Cargar los datos
iris_data = pd.read_csv("../iris.csv")
iris_labels = pd.read_csv("../iris-con-respuestas.csv")

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris_data)

# Evaluar inercia para distintos valores de k (1 a 10)
inertia_values = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

# Determinar el punto de codo usando la librería kneed
knee_locator = KneeLocator(k_values, inertia_values, curve="convex", direction="decreasing")
optimal_k_kneed = knee_locator.elbow

# Visualización del método del codo
plt.figure(figsize=(8,5))
plt.plot(k_values, inertia_values, marker='o', linestyle='--', label="Inertia")
plt.axvline(optimal_k_kneed, color='red', linestyle='--', label=f"Optimal k = {optimal_k_kneed} (Kneed)")
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Inercia")
plt.title("Método del Codo con Kneed")
plt.legend()
plt.grid(True)
plt.show()

# Aplicar KMeans con k=2 y k=3 para comparar
kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_scaled)
kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)

# Convertir etiquetas reales a valores numéricos
species_mapping = {species: idx for idx, species in enumerate(iris_labels["species"].unique())}
y_true = iris_labels["species"].map(species_mapping).values

# Calcular el Índice Rand Ajustado para evaluar calidad de clustering
ari_k2 = adjusted_rand_score(y_true, kmeans_2.labels_)
ari_k3 = adjusted_rand_score(y_true, kmeans_3.labels_)

print(f"ARI para k=2: {ari_k2:.3f}")
print(f"ARI para k=3: {ari_k3:.3f}")

# Visualización de los clusters con k=3
plt.figure(figsize=(8,5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_3.labels_, cmap='viridis', alpha=0.6)
plt.title("Clustering con k=3")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.grid(True)
plt.show()
