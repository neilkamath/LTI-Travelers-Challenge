import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from qiskit import Aer
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from tqdm import tqdm

# load and clean data
df = pd.read_csv("Copy of tornado_severity_data.csv")
df = df.rename(columns={'ACC_STD_LAT_NBR': 'lat', 'ACC_STD_LON_NBR': 'lon', 'CAT Severity Code': 'severity'})
df = df.dropna(subset=['lat', 'lon', 'severity'])

# Normalize severity
scaler = MinMaxScaler()
df['handling_norm'] = scaler.fit_transform(df[['severity']])
X = df[['lat', 'lon', 'handling_norm']].to_numpy()

# Evaluate KMeans with Elbow and Silhouette methods
inertias = []
silhouettes = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X, labels))

# Plot Elbow and Silhouette graphs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, marker='o', color='orange')
plt.title('Elbow Method (Inertia)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouettes, marker='o', color='green')
plt.title('Silhouette Score')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

# Select optimal k from silhouette
optimal_k = k_range[np.argmax(silhouettes)]
print(f"Optimal number of clusters (k): {optimal_k}")

# Create QUBO for centroid selection
print("Creating QUBO problem...")
n = len(X)
qp = QuadraticProgram()
for i in range(n):
    qp.binary_var(name=f'x{i}')

penalty = 100
for i in tqdm(range(n), desc="Building QUBO"):
    qp.minimize(linear={f'x{i}': -penalty})
    for j in range(i + 1, n):
        dist = np.linalg.norm(X[i] - X[j])
        qp.minimize(quadratic={(f'x{i}', f'x{j}'): dist})

qp.linear_constraint(
    linear={f'x{i}': 1 for i in range(n)},
    sense='E',
    rhs=optimal_k,
    name='sum_constraint'
)

# Solve using QAOA
print("Running QAOA optimization...")
backend = Aer.get_backend('qasm_simulator')
qaoa = QAOA(optimizer=COBYLA(), reps=1, quantum_instance=backend)
optimizer = MinimumEigenOptimizer(qaoa)
result = optimizer.solve(qp)

# Extract QAOA-selected centroid indices
selected = [i for i, val in enumerate(result.x) if val > 0.5]
if len(selected) > optimal_k:
    selected = selected[:optimal_k]
elif len(selected) < optimal_k:
    unselected = list(set(range(n)) - set(selected))
    selected.extend(unselected[:optimal_k - len(selected)])

# Final clustering using QAOA-selected centroids
print("Performing KMeans with quantum-selected centroids...")
initial_centroids = X[selected]
kmeans = KMeans(n_clusters=optimal_k, init=initial_centroids, n_init=1, random_state=42)
df['zone'] = kmeans.fit_predict(X)

# Plot clusters and quantum centroids
plt.figure(figsize=(10, 7))
colors = ['gold', 'darkorange', 'crimson', 'violet', 'skyblue', 'limegreen', 'brown', 'grey', 'teal', 'navy']

for i in range(optimal_k):
    zone_data = df[df['zone'] == i]
    plt.scatter(zone_data['lon'], zone_data['lat'], s=30, label=f'Zone {i}', color=colors[i % len(colors)], alpha=0.7)

plt.scatter(X[selected][:, 1], X[selected][:, 0],
            color='black', s=120, edgecolor='white', marker='X', label='Quantum Centroid (QAOA)')

plt.title("Quantum Smart Zones (QAOA + KMeans)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()