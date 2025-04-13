import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from qiskit import Aer
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import folium
from tqdm import tqdm

# Load and preprocess your data
print("Loading and preprocessing data...")
df = pd.read_csv("../data/tornado_severity_data.csv")
df = df.rename(columns={'ACC_STD_LAT_NBR': 'lat', 'ACC_STD_LON_NBR': 'lon', 'CAT Severity Code': 'severity'})
df = df.dropna(subset=['lat', 'lon', 'severity'])

df = df.sample(n=100, random_state=42)
print(f"Using {len(df)} data points for clustering")

# Normalize severity
scaler = MinMaxScaler()
df['handling_norm'] = scaler.fit_transform(df[['severity']])
X = df[['lat', 'lon', 'handling_norm']].to_numpy()

# Determine optimal number of clusters using silhouette score
print("Determining optimal k using silhouette method...")
inertias = []
silhouettes = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X, labels))
k = k_range[np.argmax(silhouettes)]
print(f"Optimal number of clusters: {k}")

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
    rhs=k,
    name='sum_constraint'
)

# Run QAOA
print("Running QAOA optimization...")
backend = Aer.get_backend('qasm_simulator')
qaoa = QAOA(optimizer=COBYLA(), reps=2, quantum_instance=backend)
optimizer = MinimumEigenOptimizer(qaoa)
result = optimizer.solve(qp)

# Extract selected centroids
selected = [i for i, val in enumerate(result.x) if val > 0.5]
if len(selected) > k:
    selected = selected[:k]
elif len(selected) < k:
    unselected = list(set(range(n)) - set(selected))
    selected.extend(unselected[:k - len(selected)])

# Final KMeans clustering
print("Performing final clustering...")
centroids = X[selected]
kmeans = KMeans(n_clusters=k, init=centroids, n_init=1, random_state=42)
df['zone'] = kmeans.fit_predict(X)

print("Creating visualization...")
m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=6)
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'black', 'gray']

for _, row in tqdm(df.iterrows(), total=len(df), desc="Plotting points"):
    folium.CircleMarker(
        location=(row['lat'], row['lon']),
        radius=4,
        color=colors[row['zone'] % len(colors)],
        fill=True,
        fill_opacity=0.6,
        popup=f"Severity: {int(row['severity'])} | Zone: {row['zone']}"
    ).add_to(m)

for idx in selected:
    folium.CircleMarker(
        location=(X[idx][0], X[idx][1]),
        radius=8,
        color='black',
        fill=True,
        fill_opacity=0.8,
        popup=f"Quantum-selected centroid (Severity: {int(df.iloc[idx]['severity'])})"
    ).add_to(m)

m.save("qmeans_map.html")

# Print summary
print("\nQuantum-selected centroids:")
for idx in selected:
    print(f"Centroid at ({X[idx][0]:.2f}, {X[idx][1]:.2f}) with severity {df.iloc[idx]['severity']}")

print("\nCluster sizes:")
print(df['zone'].value_counts())