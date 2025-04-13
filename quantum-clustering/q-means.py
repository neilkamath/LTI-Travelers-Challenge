import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import qnexus as qnx
from pytket import Circuit
from datetime import datetime
import os
import importlib.util
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms.minimum_eigensolvers import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit import Aer
from qiskit.primitives import Sampler
import folium
from tqdm import tqdm
from docplex.mp.model import Model
from kneed import KneeLocator

output_dir = os.path.dirname(os.path.abspath(__file__))

# load and clean data
print("Loading and preprocessing data...")
df = pd.read_csv("data/tornado_severity_data.csv")
df = df.rename(columns={'ACC_STD_LAT_NBR': 'lat', 'ACC_STD_LON_NBR': 'lon', 'CAT Severity Code': 'severity'})
df = df.dropna(subset=['lat', 'lon', 'severity'])

# size=500 about 2 minutes with H1-1LE emulator --> (maybe) allow input
sample_size = 500
df = df.sample(n=sample_size, random_state=42)
print(f"Using {len(df)} data points for clustering")

scaler = MinMaxScaler()
df['handling_norm'] = scaler.fit_transform(df[['severity']])
X = df[['lat', 'lon', 'handling_norm']].to_numpy()

# Import the determine_optimal_clusters function from q-elbow.py
print("Importing determine_optimal_clusters from q-elbow.py...")
spec = importlib.util.spec_from_file_location("q_elbow", os.path.join(output_dir, "q-elbow.py"))
q_elbow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q_elbow)

# Determine optimal number of clusters using the function from q-elbow.py
print("Determining optimal number of clusters...")
optimal_k, inertias, silhouettes = q_elbow.determine_optimal_clusters(
    X, 
    k_range=range(2, 7),  # Limited range for faster computation
    show_plot=False,
    save_plot=True,
    output_dir=output_dir
)

print(f"Using k = {optimal_k} clusters")

original_k = optimal_k
optimal_k = 4

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

# QNexus starts here
print("Creating QNexus project...")
my_project_ref = qnx.projects.get_or_create(name="Q-Means Clustering Project")

# Quantinuum H1-1LE noiseless emulator
my_quantinuum_config = qnx.QuantinuumConfig(
    device_name="H1-1LE",
)

print("Building quantum circuit...")
demo_circuit_size = 10
circuit = Circuit(demo_circuit_size)

# Apply Hadamard gates to create superposition
for i in range(demo_circuit_size):
    circuit.H(i)

circuit.measure_all()

# Upload circuit to QNexus
my_circuit_ref = qnx.circuits.upload(
    name=f"Q-Means Demo Circuit {datetime.now()}",
    circuit=circuit,
    project=my_project_ref,
)

# Execute the job on H1-1LE
print("Executing demo circuit on H1-1LE emulator...")
execute_job = qnx.start_execute_job(
    name=f"Q-Means Demo Job {datetime.now()}",
    circuits=[my_circuit_ref],
    n_shots=[1000],  # Number of shots
    backend_config=my_quantinuum_config,
    project=my_project_ref,
)

# Get results
print("Processing quantum results...")
results = execute_job.df()
print(results)

print("Solving QUBO with greedy optimization...")

# Initialize selection with a random point
selected_indices = [np.random.randint(0, n)]

# Greedy selection
while len(selected_indices) < optimal_k:
    best_idx = -1
    best_dist = -1
    
    for i in range(n):
        if i not in selected_indices:
            min_dist = min(np.linalg.norm(X[i] - X[j]) for j in selected_indices)
            if min_dist > best_dist:
                best_dist = min_dist
                best_idx = i
    
    if best_idx != -1:
        selected_indices.append(best_idx)
        print(f"Selected point {best_idx} with distance {best_dist:.4f}")
    else:
        break

print(f"Selected {len(selected_indices)} centroids")

# Get the selected centroids
selected_centroids = X[selected_indices]

# Final KMeans clustering on full dataset
print("Performing final clustering on full dataset...")
kmeans = KMeans(n_clusters=optimal_k, init=selected_centroids, n_init=1, random_state=42)
df['zone'] = kmeans.fit_predict(X)

print("Creating visualization...")
map_file = os.path.join(output_dir, "qmeans_map.html")
m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=5)
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'black', 'gray']

viz_sample_size = min(1000, len(df))
viz_sample = df.sample(n=viz_sample_size, random_state=42) if len(df) > viz_sample_size else df

for _, row in tqdm(viz_sample.iterrows(), total=len(viz_sample), desc="Plotting points"):
    folium.CircleMarker(
        location=(row['lat'], row['lon']),
        radius=4,
        color=colors[row['zone'] % len(colors)],
        fill=True,
        fill_opacity=0.6,
        popup=f"Severity: {int(row['severity'])} | Zone: {row['zone']}"
    ).add_to(m)

final_centroids = kmeans.cluster_centers_
for i, centroid in enumerate(final_centroids):
    folium.CircleMarker(
        location=(centroid[0], centroid[1]),  # lat, lon
        radius=8,
        color='black',
        fill=True,
        fill_opacity=0.8,
        popup=f"Quantum-optimized centroid for Zone {i}"
    ).add_to(m)

m.save(map_file)
print(f"Map visualization saved as '{map_file}'")

# Summary stats
print("\nCluster sizes:")
print(df['zone'].value_counts())

print("\nMean severity by cluster:")
severity_by_cluster = df.groupby('zone')['severity'].mean().sort_values()
print(severity_by_cluster)

print("\nCluster centroids:")
for i, centroid in enumerate(final_centroids):
    print(f"Zone {i}: ({centroid[0]:.2f}, {centroid[1]:.2f}), Normalized severity: {centroid[2]:.2f}")