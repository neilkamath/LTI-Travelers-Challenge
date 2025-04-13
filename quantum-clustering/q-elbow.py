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
from kneed import KneeLocator
import os

def determine_optimal_clusters(X, k_range=range(2, 11), show_plot=True, save_plot=False, output_dir=None):
    """
    Determine the optimal number of clusters using both elbow and silhouette methods.
    
    Args:
        X: Input data array with shape (n_samples, n_features)
        k_range: Range of k values to evaluate
        show_plot: Whether to display the plot
        save_plot: Whether to save the plot
        output_dir: Directory to save the plot if save_plot is True
        
    Returns:
        tuple: (optimal_k, inertias, silhouettes)
    """
    inertias = []
    silhouettes = []

    print("Evaluating optimal number of clusters...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    
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
    
    if save_plot and output_dir:
        plot_path = os.path.join(output_dir, "elbow_method_plot.png")
        plt.savefig(plot_path)
        print(f"Saved elbow method plot to {plot_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Select optimal k
    kneedle = KneeLocator(list(k_range), inertias, S=1.0, curve='convex', direction='decreasing')
    k_elbow = kneedle.elbow
    
    # If KneeLocator --> use silhouette score
    k_silhouette = k_range[np.argmax(silhouettes)]
    
    # Final k
    optimal_k = k_elbow if k_elbow else k_silhouette
    
    print(f"Elbow method suggests k = {k_elbow}")
    print(f"Silhouette method suggests k = {k_silhouette}")
    print(f"Selected optimal k = {optimal_k}")
    
    return optimal_k, inertias, silhouettes

if __name__ == "__main__":
    df = pd.read_csv(".../data/tornado_severity_data.csv")
    df = df.rename(columns={'ACC_STD_LAT_NBR': 'lat', 'ACC_STD_LON_NBR': 'lon', 'CAT Severity Code': 'severity'})
    df = df.dropna(subset=['lat', 'lon', 'severity'])

    scaler = MinMaxScaler()
    df['handling_norm'] = scaler.fit_transform(df[['severity']])
    X = df[['lat', 'lon', 'handling_norm']].to_numpy()

    optimal_k, _, _ = determine_optimal_clusters(X, show_plot=True)

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

    print("Running QAOA optimization...")
    backend = Aer.get_backend('qasm_simulator')
    qaoa = QAOA(optimizer=COBYLA(), reps=1, quantum_instance=backend)
    optimizer = MinimumEigenOptimizer(qaoa)
    result = optimizer.solve(qp)

    selected = [i for i, val in enumerate(result.x) if val > 0.5]
    if len(selected) > optimal_k:
        selected = selected[:optimal_k]
    elif len(selected) < optimal_k:
        unselected = list(set(range(n)) - set(selected))
        selected.extend(unselected[:optimal_k - len(selected)])

    print("Performing KMeans with quantum-selected centroids...")
    initial_centroids = X[selected]
    kmeans = KMeans(n_clusters=optimal_k, init=initial_centroids, n_init=1, random_state=42)
    df['zone'] = kmeans.fit_predict(X)

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