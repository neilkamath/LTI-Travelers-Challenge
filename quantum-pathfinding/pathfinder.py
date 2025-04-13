import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from qiskit import Aer
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.utils import algorithm_globals

# Set random seed for reproducibility
algorithm_globals.random_seed = 42

# Load your tornado severity dataset
df = pd.read_csv("Copy of tornado_severity_data.csv")

# Step 1: Assign handlers based on severity level (as a placeholder)
def assign_handler(severity):
    if severity <= 2:
        return 'H1'
    elif severity == 3:
        return 'H2'
    else:
        return 'H3'

df["CAT Severity Code"] = df["CAT Severity Code"].astype(int)
df['Handler'] = df["CAT Severity Code"].apply(assign_handler)

# Step 2: Group coordinates for each handler
handler_routes = {}
for handler, group in df.groupby('Handler'):
    # Limit to manageable size for quantum computation
    coords = list(zip(group["ACC_STD_LAT_NBR"], group["ACC_STD_LON_NBR"]))
    if len(coords) > 10:  # Limit for quantum processing
        coords = coords[:10]
    handler_routes[handler] = coords

# Step 3: Use quantum-inspired TSP optimization
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((np.array(p1) - np.array(p2))**2))

def create_distance_matrix(coords):
    n = len(coords)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = euclidean_distance(coords[i], coords[j])
    return distance_matrix

def solve_tsp_quantum(coords):
    if len(coords) <= 1:
        return 0, coords
    
    # Create distance matrix
    distance_matrix = create_distance_matrix(coords)
    n = len(coords)
    
    # For very small problems, use simple nearest neighbor
    if n <= 3:
        return nearest_neighbor_tsp(coords)
    
    # Use Qiskit's QAOA to solve a simplified path optimization
    # For larger problems, we'll use a quantum-classical hybrid approach
    
    # Create a quadratic program for a simplified path ordering problem
    qp = QuadraticProgram()
    
    # Add variables
    for i in range(n):
        for j in range(n):
            if i != j:
                qp.binary_var(f'x_{i}_{j}')
    
    # Minimize the total distance
    linear = {}
    quadratic = {}
    
    for i in range(n):
        for j in range(n):
            if i != j:
                linear[f'x_{i}_{j}'] = distance_matrix[i][j]
    
    qp.minimize(linear=linear, quadratic=quadratic)
    
    # Add constraints to ensure each city is visited exactly once
    # Each node has exactly one incoming edge
    for j in range(n):
        qp.linear_constraint(
            linear={f'x_{i}_{j}': 1 for i in range(n) if i != j},
            sense='==',
            rhs=1,
            name=f'incoming_{j}'
        )
    
    # Each node has exactly one outgoing edge
    for i in range(n):
        qp.linear_constraint(
            linear={f'x_{i}_{j}': 1 for j in range(n) if i != j},
            sense='==',
            rhs=1,
            name=f'outgoing_{i}'
        )
    
    # Solve using QAOA
    try:
        # Set up the quantum instance
        backend = Aer.get_backend('qasm_simulator')
        
        # Set up QAOA
        qaoa = QAOA(optimizer=COBYLA(), reps=1)
        quantum_optimizer = MinimumEigenOptimizer(qaoa)
        
        # Solve the problem
        print(f"Solving TSP for {n} points using QAOA...")
        result = quantum_optimizer.solve(qp)
        
        # Extract the tour from the result
        path = [0]  # Start at node 0
        for _ in range(1, n):
            for j in range(n):
                if j not in path:
                    var_name = f'x_{path[-1]}_{j}'
                    if var_name in result.x_dict and result.x_dict[var_name] > 0.5:
                        path.append(j)
                        break
        
        # Calculate the total distance
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += distance_matrix[path[i]][path[i+1]]
        
        # Convert path indices back to coordinates
        path_coords = [coords[i] for i in path]
        
        return total_distance, path_coords
    
    except Exception as e:
        print(f"Quantum computation error: {e}. Falling back to classical method.")
        return nearest_neighbor_tsp(coords)

def nearest_neighbor_tsp(coords):
    if not coords:
        return 0, []
    
    unvisited = coords[:]
    current = unvisited.pop(0)
    path = [current]
    path_length = 0
    
    while unvisited:
        next_point = min(unvisited, key=lambda p: euclidean_distance(current, p))
        path_length += euclidean_distance(current, next_point)
        current = next_point
        path.append(current)
        unvisited.remove(next_point)
    
    return path_length, path

# Step 4: Compute metrics for each handler
handler_metrics = []
handler_paths = {}

# Colors for visualization
colors = {'H1': 'blue', 'H2': 'green', 'H3': 'red'}

plt.figure(figsize=(12, 8))
for handler, coords in handler_routes.items():
    print(f"Processing handler {handler} with {len(coords)} points...")
    
    # Use quantum optimization if possible, otherwise fall back to classical
    try:
        dist, path = solve_tsp_quantum(coords)
        method = "Quantum-enhanced"
    except Exception as e:
        print(f"Error in quantum computation: {e}")
        dist, path = nearest_neighbor_tsp(coords)
        method = "Classical"
    
    travel_time_hours = dist / 0.5  # Assume 0.5 units/hour (like 50 km/h in scaled units)
    prod_loss = (travel_time_hours * 2) * 0.20  # Round trip, 20% per 30 mins
    cost_penalty = 200 * (len(coords) - 1)
    handler_metrics.append({
        "Handler": handler,
        "Num_Claims": len(coords),
        "Method": method,
        "Euclidean_Path_Length": round(dist, 2),
        "Estimated_Travel_Hours": round(travel_time_hours, 2),
        "Productivity_Loss_%": round(prod_loss * 100, 2),
        "Cost_Penalty_$": round(cost_penalty, 2)
    })
    
    # Store path for visualization
    handler_paths[handler] = path
    
    # Plot the path
    path_array = np.array(path)
    plt.plot(path_array[:, 1], path_array[:, 0], '-o', 
             color=colors[handler], label=f"{handler} ({len(coords)} claims, {method})", 
             alpha=0.7, markersize=6)

plt.title("Quantum-Enhanced Handler Route Optimization")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("quantum_handler_routes.png", dpi=300)
plt.show()

# Step 5: Create DataFrame and print results
handler_df = pd.DataFrame(handler_metrics)
print(handler_df)

# Show comparative metrics visualization
plt.figure(figsize=(12, 6))

# Bar chart of path lengths
plt.subplot(1, 2, 1)
handler_names = [f"{m['Handler']} ({m['Method']})" for m in handler_metrics]
path_lengths = [m['Euclidean_Path_Length'] for m in handler_metrics]
plt.bar(handler_names, path_lengths, color=[colors[h.split()[0]] for h in handler_names])
plt.title("Path Length by Handler")
plt.ylabel("Euclidean Distance")
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

# Bar chart of claims per handler
plt.subplot(1, 2, 2)
num_claims = [m['Num_Claims'] for m in handler_metrics]
plt.bar(handler_names, num_claims, color=[colors[h.split()[0]] for h in handler_names])
plt.title("Number of Claims by Handler")
plt.ylabel("Number of Claims")
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("quantum_handler_metrics.png", dpi=300)
plt.show()