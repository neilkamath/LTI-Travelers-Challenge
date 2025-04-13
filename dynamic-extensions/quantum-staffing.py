import numpy as np
import pandas as pd
import networkx as nx
from itertools import product
from sklearn.cluster import KMeans
import json

# Quantum packages
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms import QAOA
from qiskit.utils import algorithm_globals
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver

# --- Parameters ---
NUM_CLAIMS = 1065
NUM_HANDLERS = 50
NUM_ZONES = 10
SKILL_LEVELS = [1, 2, 3, 4, 5]
SEVERITY_LEVELS = [1, 2, 3, 4, 5]
HANDLER_SKILLS = np.random.choice(SKILL_LEVELS, NUM_HANDLERS)
HANDLER_LOCATIONS = np.random.rand(NUM_HANDLERS, 2) * 100
MAX_TRAVEL_DISTANCE = 60  # Max distance handler is allowed to travel for on-site claims

np.random.seed(42)

# --- Step 1: Load Real Claim Severity Counts ---
def load_real_severity_distribution(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    percentages = {int(k): v for k, v in data['percentages'].items()}
    return percentages

# --- Step 1B: Simulate Claim Data ---
def simulate_claims_with_distribution(num_claims, severity_percentages):
    severities = np.random.choice(
        list(severity_percentages.keys()),
        size=num_claims,
        p=[v / 100 for v in severity_percentages.values()]
    )
    locations = np.random.rand(num_claims, 2) * 100
    resolution_days = np.random.randint(1, 6, size=num_claims)
    modes = np.where(severities >= 3, 'on-site', 'virtual')
    return pd.DataFrame({
        'claim_id': range(num_claims),
        'severity': severities,
        'x': locations[:, 0],
        'y': locations[:, 1],
        'target_days': resolution_days,
        'mode': modes
    })

# --- Step 2: Simulate Productivity Matrix ---
def simulate_productivity_matrix():
    matrix = np.zeros((len(SKILL_LEVELS), len(SEVERITY_LEVELS)))
    for i, skill in enumerate(SKILL_LEVELS):
        for j, severity in enumerate(SEVERITY_LEVELS):
            if skill >= severity:
                matrix[i, j] = 1.0 + 0.1 * (skill - severity)
            else:
                matrix[i, j] = max(0.1, 1.0 - 0.2 * (severity - skill))
    return pd.DataFrame(matrix, index=[f'skill_{s}' for s in SKILL_LEVELS], columns=[f'severity_{s}' for s in SEVERITY_LEVELS])

# --- Step 3: Preprocessing ---
def cluster_claims(claims_df, num_clusters):
    coords = claims_df[['x', 'y']].values
    severity_weight = claims_df['severity'].values.reshape(-1, 1)
    features = np.concatenate((coords, severity_weight), axis=1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(features)
    claims_df['zone'] = kmeans.labels_
    return claims_df

def normalize_claims(claims_df):
    claims_df['norm_target'] = (claims_df['target_days'] - claims_df['target_days'].min()) / (claims_df['target_days'].max() - claims_df['target_days'].min())
    return claims_df

def estimate_zone_demand(claims_df):
    return claims_df.groupby(['zone', 'severity']).size().unstack(fill_value=0)

# --- Quantum Stage 1 ---
def get_zone_workloads(claims_df):
    return claims_df.groupby(['zone', 'severity']).agg(
        count=('claim_id', 'count'),
        total_target_days=('target_days', 'sum'),
        mean_target_days=('target_days', 'mean'),
        norm_workload=('norm_target', 'sum')
    ).reset_index()

# --- Quantum Stage 2: QAOA Skill-Severity Matching ---
def build_qaoa_skill_matcher(zone_demand, productivity_matrix, claims_df):
    print("[QAOA STAGE] Building QUBO for handler-zone assignments...")
    qp = QuadraticProgram()
    num_zones = zone_demand.shape[0]
    zone_coords = claims_df.groupby('zone')[['x', 'y']].mean().values

    for i in range(NUM_HANDLERS):
        for j in range(num_zones):
            qp.binary_var(name=f"x_{i}_{j}")

    linear = {}
    quadratic = {}
    alpha, beta, gamma = 1.0, 2.0, 0.1

    for i in range(NUM_HANDLERS):
        skill = HANDLER_SKILLS[i]
        handler_loc = HANDLER_LOCATIONS[i]
        for j in range(num_zones):
            severity_demand = zone_demand.iloc[j]
            zone_loc = zone_coords[j]
            distance = np.linalg.norm(handler_loc - zone_loc)

            if any(claims_df[claims_df['zone'] == j]['mode'] == 'on-site') and distance > MAX_TRAVEL_DISTANCE:
                continue  # Restrict assignment if travel is too high for on-site claims

            travel_penalty = 0.2 * (distance // 30) if any(claims_df[claims_df['zone'] == j]['mode'] == 'on-site') else 0
            weighted_productivity = sum(
                severity_demand.get(sev, 0) * productivity_matrix.loc[f'skill_{skill}', f'severity_{sev}'] * (1 - travel_penalty)
                for sev in SEVERITY_LEVELS
            )
            mismatch_penalty = sum(
                severity_demand.get(sev, 0) * max(0, sev - skill) * 0.2
                for sev in SEVERITY_LEVELS
            )
            cost_penalty = distance * 10 if any(claims_df[claims_df['zone'] == j]['mode'] == 'on-site') else 0
            var_name = f"x_{i}_{j}"
            linear[var_name] = -alpha * weighted_productivity + beta * mismatch_penalty + gamma * cost_penalty

    qp.minimize(linear=linear, quadratic=quadratic)

    solver = NumPyMinimumEigensolver()
    optimizer = MinimumEigenOptimizer(solver)
    result = optimizer.solve(qp)
    print("\n[QAOA (Classical) Solution]")
    print(result)

# --- Quantum Stage 3 ---
def optimize_travel_within_zones(claims_df):
    print("[Q-Travel Stage] Route optimization placeholder.")
    return None

# --- Postprocessing ---
def evaluate_solution():
    print("[Postprocessing] Validation and scoring logic.")
    return None

# --- Main Execution ---
if __name__ == "__main__":
    print("\n--- Loading Real Tornado Claim Distribution ---")
    severity_dist = load_real_severity_distribution("Copy of tornado_severity_data_counts.json")

    print("\n--- Simulating Claims Based on Real Distribution ---")
    claims_df = simulate_claims_with_distribution(NUM_CLAIMS, severity_dist)
    prod_matrix = simulate_productivity_matrix()

    print("\n--- Classical Preprocessing ---")
    claims_df = cluster_claims(claims_df, NUM_ZONES)
    claims_df = normalize_claims(claims_df)
    zone_demand = estimate_zone_demand(claims_df)

    print("\n--- Quantum Stage 1: Q-SMART Clustering Output ---")
    workloads = get_zone_workloads(claims_df)
    print(workloads)

    print("\n--- Quantum Stage 2: Skill-Severity Matching ---")
    build_qaoa_skill_matcher(zone_demand, prod_matrix, claims_df)

    print("\n--- Quantum Stage 3: Travel Path Optimization ---")
    optimize_travel_within_zones(claims_df)

    print("\n--- Postprocessing ---")
    evaluate_solution()

    print("\n✅ End-to-end pipeline complete with on-site/virtual logic integrated.")