import pandas as pd
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms.minimum_eigensolvers import QAOA
from qiskit.algorithms.optimizers import SPSA
from qiskit.utils import algorithm_globals
from qiskit import Aer
from qiskit.primitives import Sampler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load and preprocess data
claims_df = pd.read_csv("../data/tornado_severity_data.csv")
claims_df_sample = claims_df.head(5).copy()
claim_severities = claims_df_sample["CAT Severity Code"].astype(int).tolist()
num_claims = len(claim_severities)

# Geo-clustering of claims into zones
coords = claims_df_sample[["ACC_STD_LAT_NBR", "ACC_STD_LON_NBR"]]
kmeans = KMeans(n_clusters=2, random_state=42).fit(coords)
claims_df_sample["Cluster"] = kmeans.labels_

# Assign 1 handler per cluster
num_handlers_per_cluster = 1
handler_skills = []
handler_clusters = []
for cluster in range(2):
    for _ in range(num_handlers_per_cluster):
        skill = np.random.randint(1, 6)
        handler_skills.append(skill)
        handler_clusters.append(cluster)

num_handlers = len(handler_skills)

# Build mismatch + geo penalty cost matrix
assignment_cost = np.zeros((num_claims, num_handlers))
for i in range(num_claims):
    claim_sev = claim_severities[i]
    claim_cluster = claims_df_sample["Cluster"].iloc[i]
    for j in range(num_handlers):
        skill = handler_skills[j]
        skill_penalty = 10 * (claim_sev - skill) if skill < claim_sev else (skill - claim_sev)
        geo_penalty = 5 if claim_cluster != handler_clusters[j] else 0
        assignment_cost[i, j] = skill_penalty + geo_penalty

# Build QUBO
qp = QuadraticProgram()
for i in range(num_claims):
    for j in range(num_handlers):
        qp.binary_var(name=f"x_{i}_{j}")

linear_terms = {f"x_{i}_{j}": assignment_cost[i, j] for i in range(num_claims) for j in range(num_handlers)}
qp.minimize(linear=linear_terms)

# Constraints
for i in range(num_claims):
    qp.linear_constraint(
        linear={f"x_{i}_{j}": 1 for j in range(num_handlers)},
        sense="==",
        rhs=1,
        name=f"claim_{i}_assignment"
    )

for j in range(num_handlers):
    qp.linear_constraint(
        linear={f"x_{i}_{j}": 1 for i in range(num_claims)},
        sense="<=",
        rhs=2,
        name=f"handler_{j}_capacity"
    )

# Connect to simulator
backend = Aer.get_backend('qasm_simulator')
sampler = Sampler()

# Run QAOA
algorithm_globals.random_seed = 42
qaoa = QAOA(optimizer=SPSA(maxiter=50), reps=1, sampler=sampler)
meo = MinimumEigenOptimizer(qaoa)
result = meo.solve(qp)

# Process results
assignments = result.x
assignment_summary = []
for i in range(num_claims):
    for j in range(num_handlers):
        idx = i * num_handlers + j
        if assignments[idx] > 0.5:
            assignment_summary.append({
                "Claim ID": i,
                "Severity": claim_severities[i],
                "Cluster": claims_df_sample.iloc[i]["Cluster"],
                "Assigned Handler": f"H{j} (Skill {handler_skills[j]}, Zone {handler_clusters[j]})",
                "Handler Skill": handler_skills[j],
                "Penalty": assignment_cost[i, j]
            })

assignment_df = pd.DataFrame(assignment_summary)
print("QAOA Assignment Results (Aer Simulator):\n")
print(assignment_df)

# Save to CSV
assignment_df.to_csv("qaoa_assignment_results.csv", index=False)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
for handler in sorted(assignment_df["Assigned Handler"].unique()):
    handler_df = assignment_df[assignment_df["Assigned Handler"] == handler]
    ax.bar(handler_df["Claim ID"], handler_df["Penalty"], label=handler)

ax.set_title("Penalty per Claim Assignment by Handler (Aer Simulator)")
ax.set_xlabel("Claim ID")
ax.set_ylabel("Penalty")
ax.legend()
plt.tight_layout()
plt.savefig("qaoa_assignment_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# Print plot confirmation
print("\nPlot saved as 'qaoa_assignment_plot.png'")