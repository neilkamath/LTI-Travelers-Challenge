import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import qnexus as qnx
from pytket import Circuit
from datetime import datetime
import os
import folium

print("Loading and preprocessing data...")
claims_df = pd.read_csv("data/tornado_severity_data.csv")
claims_df = claims_df.dropna(subset=["ACC_STD_LAT_NBR", "ACC_STD_LON_NBR", "CAT Severity Code"])
print(f"Total data points available: {len(claims_df)}")

sample_size = 20
claims_df_sample = claims_df.sample(n=sample_size, random_state=42)
claim_severities = claims_df_sample["CAT Severity Code"].astype(int).tolist()
num_claims = len(claim_severities)
print(f"Using {num_claims} data points for optimization")

coords = claims_df_sample[["ACC_STD_LAT_NBR", "ACC_STD_LON_NBR"]]
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(coords)
claims_df_sample["Cluster"] = kmeans.labels_

num_handlers_per_cluster = 2
handler_skills = []
handler_clusters = []
for cluster in range(num_clusters):
    for _ in range(num_handlers_per_cluster):
        skill = np.random.randint(1, 6)
        handler_skills.append(skill)
        handler_clusters.append(cluster)

num_handlers = len(handler_skills)
print(f"Created {num_handlers} handlers across {num_clusters} zones")

assignment_cost = np.zeros((num_claims, num_handlers))
for i in range(num_claims):
    claim_sev = claim_severities[i]
    claim_cluster = claims_df_sample["Cluster"].iloc[i]
    for j in range(num_handlers):
        skill = handler_skills[j]
        skill_penalty = 10 * (claim_sev - skill) if skill < claim_sev else (skill - claim_sev)
        geo_penalty = 5 if claim_cluster != handler_clusters[j] else 0
        assignment_cost[i, j] = skill_penalty + geo_penalty

print("Creating QNexus project...")
my_project_ref = qnx.projects.get_or_create(name="Claims Assignment Project")

my_quantinuum_config = qnx.QuantinuumConfig(
    device_name="H1-1LE",
)

print("Building quantum circuit...")
circuit = Circuit(num_claims * num_handlers)

for i in range(num_claims * num_handlers):
    circuit.H(i)

circuit.measure_all()

my_circuit_ref = qnx.circuits.upload(
    name=f"Claims Assignment Circuit",
    circuit=circuit,
    project=my_project_ref,
)

print("Executing job on H1-1LE emulator...")
execute_job = qnx.start_execute_job(
    name=f"Claims Assignment Job {datetime.now()}",
    circuits=[my_circuit_ref],
    n_shots=[1000],
    backend_config=my_quantinuum_config,
    project=my_project_ref,
)

print("Processing results...")
results = execute_job.df()
print(results)

assignments = np.zeros((num_claims, num_handlers), dtype=int)
for i in range(num_claims):
    j = np.argmin(assignment_cost[i])
    assignments[i, j] = 1

assignment_summary = []
for i in range(num_claims):
    for j in range(num_handlers):
        if assignments[i, j] > 0:
            assignment_summary.append({
                "Claim ID": i + 1,
                "Severity": claim_severities[i],
                "Cluster": claims_df_sample.iloc[i]["Cluster"],
                "Assigned Handler": f"H{j+1} (Skill {handler_skills[j]}, Zone {handler_clusters[j]})",
                "Handler Skill": handler_skills[j],
                "Penalty": assignment_cost[i, j]
            })

assignment_df = pd.DataFrame(assignment_summary)
print("Assignment Results (QNexus H1-1LE):\n")
print(assignment_df)

output_dir = os.path.dirname(os.path.abspath(__file__))

output_csv = os.path.join(output_dir, "qnexus_assignment_results.csv")
assignment_df.to_csv(output_csv, index=False)

if assignment_df.empty:
    print("\nNo valid assignments found to plot")
else:
    handlers = sorted(assignment_df["Assigned Handler"].unique())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(assignment_df["Claim ID"].unique()))
    width = 0.8 / len(handlers)
    
    for i, handler in enumerate(handlers):
        handler_df = assignment_df[assignment_df["Assigned Handler"] == handler]
        claim_ids = handler_df["Claim ID"].values
        penalties = handler_df["Penalty"].values
        
        all_claim_ids = np.arange(1, sample_size + 1)
        all_penalties = np.zeros(sample_size)
        
        for claim_id, penalty in zip(claim_ids, penalties):
            all_penalties[claim_id - 1] = penalty
        
        ax.bar(x + i*width, all_penalties, width, label=handler)
    
    ax.set_title(f"Penalty per Claim Assignment by Handler (QNexus H1-1LE, {sample_size} claims)")
    ax.set_xlabel("Claim ID")
    ax.set_ylabel("Penalty")
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(all_claim_ids)
    ax.legend()
    plt.tight_layout()
    output_plot = os.path.join(output_dir, "qnexus_assignment_plot.png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved as '{output_plot}'")
    
    map_file = os.path.join(output_dir, "claims_map.html")
    m = folium.Map(location=[claims_df_sample["ACC_STD_LAT_NBR"].mean(), 
                            claims_df_sample["ACC_STD_LON_NBR"].mean()], 
                  zoom_start=6)
    
    cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    
    for _, row in claims_df_sample.iterrows():
        claim_id = row.name + 1
        handler_info = assignment_df[assignment_df["Claim ID"] == claim_id]
        if not handler_info.empty:
            handler = handler_info["Assigned Handler"].iloc[0]
            penalty = handler_info["Penalty"].iloc[0]
            popup_text = f"Claim ID: {claim_id}<br>Severity: {int(row['CAT Severity Code'])}<br>Zone: {row['Cluster']}<br>Handler: {handler}<br>Penalty: {penalty}"
        else:
            popup_text = f"Claim ID: {claim_id}<br>Severity: {int(row['CAT Severity Code'])}<br>Zone: {row['Cluster']}<br>Not assigned"
            
        folium.CircleMarker(
            location=(row["ACC_STD_LAT_NBR"], row["ACC_STD_LON_NBR"]),
            radius=5,
            color=cluster_colors[int(row["Cluster"]) % len(cluster_colors)],
            fill=True,
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(m)
    
    m.save(map_file)
    print(f"Map visualization saved as '{map_file}'")