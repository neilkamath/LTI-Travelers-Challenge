# LTI-Travelers-Challenge
Quantum Computing to Optimize Claim Handler Staffing for Target
Resolution, YQuantum 2025

This repository contains quantum computing applications for data analysis, specifically focused on clustering and optimization problems using tornado severity data.

## Quantum Clustering (`quantum-clustering/`)

Implementations of quantum-enhanced clustering algorithms:

- **Q-means**: A quantum-enhanced version of k-means clustering that uses quantum optimization to select optimal centroids
- **Elbow method determination**: Algorithms to determine the optimal number of clusters using both classical and quantum methods
- **Interactive visualizations**: HTML-based map visualizations of the resulting clusters

The implementation leverages Qiskit for quantum circuit design and QNexus for executing quantum jobs on Quantinuum's H1-1LE quantum system.

## Quantum Constraint Optimization (`quantum-constraint-graph/`)

Quantum approach to constraint optimization problem:

- **Claims assignment optimization**: Optimally assigns insurance claims to handlers based on severity and geographical constraints
- **Penalty minimization**: Minimizes penalties based on handler skill level and geographical zones
- **Visualization tools**: Includes bar charts for penalty comparison and interactive maps showing optimized assignments

## Geo-Temporal Load Balancing (`geo-temp-load-balancing/`)

Helper functions that use quantum computing for geo-temporal resource allocation optimization:

- **QUBO-based optimization**: Formulates geo-temporal load balancing as a Quadratic Unconstrained Binary Optimization problem

- **Quantinuum integration**: Connects directly to Quantinuum's quantum hardware (H1-2E) for solving complex optimization problems

- **Hybrid quantum-classical approach**: Uses QAOA (Quantum Approximate Optimization Algorithm) combined with classical pre-processing
