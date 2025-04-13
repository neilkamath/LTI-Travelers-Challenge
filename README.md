# LTI-Travelers-Challenge
Quantum Computing to Optimizine Claim Handler Staffing for Target
Resolution, YQuantum 2025

This repository contains quantum computing applications for data analysis, specifically focused on clustering and optimization problems using tornado severity data.

## Quantum Clustering (`quantum-clustering/`)

This folder contains implementations of quantum-enhanced clustering algorithms:

- **Q-means**: A quantum-enhanced version of k-means clustering that uses quantum optimization to select optimal centroids
- **Elbow method determination**: Algorithms to determine the optimal number of clusters using both classical and quantum methods
- **Interactive visualizations**: HTML-based map visualizations of the resulting clusters

The implementation leverages Qiskit for quantum circuit design and QNexus for executing quantum jobs on Quantinuum's H1-1LE quantum system.

## Quantum Constraint Optimization (`quantum-constraint-graph/`)

This folder demonstrates quantum approaches to constraint optimization problems:

- **Claims assignment optimization**: Optimally assigns insurance claims to handlers based on severity and geographical constraints
- **Penalty minimization**: Minimizes penalties based on handler skill level and geographical zones
- **Visualization tools**: Includes bar charts for penalty comparison and interactive maps showing optimized assignments