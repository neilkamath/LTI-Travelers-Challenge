import json
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from pytket.extensions.qiskit import AerBackend, QuantinuumBackend
from pytket.extensions.qiskit import tk_to_qiskit
from qiskit.utils import QuantumInstance
from qiskit import transpile
from qiskit import QuantumCircuit
from pytket.backends import BackendResult


def load_qubo_from_file(filepath):
    with open(filepath, "r") as f:
        qubo_dict = json.load(f)

    qp = QuadraticProgram()
    variables = set()

    for (u, v) in qubo_dict.keys():
        variables.add(u)
        variables.add(v)

    for var in variables:
        qp.binary_var(var)

    linear = {}
    quadratic = {}

    for (u, v), coeff in qubo_dict.items():
        if u == v:
            linear[u] = linear.get(u, 0.0) + coeff
        else:
            quadratic[(u, v)] = quadratic.get((u, v), 0.0) + coeff

    qp.minimize(linear=linear, quadratic=quadratic)
    return qp


def solve_with_quantinuum(qp, api_key, device="H1-2E", shots=100):
    # Convert QuadraticProgram to QUBO
    converter = QuadraticProgramToQubo()
    qubo = converter.convert(qp)

    quant_backend = QuantinuumBackend(device_name=device, api_key=api_key)
    quant_backend.login()

    # Use pytket to generate a circuit from QAOA
    quantum_instance = QuantumInstance(backend=AerBackend().backend)
    qaoa = QAOA(optimizer=COBYLA(), quantum_instance=quantum_instance)
    optimizer = MinimumEigenOptimizer(qaoa)

    result = optimizer.solve(qubo)
    print(" QAOA finished with best result (simulated before sending to Quantinuum):")
    print(result)

    # Compile circuit and submit to Quantinuum
    qc = qaoa.ansatz
    transpiled = transpile(qc, backend=quant_backend.backend, optimization_level=3)
    job = quant_backend.process_circuit(transpiled, n_shots=shots)
    quant_backend.retrieve_job(job.job_id)
    counts = job.get_counts()
    
    print("Quantinuum raw result counts:")
    print(counts)


if __name__ == "__main__":
    api_key = ...
    qp = load_qubo_from_file("qubo_geo_temporal.json")
    solve_with_quantinuum(qp, api_key)