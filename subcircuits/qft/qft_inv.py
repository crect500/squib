from qiskit import QuantumCircuit
from math import pi


def qft_inv(qc: QuantumCircuit) -> QuantumCircuit:
    n = qc.num_qubits
    if n < 1:
        raise ValueError("The circuit must have at least one qubit")

    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            qc.cp(pi/(2**(j - i)), j, i)
        qc.h(i)

    return qc