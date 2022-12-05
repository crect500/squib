from qiskit import QuantumCircuit,  \
                    QuantumRegister, \
                    ClassicalRegister
from qiskit.circuit.library import HGate
from math import pi


def qft(qc: QuantumCircuit) -> QuantumCircuit:
    n = qc.num_qubits
    if n < 1:
        raise ValueError("The circuit must have at least one qubit")

    for i in range(0, n):
        qc.h(i)
        for j in range(i + 1, n):
            qc.cp(pi/(2**(j - i)), j, i)

    return qc