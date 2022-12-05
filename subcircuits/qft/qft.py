from qiskit import QuantumCircuit,  \
                    QuantumRegister, \
                    ClassicalRegister
from qiskit.circuit.library import HGate
from math import log2, pi


def qft(qc: QuantumCircuit) -> QuantumCircuit:
    n = qc.num_qubits
    for i in range(0, n):
        qc.h(i)
        for j in range(i + 1, n):
            qc.cp(pi/(2**(j - i)), j, i)

    return qc
        

if __name__ == "__main__":
    circuit = QuantumCircuit(4)
    circuit = qft(circuit)
    print(circuit)