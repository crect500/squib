from qiskit import QuantumCircuit,  \
                    QuantumRegister, \
                    ClassicalRegister
from math import pi

def qpe(unitary_circuit: QuantumCircuit,
        ancilla_qubits: int) -> QuantumCircuit:
    num_qubits = unitary_circuit.num_qubits + ancilla_qubits
    qc = QuantumCircuit(QuantumRegister(num_qubits), ClassicalRegister(ancilla_qubits))
    for i in range(0, ancilla_qubits):
        qc.h(i)
    