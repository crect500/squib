from typing import Union

from qiskit import QuantumCircuit as qc, \
                    QuantumRegister as qr, \
                    AncillaRegister as ar
from qiskit.circuit import Gate
from qiskit.circuit.library import XGate
from qiskit.converters import circuit_to_gate as ctg

import utils


def quant_compare_int(number1: Union[int, str],
                      number2: Union[int, str],
                      as_circuit: bool = False) -> Union[Gate, qc]:
    """
    Creates a quantum circuit which encodes the comparison between two unsigned
    binary integers as one of three binary states in the last 2-qubit register

    00 : Numbers are equal
    10 : First number is greater
    01 : Second number is greater

    Parameters
    ----------
    number1 : Union[int, str]
        A natural number
    number2 : Union[int, str]
        A natural number
    as_circuit: bool
        Optionally returns a quantum circuit

    Returns
    -------
    Union[qiskit.circuit.Gate, qiskit.QuantumCircuit]
    """
    if number1 < 0:
        raise ValueError('Operation is not supported for negative number {}'.format(number1))
    if number2 < 0:
        raise ValueError('Operation is not supported for negative number {}'.format(number2))

    binary_number1, binary_number2 = utils.prep_binary_operands(number1, number2)
    n = len(binary_number1)
    m = 2 * n

    qx = qr(n, 'x')
    qy = qr(n, 'y')
    qe = ar(2, 'e')
    if n != 1:
        qa = ar(m - 2, 'a')
        circuit = qc(qx, qy, qa, qe)
    else:
        circuit = qc(qx, qy, qe)

    binary_gate1 = ctg(utils.prep_binary_state(binary_number1, n, reverse=True), label='to_bin')
    binary_gate2 = ctg(utils.prep_binary_state(binary_number2, n, reverse=True), label='to_bin')

    circuit.append(binary_gate1, qx)
    circuit.append(binary_gate2, qy)

    for i in range(0, n):
        gate1 = XGate().control(num_ctrl_qubits=2*(i+1), ctrl_state=2**((i + 1) * 2 - 2))
        gate2 = XGate().control(num_ctrl_qubits=2*(i+1), ctrl_state=2**((i + 1) * 2 - 1))
        control = []
        if i != 0:
            for qubit in qa[:(2 * i)]:
                control.append(qubit)

        if i != (n - 1):
            control.append(qx[i])
            control.append(qy[i])
            control.append(qe[0])
            circuit.append(gate1, control)
            control[-1] = qa[2 * i]
            circuit.append(gate1, control)
            control[-1] = qe[1]
            circuit.append(gate2, control)
            control[-1] = qa[(2 * i) + 1]
            circuit.append(gate2, control)
        else:
            control.append(qx[i])
            control.append(qy[i])
            control.append(qe[0])
            circuit.append(gate1, control)
            control[-1] = qe[1]
            circuit.append(gate2, control)

    if as_circuit:
        return circuit

    return ctg(circuit, label='comp')


def increment_state(num_qubits: int,
                    as_circuit: bool = False,
                    reverse: bool = False) -> Union[Gate, qc]:
    """
    Increments the binary number represented by the quantum state by 1

    Parameters
    ----------
    num_qubits : qiskit.QuantumCircuit
        The width of the quantum circuit representing a binary state,
        with the least significant bit as the last qubit
    as_circuit : bool
        Optionally returns a circuit
    reverse: bool
        Optionally use apply with the least significant bit as first qubit

    Returns
    -------
    Union[qiskit.circuit.Gate, qiskit.QuantumCircuit]
    """
    if num_qubits < 1:
        raise ValueError('Cannot create gate for non-positive number of qubits {}'.format(num_qubits))

    if reverse:
        decrement_state(num_qubits, as_circuit=as_circuit)

    register = qr(num_qubits, 'q0')
    circuit = qc(register)
    for i in range(num_qubits - 1, 0, -1):
        gate = XGate().control(i)
        qubit_list = []
        for j in range(num_qubits - i, num_qubits):
            qubit_list.append(j)
        qubit_list.append(num_qubits - i - 1)
        circuit.append(gate, qubit_list)
    circuit.x(num_qubits - 1)

    if as_circuit:
        return circuit

    return ctg(circuit, label='++')

def decrement_state(num_qubits: int,
                    as_circuit: bool = False,
                    reverse: bool = False) -> Union[Gate, qc]:
    """
    Decrements the binary number represented by the quantum state by 1

    Parameters
    ----------
    num_qubits : qiskit.QuantumCircuit
        The width of the quantum circuit representing a binary state,
        with the least significant bit as the last qubit
    as_circuit : bool
        Optionally returns a circuit
    reverse: bool
        Optionally use apply with the least significant bit as first qubit

    Returns
    -------
    Union[qiskit.circuit.Gate, qiskit.QuantumCircuit]
    """
    if num_qubits < 1:
        raise ValueError('Cannot create gate for non-positive number of qubits {}'.format(num_qubits))

    if reverse:
        increment_state(num_qubits, as_circuit=as_circuit)

    register = qr(num_qubits, 'q0')
    circuit = qc(register)
    circuit.x(num_qubits - 1)
    for i in range(1, num_qubits):
        gate = XGate().control(i)
        qubit_list = []
        for j in range(num_qubits - 1, num_qubits - i - 1, -1):
            qubit_list.append(j)
        qubit_list.append(num_qubits - i - 1)
        circuit.append(gate, qubit_list)

    if as_circuit:
        return circuit

    return ctg(circuit, label='--')


def copy_state(num_qubits: int,
               as_circuit: bool = False) -> Union[Gate, qc]:
    """
    Copies a state into an ancillary register

    Parameters
    ----------
    num_qubits : qiskit.QuantumCircuit
        The width of the quantum circuit representing a binary state,
        with the least significant bit as the last qubit
    as_circuit : bool
        Optionally returns a circuit

    Returns
    -------
    Union[qiskit.circuit.Gate, qiskit.QuantumCircuit]
    """
    if num_qubits < 1:
        raise ValueError('Cannot create gate for non-positive number of qubits {}'.format(num_qubits))

    register = qr(num_qubits, 'q0')
    copy_register = qr(num_qubits, 'q1')
    circuit = qc(register, copy_register)
    for i in range(0, num_qubits):
        circuit.cx(register[i], copy_register[i])

    if as_circuit:
        return circuit

    return ctg(circuit, label='copy')


def flip_state(num_qubits: int,
               as_circuit: bool = False) -> Union[Gate, qc]:
    """
    Flips the state of all qubits

    Parameters
    ----------
    num_qubits : qiskit.QuantumCircuit
        The width of the quantum circuit representing a binary state,
        with the least significant bit as the last qubit
    as_circuit : bool
        Optionally returns a circuit

    Returns
    -------
    Union[qiskit.circuit.Gate, qiskit.QuantumCircuit]
    """
    if num_qubits < 1:
        raise ValueError('Cannot create gate for non-positive number of qubits {}'.format(num_qubits))

    register = qr(num_qubits, 'q0')
    circuit = qc(register)
    circuit.x(register)

    if as_circuit:
        return circuit

    return ctg(circuit, label='flip')


def flip_plus_one(num_qubits: int,
                  as_circuit: bool = False) -> Union[Gate, qc]:
    """
    Flips the state of all qubits and then increments the resultant state

    Parameters
    ----------
    num_qubits : qiskit.QuantumCircuit
        The width of the quantum circuit representing a binary state,
        with the least significant bit as the last qubit
    as_circuit : bool
        Optionally returns a circuit

    Returns
    -------
    Union[qiskit.circuit.Gate, qiskit.QuantumCircuit]
    """
    if num_qubits < 1:
        raise ValueError('Cannot create gate for non-positive number of qubits {}'.format(num_qubits))

    register = qr(num_qubits, 'q0')
    circuit = qc(register)
    circuit.x(register)
    circuit.append(increment_state(num_qubits), register)

    if as_circuit:
        return circuit

    return ctg(circuit, label='flip1')