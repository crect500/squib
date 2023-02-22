from typing import Union

from qiskit import QuantumCircuit as qc, \
                    QuantumRegister as qr, \
                    AncillaRegister as ar
from qiskit.circuit import ControlledGate
from qiskit.circuit.library import XGate
from qiskit.converters import circuit_to_gate as ctg

import utils


def quant_compare_int(number1: Union[int, str],
                          number2: Union[int, str]):
    binary_number1, binary_number2 = utils.prep_binary_operands(number1, number2)
    binary_number1.reverse()
    binary_number2.reverse()
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

    binary_gate1 = ctg(utils.prep_binary_state(binary_number1, n), label='to_bin')
    binary_gate2 = ctg(utils.prep_binary_state(binary_number2, n), label='to_bin')

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

    return circuit
