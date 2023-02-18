from typing import Union
from qiskit import QuantumCircuit as qc, \
                    QuantumRegister as qr

def to_binary(NUMBER: int,
              bits: int = 8) -> list:
    if NUMBER == 0:
        return [0]*bits

    quotient = NUMBER
    binary_number = []

    while quotient != 0:
        remainder = quotient % 2
        quotient = int(quotient / 2)
        binary_number.append(remainder)

    if len(binary_number) > bits:
        raise ValueError('{} bits required store number {}'.format(len(binary_number), NUMBER))

    if len(binary_number) == bits:
        return binary_number

    binary_number = binary_number + [0]*(bits - len(binary_number))
    return binary_number

def prep_binary_state(number: int,
                        bits: int) -> qc:
    binary_state = to_binary(number, bits)
    quantum_binary_state = qc(qr(bits, 'q0'))
    for i in range(0, bits):
        if binary_state[i] == 1:
            quantum_binary_state.x(i)
    
    return quantum_binary_state
