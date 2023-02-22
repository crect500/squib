from typing import Union, Tuple

from qiskit import QuantumCircuit as qc, \
                    QuantumRegister as qr

def to_binary(NUMBER: Union[int, str],
              bits: int = -1) -> list:
    def reverse_and_convert(bit_string: str) -> list:
        binary_state = []
        for i in range(len(bit_string) - 1, -1, -1):
            if bit_string[i] == '0':
                binary_state.append(False)
            else:
                binary_state.append(True)
        return binary_state

    if type(NUMBER) == str:
        if bits != -1 and len(NUMBER) > bits:
            raise ValueError('{} bits is not enough to store {}'.format(bits, NUMBER))
        elif bits != -1 and len(NUMBER) < bits:
            binary_number = reverse_and_convert(NUMBER)
            binary_number = binary_number + [False] * (bits - len(NUMBER))
        else:
            binary_number = reverse_and_convert(NUMBER)

        return binary_number

    if NUMBER == 0 and bits != -1:
        return [False]*bits
    elif NUMBER == 0:
        return [False]

    quotient = NUMBER
    binary_number = []

    while quotient != 0:
        remainder = quotient % 2
        quotient = int(quotient / 2)
        if remainder == 0:
            binary_number.append(False)
        else:
            binary_number.append(True)

    if bits != -1 and len(binary_number) > bits:
        raise ValueError('{} bits required store number {}'.format(len(binary_number), NUMBER))

    if bits != -1 and len(binary_number) == bits:
        return binary_number

    if bits != -1:
        binary_number = binary_number + [False]*(bits - len(binary_number))

    return binary_number


def prep_binary_state(NUMBER: Union[int, str, list],
                        bits: int) -> qc:
    if type(NUMBER) != list:
        binary_state = to_binary(NUMBER, bits)
    elif len(NUMBER) < bits:
        binary_state = NUMBER + [False] * (bits - len(NUMBER))
    else:
        binary_state = NUMBER

    quantum_binary_state = qc(qr(bits, 'q0'))
    for i in range(0, bits):
        if binary_state[i]:
            quantum_binary_state.x(i)
    
    return quantum_binary_state


def prep_binary_operands(number1: Union[int, str],
                         number2: Union[int, str]) -> Tuple[list, list]:
    if not number1 and type(number1) is not int:
        raise ValueError('First operand does not contain a value')
    if not number2 and type(number2) is not int:
        raise ValueError('Second operand does not contain a value')

    binary_number1 = to_binary(number1)
    binary_number2 = to_binary(number2)

    bits1 = len(binary_number1)
    bits2 = len(binary_number2)

    if bits2 > bits1:
        binary_number1 = binary_number1 + [False] * (bits2 - bits1)
    elif bits1 > bits2:
        binary_number2 = binary_number2 + [False] * (bits1 - bits2)

    del bits1, bits2

    return binary_number1, binary_number2
