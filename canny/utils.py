from typing import Union, Tuple
from math import log2

from qiskit import QuantumCircuit as qc, \
                    QuantumRegister as qr


def to_binary(NUMBER: Union[int, str],
              bits: int = -1) -> list:
    """
    Creates a list of booleans representing an unsigned binary integer.
    The least significant bit is in the 0 index

    Parameters
    ----------
    NUMBER : Union[int, str]
        Natural number to be converted
    bits : int
        Number of bits to be returned

    Returns
    -------
    list
    """
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

    if NUMBER < 0:
        raise ValueError('Cannot represent {} as a unsigned binary number'.format(NUMBER))

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

    if bits != -1:
        if len(binary_number) > bits:
            raise ValueError('{} bits required store number {}'.format(len(binary_number), NUMBER))

        if len(binary_number) == bits:
            return binary_number

        binary_number = binary_number + [False]*(bits - len(binary_number))

    return binary_number


def to_twos_complement(NUMBER: Union[int, str],
                       bits: int = -1) -> list:
    """
    Creates a list of booleans representing a signed binary integer
    in two's complement encoding the least significant bit is in the 0 index.


    Parameters
    ----------
    NUMBER : Union[int, str]
        Integer to be converted
    bits : int
        Number of bits to be returned

    Returns
    -------
    list
    """
    if bits != -1:
        if NUMBER == 0:
            return [False] * bits

        if NUMBER > 0:
            twos_binary = to_binary(NUMBER, bits - 1)
            return twos_binary + [False]
        else:
            complement = 2**(bits - 1) - abs(NUMBER)
            if complement < 0:
                raise ValueError('More bits required to store number {}'.format(NUMBER))
            return to_binary(complement, bits - 1) + [True]

    else:
        if NUMBER == 0:
            return [False]
        if NUMBER > 0:
            return to_binary(NUMBER) + [False]
        n = log2(abs(NUMBER))
        if float.is_integer(n):
            return [False] * int(n) + [True]
        complement = int(log2(int(n) + 1))
        return to_binary(complement, int(n) + 1) + [True]


def prep_binary_state(NUMBER: Union[int, str, list],
                        bits: int,
                        reverse: bool = False) -> qc:
    """
    Converts an unsigned integer into a quantum state of the size
    specified by bits. The least significant bit is stored in the
    first qubit

    Parameters
    ----------
    NUMBER : Union[int, str, list]
        Integer to be converted
    bits : int
        Number of bits to represent as qubits
    """

    if type(NUMBER) != list:
        binary_state = to_binary(NUMBER, bits)
    elif len(NUMBER) < bits:
        binary_state = NUMBER + [False] * (bits - len(NUMBER))
    else:
        binary_state = NUMBER

    if reverse:
        binary_state.reverse()

    quantum_binary_state = qc(qr(bits, 'q0'))
    for i in range(0, bits):
        if binary_state[i]:
            quantum_binary_state.x(i)
    
    return quantum_binary_state


def prep_binary_operands(NUMBER1: Union[int, str],
                         NUMBER2: Union[int, str]) -> Tuple[list, list]:
    """
    Creates two lists representing unsigned binary integers,
    ensuring that they are the same length.

    Parameters
    ----------
    NUMBER1 :
        A natural number
    NUMBER2 : Union[int, str]
        A natural number

    Returns
    -------
    Tuple[list, list]
    """
    if not NUMBER1 and type(NUMBER1) is not int:
        raise ValueError('First operand does not contain a value')
    if not NUMBER2 and type(NUMBER2) is not int:
        raise ValueError('Second operand does not contain a value')

    binary_number1 = to_binary(NUMBER1)
    binary_number2 = to_binary(NUMBER2)

    bits1 = len(binary_number1)
    bits2 = len(binary_number2)

    if bits2 > bits1:
        binary_number1 = binary_number1 + [False] * (bits2 - bits1)
    elif bits1 > bits2:
        binary_number2 = binary_number2 + [False] * (bits1 - bits2)

    del bits1, bits2

    return binary_number1, binary_number2

def prep_twos_complement_operands(NUMBER1: int,
                                  NUMBER2: int) -> Tuple[list, list]:
    """
    Creates two lists representing twos complement signed binary integers,
    ensuring that they are the same length.

    Parameters
    ----------
    NUMBER1 :
        An integer
    NUMBER2 : Union[int, str]
        An integer

    Returns
    -------
    Tuple[list, list]
    """
    if not NUMBER1 and type(NUMBER1) is not int:
        raise ValueError('First operand does not contain a value')
    if not NUMBER2 and type(NUMBER2) is not int:
        raise ValueError('Second operand does not contain a value')

    twos_number1 = to_twos_complement(NUMBER1)
    twos_number2 = to_twos_complement(NUMBER2)

    bits1 = len(twos_number1)
    bits2 = len(twos_number2)

    if bits2 > bits1:
        if NUMBER1 < 0:
            twos_number1 = twos_number1 + [True] * (bits2 - bits1)
        else:
            twos_number1 = twos_number1 + [False] * (bits2 - bits1)
    elif bits1 > bits2:
        if NUMBER2 < 0:
            twos_number2 = twos_number2 + [True] * (bits1 - bits2)
        else:
            twos_number2 = twos_number2 + [False] * (bits1 - bits2)

    del bits1, bits2

    return twos_number1, twos_number2
