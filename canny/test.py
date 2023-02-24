import pytest
from qiskit.circuit import QuantumCircuit as qc, \
                            QuantumRegister as qr, \
                            AncillaRegister as ar

import utils
import operations


class TestUtils:
    @pytest.mark.parametrize('number,binary_number',
                             [(0, [False]),
                             (8, [False, False, False, True])])
    def test_to_binary_default(self,
                               number,
                               binary_number):
        assert utils.to_binary(number) == binary_number

    @pytest.mark.parametrize('number,bits,binary_number',
                             [(0, 3, [False] * 3),
                              (8, 4, [False, False, False, True])])
    def test_to_binary_bits(self,
                            number,
                            bits,
                            binary_number):
        assert utils.to_binary(number, bits) == binary_number

    @pytest.mark.parametrize('number,binary_number',
                             [(0, [False]),
                              (-1, [True]),
                              (1, [True, False]),
                              (-4, [False, False, True]),
                              (-3, [True, False, True])])
    def test_to_twos_complement(self,
                                number,
                                binary_number):
        assert utils.to_twos_complement(number) == binary_number

    @pytest.mark.parametrize('number,bits,binary_number',
                             [(0, 4, [False] * 4),
                              (-1, 3, [True] * 3),
                              (1, 5, [True] + [False] * 4),
                              (-4, 3, [False, False, True]),
                              (-5, 5, [True, True, False, True, True]),
                              (6, 4, [False, True, True, False])])
    def test_to_twos_complement_specified(self,
                                number,
                                bits,
                                binary_number):
        assert utils.to_twos_complement(number, bits) == binary_number

    @pytest.mark.parametrize('bit_string,binary_number',
                             [('010101', [True, False, True, False, True, False]),
                              ('00000', [False] * 5),
                              ('111111', [True] * 6)])
    def test_bit_string_to_binary(self,
                                  bit_string,
                                  binary_number):
        assert utils.to_binary(bit_string) == binary_number

    @pytest.mark.parametrize('bit_string,bits,binary_number',
                             [('010101', 6, [True, False, True, False, True, False]),
                              ('000', 5, [False] * 5),
                              ('111111', 9, [True] * 6 + [False] * 3)])
    def test_bit_string_to_binary_specified(self,
                                            bit_string,
                                            bits,
                                            binary_number):
        assert utils.to_binary(bit_string, bits) == binary_number

    @pytest.mark.parametrize('number,bits,required_bits',
                             [(256, 8, 9),
                              (8, 4, 5)])
    def test_to_binary_overflow(self,
                                number,
                                bits,
                                required_bits):
        try:
            utils.to_binary(number, bits)
        except ValueError as e:
            assert str(e) == '{} bits required store number {}'.format(required_bits, number)

    def test_prep_binary_state_int(self):
        # Test zero state
        solution_circuit = qc(qr(4, 'q0'))
        test_circuit = utils.prep_binary_state(0, 4)
        assert test_circuit.data == solution_circuit.data

        # Test 4 bit state
        solution_circuit.x(0)
        solution_circuit.x(2)
        test_circuit = utils.prep_binary_state(5, 4)
        assert test_circuit.data == solution_circuit.data

        # Test 8 bit state
        solution_circuit = qc(qr(8, 'q0'))
        solution_circuit.x(0)
        solution_circuit.x(2)
        solution_circuit.x(7)
        test_circuit = utils.prep_binary_state(133, 8)
        assert test_circuit.data == solution_circuit.data

    def test_prep_binary_state_str(self):
        # Test zero state
        solution_circuit = qc(qr(5, 'q0'))
        test_circuit = utils.prep_binary_state('00000', 5)
        assert test_circuit.data == solution_circuit.data

        # Test non-zero state
        solution_circuit.x(1)
        solution_circuit.x(3)
        test_circuit = utils.prep_binary_state('01010', 5)
        assert test_circuit.data == solution_circuit.data

    def test_prep_binary_state_list(self):
        # Test zero state
        solution_circuit = qc(qr(4, 'q0'))
        test_circuit = utils.prep_binary_state([False, False, False, False], 4)
        assert test_circuit.data == solution_circuit.data

        # Test non-zero state
        solution_circuit.x(0)
        solution_circuit.x(3)
        test_circuit = utils.prep_binary_state([True, False, False, True], 4)
        assert test_circuit.data == solution_circuit.data

    @pytest.mark.parametrize('number1, number2, solution1, solution2',
                             [(0, 2, [False, False], [False, True]),
                              (2, 0, [False, True], [False, False]),
                              (5, 7, [True, False, True], [True, True, True]),
                              ('110', '0101', [False, True, True, False], [True, False, True, False]),
                              ('1010', '010', [False, True, False, True], [False, True, False, False]),
                              ('1010', '0101', [False, True, False, True], [True, False, True, False])])
    def test_binary_operands(self,
                             number1,
                             number2,
                             solution1,
                             solution2):
        test_solution1, test_solution2 = utils.prep_binary_operands(number1, number2)
        assert test_solution1 == solution1
        assert test_solution2 == solution2

    @pytest.mark.parametrize('number1, number2, solution1, solution2',
                             [(0, 2, [False] * 3, [False, True, False]),
                              (2, 0, [False, True, False], [False] * 3),
                              (5, 7, [True, False, True, False], [True, True, True, False]),
                              (-1, 1, [True, True], [True, False]),
                              (1, -1, [True, False], [True, True]),
                              (-3, -4, [True, False, True], [False, False, True])])
    def test_twos_complement_operands(self,
                             number1,
                             number2,
                             solution1,
                             solution2):
        test_solution1, test_solution2 = utils.prep_twos_complement_operands(number1, number2)
        assert test_solution1 == solution1
        assert test_solution2 == solution2

class TestOperations:
    def test_quant_compare_int(self):
        # Test 0s
        qx = qr(1, 'x')
        qy = qr(1, 'y')
        qe = ar(2, 'e')
        circuit = qc(qx, qy, qe)
        test_circuit = operations.quant_compare_int(0, 0)
        assert circuit.qregs == test_circuit.qregs
        assert circuit.ancillas == circuit.ancillas
        assert circuit.num_qubits == test_circuit.num_qubits

        # Test same length
        qx = qr(3, 'x')
        qy = qr(3, 'y')
        qa = ar(4, 'a')
        qe = ar(2, 'e')
        circuit = qc(qx, qy, qa, qe)
        test_circuit = operations.quant_compare_int(5, 6)
        assert circuit.qregs == test_circuit.qregs
        assert circuit.ancillas == test_circuit.ancillas
        assert circuit.num_qubits == test_circuit.num_qubits

        # Test first register greater
        qx = qr(3, 'x')
        qy = qr(3, 'y')
        qa = ar(4, 'a')
        qe = ar(2, 'e')
        circuit = qc(qx, qy, qa, qe)
        test_circuit = operations.quant_compare_int(6, 1)
        assert circuit.qregs == test_circuit.qregs
        assert circuit.ancillas == test_circuit.ancillas
        assert circuit.num_qubits == test_circuit.num_qubits

        # Test second register greater
        qx = qr(3, 'x')
        qy = qr(3, 'y')
        qa = ar(4, 'a')
        qe = ar(2, 'e')
        circuit = qc(qx, qy, qa, qe)
        test_circuit = operations.quant_compare_int(2, 5)
        assert circuit.qregs == test_circuit.qregs
        assert circuit.ancillas == test_circuit.ancillas
        assert circuit.num_qubits == test_circuit.num_qubits

    def test_increment_state(self):
        gate = operations.increment_state(4)
        assert gate.num_qubits == 4
        assert gate.definition.data[0].operation.name == 'mcx'
        assert gate.definition.data[0].operation.num_qubits == 4
        assert gate.definition.data[-1].operation.name == 'x'

    def test_decrement_state(self):
        gate = operations.decrement_state(4)
        assert gate.num_qubits == 4
        assert gate.definition.data[0].operation.name == 'x'
        assert gate.definition.data[-1].operation.num_qubits == 4
        assert gate.definition.data[-1].operation.name == 'mcx'
