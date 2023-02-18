import pytest
from qiskit.circuit import QuantumCircuit as qc, \
                            QuantumRegister as qr

import utils


class TestUtils:
    @pytest.mark.parametrize('number,binary_number',
                             [(0, [0]*8),
                             (8, [0, 0, 0, 1, 0, 0, 0, 0])])
    def test_to_binary_default(self, number, binary_number):
        assert utils.to_binary(number) == binary_number

    @pytest.mark.parametrize('number,bits,binary_number',
                             [(0, 3, [0] * 3),
                              (8, 4, [0, 0, 0, 1])])
    def test_to_binary_bits(self, number, bits, binary_number):
        assert utils.to_binary(number, bits) == binary_number

    @pytest.mark.parametrize('number,bits,required_bits',
                             [(256, 8, 9),
                              (8, 4, 5)])
    def test_to_binary_overflow(self, number, bits, required_bits):
        try:
            utils.to_binary(number, bits)
        except ValueError as e:
            assert str(e) == '{} bits required store number {}'.format(required_bits, number)

    def test_prep_binary_state(self):
        solution_circuit = qc(qr(4, 'q0'))
        test_circuit = utils.prep_binary_state(0, 4)
        assert test_circuit.data == solution_circuit.data

        solution_circuit.x(0)
        solution_circuit.x(2)
        test_circuit = utils.prep_binary_state(5, 4)
        assert test_circuit.data == solution_circuit.data

        solution_circuit = qc(qr(8, 'q0'))
        solution_circuit.x(0)
        solution_circuit.x(2)
        solution_circuit.x(7)
        test_circuit = utils.prep_binary_state(133, 8)
        assert test_circuit.data == solution_circuit.data
