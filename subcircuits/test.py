import unittest
from math import pi
from qiskit import QuantumCircuit
from qft.qft import qft
from qft.qft_inv import qft_inv

class TestQft(unittest.TestCase):
    def test_qft(self):
        test_circuit = QuantumCircuit(3)
        test_circuit = qft(test_circuit)
        self.assertEqual(len(test_circuit), 6)
        self.assertEqual(test_circuit.data[3].operation.name, "h")
        self.assertEqual(test_circuit.data[5].operation.name, "h")
        error = abs(pi/4 - test_circuit.data[2].operation.params[0]) / (pi / 4)
        self.assertLess(error, 10e-6)

    def test_qft_inv(self):
        test_circuit = QuantumCircuit(3)
        test_circuit = qft_inv(test_circuit)
        self.assertEqual(len(test_circuit), 6)
        self.assertEqual(test_circuit.data[0].operation.name, "h")
        self.assertEqual(test_circuit.data[5].operation.name, "h")
        error = abs(pi/4 - test_circuit.data[3].operation.params[0]) / (pi / 4)
        self.assertLess(error, 10e-6)