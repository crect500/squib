import unittest
from math import pi
from qiskit import QuantumCircuit
import qft.qft as qft

class TestQft(unittest.TestCase):
    def test_qft(self):
        test_circuit = QuantumCircuit(3)
        test_circuit = qft.qft(test_circuit)
        self.assertEqual(len(test_circuit), 6)
        self.assertEqual(test_circuit.data[3].operation.name, "h")
        self.assertEqual(test_circuit.data[5].operation.name, "h")
        error = abs(pi/4 - test_circuit.data[2].operation.params[0]) / (pi / 4)
        self.assertLess(error, 10e-6)