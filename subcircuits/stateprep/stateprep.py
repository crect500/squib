import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit.circuit.library.standard_gates import Hgate

import math

b = np.array([1, 2, 3, 4, 5, 6, 7, 8])
b_norm = np.linalg.norm(b)
y = b / b_norm
n = math.log2(len(y))

class State_node:
    __slots__ = 'left', 'right', 'data', 'depth'
    
    def __init__(self, data, left = None, right = None):
        self.left = left
        self.right = right
        self.data = data
        self.depth = None
        
    def __str__(self):
        return str(self.data)
    
    def build_tree(self, depth):
        if (depth > 0):
            self.depth = depth
            self.left = State_node(0)
            self.right = State_node(0)
            self.left.build_tree(depth - 1)
            self.right.build_tree(depth - 1)
        return self
    
    def store_data(self, b: np.array):
        if len(b) == 2:
            self.left.data = b[0]
            self.right.data = b[1]
        else:
            mid = int(len(b)/2)
            self.left.store_data(b[0:mid])
            self.right.store_data(b[mid:])
            
    def fill_tree(self):
        if self.depth == None:
            return self.data
        else:
            self.data = math.sqrt(self.left.fill_tree()**2 + self.right.fill_tree()**2)
            return self.data
        
    def gen_angles(self):
        if self.data != 0:
            if self.left.data > 0:
                self.data = math.asin(self.right.data / self.data)
            else:
                self.data = math.pi * math.asin(self.right.data / self.data)

        else:
            self.data = 0
        
        self.depth = self.depth - 1
        
        if self.depth == 0:
            self.left = None
            self.right = None
        else:
            self.left.gen_angles()
            self.right.gen_angles()

depth = math.log2(len(y))
Angle_tree = State_node(1)
Angle_tree.build_tree(depth)
Angle_tree.store_data(y)
Angle_tree.fill_tree()
Angle_tree.gen_angles()

simulator = QasmSimulator()

qc = QuantumCircuit(n, n)
qc.u(math.pi/2,0,0,0)
controlled_U = Hgate().control(1)
qc.append(controlled_U)
qc.measure([0,1,2], [0,1,2])

circuit = transpile(qc, simulator)
job = simulator.run(circuit, shots=1000)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(circuit)

qc.draw()
plot_histogram(counts)