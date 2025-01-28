import numpy as np
import scipy

from qiskit.synthesis import OneQubitEulerDecomposer, TwoQubitBasisDecomposer
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.quantum_info.random import random_unitary

ONE_QUBIT_UNITARY_DECOMPOSER = OneQubitEulerDecomposer('U3')
TWO_QUBIT_UNITARY_DECOMPOSER = TwoQubitBasisDecomposer(CXGate(), euler_basis='U3')

def is_unitary(G:np.array, atol=1e-9) -> bool:
    if G.shape[0] != G.shape[1]:
        return False  # Not square, cannot be unitary
    identity = np.eye(G.shape[0])
    return np.allclose(G.conj().T @ G, identity, atol=atol)

def closest_unitary(A:np.array):
    """ Calculate the unitary matrix U that is closest with respect to the
        operator norm distance to the general matrix A.

        Return U as a numpy matrix.
    """
    V, _, Wh = scipy.linalg.svd(A,)
    U = np.matrix(V.dot(Wh))
    return U

def unitary_to_gates(G:np.array):
    #assert is_unitary(G)

    assert G.shape[0] == G.shape[1]
    G = closest_unitary(G)

    d = G.shape[0]
    assert d in [2,4]

    if d == 2:
        qc = ONE_QUBIT_UNITARY_DECOMPOSER(G, atol=1e-10)
    else:
        qc = TWO_QUBIT_UNITARY_DECOMPOSER(G)
    # print(qc.draw())
    for idx, instruction in enumerate(qc.data):
        operation = instruction.operation
        qubits = instruction.qubits
        yield (operation.name, [qc.find_bit(q).index for q in qubits], operation.params)

if __name__ == '__main__':
    G = random_unitary(4).data
    print(G)

    print(closest_unitary(G))
    print(np.linalg.norm(closest_unitary(G) - G))

    # #G = -np.eye(2)
    # gen_gates = unitary_to_gates(G)
    # for gate in gen_gates:
    #     print(gate)