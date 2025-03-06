import numpy as np
from quimb import tensor as qtn
import torch

def to_backend(x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.tensor(x, dtype=torch.complex64, device=device)


def apply_circuit_mps(circuit: qtn.CircuitMPS | int, gates, apply_inverse=False, backend='numpy'):
    """
    Apply a list gates using a CircuitMPS
    """
    if isinstance(circuit, int):
        circuit = qtn.CircuitMPS(circuit, max_bond=4096, cutoff=1e-8, to_backend=None)

    if not apply_inverse:
        circuit.apply_gates(gates)
    else: ## Apply the inverse of the gates (order also reversed)
        inverse_gates = []
        for gate in reversed(gates):
            label = gate.label
            qubits = gate.qubits

            if gate.params is None:
                params = None
            else:
                if label == 'U3':
                    params = [-gate.params[0], -gate.params[2], -gate.params[1]]
                else:
                    try:
                        params = [-p for p in gate.params] # Negate parameters
                    except TypeError:
                        params = None
            inverse_gates.append(qtn.Gate(label, params, qubits))
        circuit.apply_gates(inverse_gates)

    return circuit

def flatten_list(list_of_lists):
    """
    Flatten a list of lists into a single list.
    """
    return [item for sublist in list_of_lists for item in sublist]


def is_unitary(U):
    return np.allclose(U.conj().T @ U, np.eye(U.shape[0]))
