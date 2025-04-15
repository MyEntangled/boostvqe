import quimb.tensor as qtn
import numpy as np

def apply_circuit_mps(circuit: qtn.CircuitMPS | int, gates, apply_inverse=False, psi0=None):
    """
    Apply a list gates using a CircuitMPS. The circuit can be initialized to psi0 when created new.
    """
    if isinstance(circuit, int):
        circuit = qtn.CircuitMPS(circuit, psi0=psi0, max_bond=4096, cutoff=1e-8, to_backend=None)

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

def append_circuits(circuit1, circuit2):
    # Append all gates from circuit2 to circuit1
    # for gate in circuit2.gates:
    #     circuit1.apply_gate(gate)
    # return circuit1

    circuit1.apply_gates(gates=circuit2.gates)
    return circuit1


def invert_circuit(circuit):
    inverse_circuit = qtn.Circuit(circuit.N)

    # Reverse the gate application
    for gate in reversed(circuit.gates):
        label = gate.label
        qubits = gate.qubits

        if isinstance(gate.params, str) and gate.params == 'raw':
            gate_to_apply = qtn.Gate.from_raw(gate.array.T.conj(), qubits=qubits)

        else:
            if gate.params is None:
                params = None
            else:
                if label == 'U3':
                    params = [-gate.params[0], -gate.params[2], -gate.params[1]]
                else:
                    try:
                        params = [-p for p in gate.params]  # Negate parameters
                    except TypeError:
                        params = None
            gate_to_apply = qtn.Gate(label, params, qubits)


        inverse_circuit.apply_gate(gate_to_apply)

    return inverse_circuit


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    qc1 = qtn.CircuitMPS(4)
    qc1.x(0)
    qc1.x(1)

    qc2 = qtn.CircuitMPS(4)
    qc2.h(1)
    qc2.h(2)

    qc = append_circuits(qc1, qc2)

    print(qc.gates)
    qc_inv = invert_circuit(qc)
    print(qc_inv.gates)


