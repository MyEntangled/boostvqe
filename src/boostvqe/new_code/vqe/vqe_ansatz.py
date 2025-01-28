import quimb.tensor as qtn

def double_ladder_ansatz(n_sites:int, layers:int, params, rotation_gate='RY', entangling_gate='CZ'):
    """
    Create a VQE ansatz circuit with a given number of sites and layers.
    """

    circuit = qtn.CircuitMPS(n_sites, max_bond=128, cutoff=10 ** -6)
    # Apply parameterized rotations and interactions
    idx = 0
    for _ in range(layers):
        for i in range(n_sites):
            circuit.apply_gate(rotation_gate, params[idx], i)
            idx += 1
            # circuit.apply_gate('RX', params[idx], i)
            # idx +=1
            # circuit.apply_gate('RY', params[idx], i)
            # idx +=1
        for i in range(0, n_sites - 1, 2):
            circuit.apply_gate(entangling_gate, i, i + 1)
            # circuit.apply_gate('RXX', params[idx], i, i + 1)
            # idx +=1

        for i in range(1, n_sites - 1, 2):
            circuit.apply_gate(entangling_gate, i, i + 1)
            # circuit.apply_gate('RXX', params[idx], i, i + 1)
            # idx +=1

    return circuit