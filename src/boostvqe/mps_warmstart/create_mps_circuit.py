import numpy as np
import quimb.tensor as qtn
import quimb as qu
import matplotlib.pyplot as plt

from typing import List

from compile_gates import unitary_to_gates
from mps_to_circuit import disentangling_gates

#qtn.set_contract_backend('jax')
#qtn.set_tensor_linop_backend('jax')

def create_circuit_from_gate_unitaries(unitaries:List[List[np.array]]):
    num_layers = len(unitaries)
    for i in range(num_layers):
        layer = unitaries[num_layers-1-i] # Compose circuit starting from the last layer

        for n, unitary in enumerate(layer):

            if n == len(layer) - 1:
                assert unitary.shape[0] == 2
                qubits_involved = [len(layer)-1]
            else:
                assert unitary.shape[0] == 4
                qubits_involved = [n, n+1]

            gate_seq = unitary_to_gates(unitary)

            for (gate_name, qubit_order, params) in gate_seq:
                #print(gate_name, qubit_order, params)
                if len(params) > 0:
                    yield qtn.Gate(gate_name, params=params, qubits=[qubits_involved[q] for q in qubit_order], round=i)
                else:
                    yield qtn.Gate(gate_name, params=None, qubits=[qubits_involved[q] for q in qubit_order], round=i)


if __name__ == "__main__":
    nqubits = 100
    ham = qtn.MPO_ham_heis(nqubits)
    dmrg = qtn.DMRG2(ham, bond_dims=[16, 32, 64, 128], cutoffs=1e-6)
    res = dmrg.solve(verbosity=0)
    gs = dmrg.state
    print('Finding ground state..')
    print(gs)
    print(f'Estimate energy = {dmrg.energy}, True energy = {qu.heisenberg_energy(nqubits)}')
    print('-------------')

    print('Converting MPS state to circuit')

    #qtn.set_contract_backend('jax')
    #qtn.set_tensor_linop_backend('jax')

    circuit_unitaries = disentangling_gates(input_mps=gs, num_layers=10, hamiltonian=ham)
    circuit_gates = list(create_circuit_from_gate_unitaries(circuit_unitaries))
    #print(circuit_gates)

    circ = qtn.CircuitMPS(nqubits)
    circ.apply_gates(circuit_gates)


    fig, ax = circ.draw()
    #print(plt.show())

    print('----------')


    print(f'Norm = {circ.psi.H @ circ.psi}')

    overlap = abs(gs.H @ circ.psi)**2
    print(overlap)

    energy = (circ.psi.conj().reindex_({f'k{n}': f'b{n}' for n in range(circ.psi.num_tensors)}) | ham | circ.psi)^...
    print(energy)