import numpy as np
import quimb.tensor as qtn
import quimb as qu

from typing import List

from src.boostvqe.new_code.mps_warmstart.compile_gates import unitary_to_gates
from src.boostvqe.new_code.mps_warmstart.helper import flatten_list

def generate_gates_from_unitaries(unitaries:List[np.array], qargs:List[tuple]):
    assert len(qargs) == len(unitaries)

    for n, unitary in enumerate(unitaries):
        qubits_involved = qargs[n]

        gate_seq = unitary_to_gates(unitary)

        for (gate_name, qubit_order, params) in gate_seq:
            # print(gate_name, qubit_order, params)
            if len(params) > 0:
                yield qtn.Gate(gate_name, params=params, qubits=[qubits_involved[q] for q in qubit_order])
            else:
                yield qtn.Gate(gate_name, params=None, qubits=[qubits_involved[q] for q in qubit_order])


if __name__ == "__main__":
    from src.boostvqe.new_code.mps_warmstart.mps_analytic_decomposition import analytic_decomposition
    nqubits = 10
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

    circuit_unitaries, qargs,_,_ = analytic_decomposition(psi_target=gs, num_layers=10, hamiltonian=ham)

    circuit_gates = [generate_gates_from_unitaries(circuit_unitaries[k], qargs[k]) for k in range(len(circuit_unitaries))]
    circuit_gates = flatten_list(circuit_gates)


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