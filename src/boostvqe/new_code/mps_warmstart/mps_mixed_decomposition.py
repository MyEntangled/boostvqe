from src.boostvqe.new_code.mps_warmstart import mps_analytic_decomposition, mps_optimized_decomposition
from src.boostvqe.new_code.mps_warmstart.create_mps_circuit import generate_gates_from_unitaries
from src.boostvqe.new_code.mps_warmstart.helper import flatten_list, apply_circuit_mps

import quimb.tensor as qtn
import quimb as qu
import numpy as np

from src.boostvqe.new_code.mps_warmstart.mps_optimized_decomposition import optimized_decomposition


def mixed_decomposition(psi_target: qtn.MatrixProductState,
                        hamiltonian=None,
                        circuit_layers=10,
                        opt_sweeps=10,
                        fid_target=0.99,
                        use_raw_gates=True):
    """
    Perform a mixed decomposition of a target MPS, using both the optimized and analytic decompositions.
    """

    if isinstance(psi_target, qtn.CircuitMPS):
        psi_target = psi_target.psi

    n = psi_target.num_tensors
    mps_circ = qtn.CircuitMPS(n, max_bond=4096, cutoff=1e-8)
    fid = np.abs(psi_target.H @ mps_circ.psi)

    circuit_unitaries = [] ## Including sublists for layers
    circuit_qargs = []
    circuit_gates = [] ## No sublist, just every gate so far

    layer = 0
    while layer < circuit_layers and fid < fid_target:
        print(f"Layer {layer}")
        print(f"Initial fidelity: {fid}")

        # New layer of analyitic decomp.
        print("Analytic decomposition..")

        layer_ana_unitaries, layer_qargs, fid, energy = mps_analytic_decomposition.analytic_decomposition(
            psi_target=apply_circuit_mps(n, circuit_gates, apply_inverse=True, psi0=psi_target),
            num_layers=1,
            hamiltonian=hamiltonian,
            fid_target=fid_target,
            use_raw_gates=use_raw_gates
        )

        assert len(layer_ana_unitaries) == 1
        assert len(layer_qargs) == 1

        layer_ana_unitaries = layer_ana_unitaries[0] # index 0 as there's only 1 layer
        layer_qargs = layer_qargs[0]
        layer_ana_gates = list(generate_gates_from_unitaries(layer_ana_unitaries, layer_qargs, use_raw_gates=use_raw_gates))

        ana_circ = apply_circuit_mps(n, layer_ana_gates + circuit_gates)

        fid = np.abs(psi_target.H @ ana_circ.psi)

        print(f"Post ana-decomp fidelity: {fid}")

        # Optimizing all layers so far
        print("Optimized decomposition..")
        circuit_unitaries, circuit_qargs, fid = optimized_decomposition(psi_target,
                                                                        layer_ana_unitaries + circuit_unitaries ,
                                                                        layer_qargs + circuit_qargs,
                                                                        opt_sweeps=opt_sweeps,
                                                                        fid_target=fid_target,
                                                                        r=0.6,
                                                                        use_raw_gates=use_raw_gates)

        circuit_gates = list(generate_gates_from_unitaries(circuit_unitaries, circuit_qargs, use_raw_gates=use_raw_gates))

        opt_circ = apply_circuit_mps(n, circuit_gates)
        fid = np.abs(psi_target.H @ opt_circ.psi)

        print(f"Post opt-decomp fidelity: {fid}")
        print()

        layer += 1

    return circuit_unitaries, circuit_qargs, circuit_gates, fid

if __name__ == '__main__':
    nqubits = 20
    bond_dim = 64

    Jx = 1  # Coupling in the x-direction
    Jy = 1  # Coupling in the y-direction
    Jz = 0  # Coupling in the z-direction
    h = +0.5  # Transverse field strength

    # Build the Hamiltonian
    #ham_build, H = build_xxz_hamiltonian(nqubits, [Jx, Jy, Jz, h])
    ham = qtn.MPO_ham_heis(nqubits, (4*Jx, 4*Jy, 4*Jz), bz=-2*h)

    #print((ham_build - ham).norm())
    bond_dims = [64]
    dmrg = qtn.DMRG2(ham, bond_dims=bond_dims, cutoffs=1e-6)
    res = dmrg.solve(verbosity=0)
    dmrg.solve()
    psi = dmrg.state

    unitaries, qargs, gates, fid = mixed_decomposition(psi, None, 3, 2, 1)
    ## Reconstruct circuit
    circ = apply_circuit_mps(nqubits, gates)

