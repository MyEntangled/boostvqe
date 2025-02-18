from hamiltonian import build_xxz_hamiltonian
from dbi import dbi_training
from src.boostvqe.new_code.dbi.dbi_training import train_dbi
from vqe.vqe_training import evaluate, train_vqe
#from vqe.vqe_ansatz import double_ladder_ansatz
from src.boostvqe.new_code.mps_warmstart import create_mps_circuit, mps_to_unitaries
import numpy as np
import quimb.tensor as qtn

import tqdm

import matplotlib.pyplot as plt


def evaluate_dbi(warmstart_circuit, couplings, H):
    dbi_best = 0

    params_d = np.random.uniform(0., 2 * np.pi, 2 * n_sites - 1)

    for s in tqdm.tqdm(np.logspace(-2, 0, 200)):
        psi = dbi_training.dbi_circuit(warmstart_circuit, couplings, s, params_d)
        energy = evaluate(psi, H)
        dbi_best = min(dbi_best, energy)

    return dbi_best

def optimize_dbi(warmstart_circuit, couplings, hamiltonian):
    dbi_params, dbi_energy = train_dbi(couplings, hamiltonian, warmstart_circuit)
    print(f"DBI optimization done, optimal params = {dbi_params}")
    return dbi_energy

def dbi_with_mps(hamiltonian, H, circuit_layers=None):
    # DMRG
    bond_dims = [16, 32, 64, 128]
    dmrg = qtn.DMRG2(hamiltonian, bond_dims=bond_dims, cutoffs=1e-6)
    res = dmrg.solve(verbosity=0)
    dmrg.solve()
    gs = dmrg.state

    if circuit_layers is None:
        circuit_layers = int(np.ceil(np.log2(gs.max_bond())))

    print("Norm", gs.H @ gs)
    print("Shape", gs.shape)
    print("Bond dim", gs.max_bond())

    print('Finding ground state..')
    print('Ground state energy:', dmrg.energy)
    print()

    # Train the VQE
    # print('Optimizing VQE ansatz..')
    # layers = 4
    # vqe_params, vqe_energy = train_vqe(double_ladder_ansatz, n_sites, layers, H)
    # warmstart_vqe = double_ladder_ansatz(n_sites, layers, vqe_params)
    # print('VQE energy:', vqe_energy)
    # print()

    # Create the MPS circuit
    print('Creating MPS circuit..')
    circuit_unitaries, init_overlap, init_energy = mps_to_unitaries.disentangling_gates(input_mps=gs, num_layers=circuit_layers, hamiltonian=hamiltonian)
    circuit_gates = list(create_mps_circuit.create_circuit_from_gate_unitaries(circuit_unitaries))

    warmstart_mps = qtn.CircuitMPS(n_sites)
    warmstart_mps.apply_gates(circuit_gates)
    mps_energy_contract = (warmstart_mps.psi.conj().reindex_(
        {f'k{n}': f'b{n}' for n in range(warmstart_mps.psi.num_tensors)}) | hamiltonian | warmstart_mps.psi) ^ ...
    #mps_energy = evaluate(warmstart_mps, H)
    print('MPS circuit energy:', mps_energy_contract)
    print("Optimizing DBI...")

    # Evaluate the DBI
    # dbi_vqe = evaluate_dbi(warmstart_vqe, couplings, H)
    #dbi_mps = evaluate_dbi(warmstart_mps, couplings, H)
    dbi_mps = optimize_dbi(warmstart_mps, couplings, hamiltonian)
    print('------')
    #print('True gs:', dmrg.energy)
    # print('VQE, VQE+DBI:', vqe_energy, dbi_vqe)
    #print('MPS, MPS+DBI:', mps_energy, dbi_mps)
    # print(warmstart_vqe.psi)
    #print(warmstart_mps.gates)

    return dbi_mps, init_overlap, init_energy, dmrg.energy


if __name__ == '__main__':
    # Define parameters for the XXZ Hamiltonian
    Jx = 1  # Coupling in the x-direction
    Jy = 1  # Coupling in the y-direction
    Jz = 1 # Coupling in the z-direction
    h = +0.5  # Transverse field strength
    couplings = np.array([Jx, Jy, Jz, h])

    #nsites_range = range(61,101,10)
    nsites_range = [14,16]

    init_fid = []
    init_energy_error = []
    dbi_energy_error = []

    for n_sites in nsites_range:
        print(f"Num sites = {n_sites}")
        # Build the Hamiltonian
        ham_build, H = build_xxz_hamiltonian(n_sites, couplings)
        ham = qtn.MPO_ham_heis(n_sites, (4*Jx, 4*Jy, 4*Jz), bz=-2*h)
        assert abs((ham_build - ham).norm()) < 1e-10

        dbi_energy, init_overlap, init_energy, ground_energy = dbi_with_mps(ham, H, circuit_layers=10)
        print(f"n_sites {n_sites}, DBI energy: {dbi_energy}, init overlap: {init_overlap}, init energy: {init_energy}, ground energy: {ground_energy}")
        print()
        init_fid.append(init_overlap)
        init_energy_error.append((init_energy - ground_energy) / n_sites)
        dbi_energy_error.append((dbi_energy - ground_energy) / n_sites)



    # plt.plot(nsites_range, init_fid, label="Initial overlap")
    # plt.plot(nsites_range, init_energy_error, label="Initial avg energy error")
    # plt.plot(nsites_range, dbi_energy_error, label="DBI avg energy error")
    # plt.legend()
    # plt.show()

    from plot import plot_result
    plot_result(nsites_range, init_fid, init_energy_error, dbi_energy_error)