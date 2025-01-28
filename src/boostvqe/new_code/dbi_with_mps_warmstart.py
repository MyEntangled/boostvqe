from boostvqe.src.boostvqe.new_code.dbi.helper import invert_circuit
from hamiltonian import build_xxz_hamiltonian
from dbi import train_dbi
from vqe.vqe_training import evaluate, train_vqe
from vqe.vqe_ansatz import double_ladder_ansatz
from boostvqe.src.boostvqe.new_code.mps_warmstart import create_mps_circuit, mps_to_unitaries

import numpy as np
import quimb.tensor as qtn

import tqdm

def evaluate_dbi(warmstart_circuit, couplings, H):
    dbi_best = 0

    for s in tqdm.tqdm(np.logspace(-2, 0, 200)):
        psi = train_dbi.dbi_circuit(warmstart_circuit, s, couplings)
        energy = evaluate(psi, H)
        dbi_best = min(dbi_best, energy)

    return dbi_best

# Define parameters for the XXZ Hamiltonian
n_sites = 100  # Number of sites in the chain
Jx = 1  # Coupling in the x-direction
Jy = 0  # Coupling in the y-direction
Jz = 0 # Coupling in the z-direction
h = +0.5  # Transverse field strength
couplings = np.array([Jx, Jy, Jz, h]) / n_sites

# Build the Hamiltonian
hamiltonian, H = build_xxz_hamiltonian(n_sites, couplings)
#hamiltonian = qtn.MPO_ham_heis(n_sites)

# DMRG
dmrg = qtn.DMRG2(hamiltonian, bond_dims=[16, 32, 64, 128], cutoffs=1e-6)
res = dmrg.solve(verbosity=0)
gs = dmrg.state

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
circuit_unitaries = mps_to_unitaries.disentangling_gates(input_mps=gs, num_layers=2, hamiltonian=hamiltonian)
circuit_gates = list(create_mps_circuit.create_circuit_from_gate_unitaries(circuit_unitaries))

warmstart_mps = qtn.CircuitMPS(n_sites)
warmstart_mps.apply_gates(circuit_gates)
mps_energy_contract = (warmstart_mps.psi.conj().reindex_(
    {f'k{n}': f'b{n}' for n in range(warmstart_mps.psi.num_tensors)}) | hamiltonian | warmstart_mps.psi) ^ ...
mps_energy = evaluate(warmstart_mps, H)
print('MPS energy:', mps_energy_contract, mps_energy)
print()


# Evaluate the DBI
#dbi_vqe = evaluate_dbi(warmstart_vqe, couplings, H)
dbi_mps = evaluate_dbi(warmstart_mps, couplings, H)

print('True gs:', dmrg.energy)
# print('VQE, VQE+DBI:', vqe_energy, dbi_vqe)
print('MPS, MPS+DBI:', mps_energy, dbi_mps)
# print(warmstart_vqe.psi)
print(warmstart_mps.gates)
# print(hamiltonian)