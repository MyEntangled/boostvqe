from boostvqe.src.boostvqe.new_code.vqe.vqe_ansatz import double_ladder_ansatz

import quimb.tensor as qtn
import quimb as qu
import numpy as np
import scipy
from functools import reduce

def multi_pauli(op_string):
    if len(op_string)==1:
        return qu.pauli(op_string)
    # Generate the Kronecker product for the Pauli string
    paulis = {'I': qu.eye(2), 'X': qu.pauli('X'), 'Y': qu.pauli('Y'), 'Z': qu.pauli('Z')}
    return reduce(qu.kron, (paulis[op] for op in op_string))

def evaluate(circuit, H):
    coeffs, paulis, sites = H
    energy = 0
    for c, p, s in zip(coeffs, paulis, sites):
        energy += c * circuit.local_expectation(multi_pauli(p), s).real

    return energy

def cost_function(params, vqe_ansatz, n_sites:int, layers:int, H):
    circuit = vqe_ansatz(n_sites, layers, params)

    return evaluate(circuit, H)

def train_vqe(vqe_ansatz, n_sites, layers, H):

    # Initialize parameters randomly
    n_params = layers * (1 * n_sites)
    initial_params = np.random.normal(0,10,size=n_params)

    # Perform the optimization
    result = scipy.optimize.minimize(cost_function, initial_params, args=(vqe_ansatz, n_sites, layers, H), method='COBYLA')

    # # Extract the optimal parameters and energy
    optimal_params = result.x
    optimal_energy = result.fun

    return optimal_params, optimal_energy


if __name__ == '__main__':
    from hamiltonian import build_xxz_hamiltonian
    import numpy as np
    import scipy

    n_sites = 4  # Number of sites in the chain
    Jx = 1  # Coupling in the x-direction
    Jy = 1  # Coupling in the y-direction
    Jz = 0.6  # Coupling in the z-direction
    h = +0.4  # Transverse field strength
    couplings = np.array([Jx, Jy, Jz, h]) / n_sites

    # Build the MPO Hamiltonian
    hamiltonian, H = build_xxz_hamiltonian(n_sites, couplings)

    dmrg = qtn.DMRG2(hamiltonian, bond_dims=[8, 16, 32, 64, 128], cutoffs=1e-8)
    dmrg.solve(max_sweeps=5, verbosity=1, cutoffs=1e-8)

    dmrg_energy = dmrg.energies[-1]

    optimal_params, optimal_energy = train_vqe(double_ladder_ansatz, n_sites, 3, H)

    print(f"Exact ground state energy: {dmrg_energy:.6f}")
    print(f"Optimal ground state energy: {optimal_energy:.6f}")
