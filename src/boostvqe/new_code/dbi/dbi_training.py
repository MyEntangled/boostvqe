from jax.example_libraries.optimizers import optimizer

from src.boostvqe.new_code.vqe.vqe_training import train_vqe
from src.boostvqe.new_code.dbi.helper import append_circuits, invert_circuit

import quimb.tensor as qtn
import numpy as np
import scipy
from cmaes import CMA

import jax
import jax.numpy as jnp
import optax

import cotengra as ctg

jax.config.update("jax_traceback_filtering", "off")


def find_contraction_path(circ:qtn.CircuitMPS, hamiltonian):
    opt = ctg.ReusableHyperOptimizer(
        max_repeats=16,
        reconf_opts={},
        parallel=False,
        progbar=True,
        #     directory=True,  # if you want a persistent path cache
    )
    tn = (circ.psi.conj().reindex_({f'k{n}': f'b{n}' for n in range(circ.psi.num_tensors)}) | hamiltonian | circ.psi)
    tree = tn.contraction_tree(opt)

    return tree


def train_dbi(couplings, hamiltonian, warmstart):
    n_sites = warmstart.N

    # Initialize parameters randomly
    params = np.zeros(2*n_sites)
    params[0] = 0.01  ## evolution time s
    params[1:] = np.random.uniform(0., 2*np.pi, 2 * n_sites - 1)


    # Get a good contraction path (for reuse)
    dbi_circ = dbi_circuit(warmstart, couplings, params[0], params[1:])
    tree = find_contraction_path(dbi_circ, hamiltonian)


    # Perform the optimization  ### COBYLA optimization for gradient-estimate opt
    result = scipy.optimize.minimize(dbi_cost_function, params, args=(couplings, hamiltonian, warmstart, tree), method='COBYLA')
    # Extract the optimal parameters and energy
    optimal_params = result.x
    optimal_energy = result.fun


    # # Perform the optimization  ### CMA-ES optimization
    # optimizer = CMA(mean=np.zeros(shape=len(params)), sigma=1, lr_adapt=True)
    # for generation in range(200):
    #     solutions = []
    #     for _ in range(optimizer.population_size):
    #         x = optimizer.ask()
    #         value = dbi_cost_function(x, couplings, hamiltonian, warmstart)
    #         solutions.append((x, value))
    #     x_mean = optimizer.mean
    #     val_mean = dbi_cost_function(x_mean, couplings, hamiltonian, warmstart)
    #     print(f"#{generation} MEAN {val_mean} (x={x_mean})")
    #     optimizer.tell(solutions)
    #     if optimizer.should_stop():
    #         break
    #
    # optimal_params = optimizer.mean
    # optimal_energy = dbi_cost_function(optimal_params, couplings, hamiltonian, warmstart)

    return optimal_params, optimal_energy


def dbi_cost_function(params, couplings , hamiltonian, warmstart, contract_tree=None):
    #circuit = vqe_ansatz(n_sites, layers, params)
    s, params_d = params[0], params[1:]
    circuit = dbi_circuit(warmstart, couplings, s, params_d)

    ## Use contraction to compute energy (Can either use local expectation, see vqe_training/evaluate)
    energy = (circuit.psi.conj().reindex_({f'k{n}': f'b{n}' for n in range(circuit.psi.num_tensors)}) | hamiltonian | circuit.psi).contract(all, optimize=contract_tree)
    #print(f"Energy: {energy}")
    return energy


def dbi_circuit(warmstart_circuit, couplings, s, params_d):
    n_sites = warmstart_circuit.N
    warmstart_circuit_inv = invert_circuit(warmstart_circuit)

    circuit = qtn.CircuitMPS(n_sites, max_bond=2*11, cutoff=1e-8)
    circuit = D(circuit, params_d)

    circuit = append_circuits(circuit, warmstart_circuit)
    circuit = Trotter(circuit, couplings, s, nsteps=4)
    circuit = append_circuits(circuit, warmstart_circuit_inv)

    circuit = D(circuit, -params_d)

    circuit = append_circuits(circuit, warmstart_circuit)

    return circuit


def D(circuit, params):
    n_sites = circuit.N
    idx = 0
    for i in range(n_sites):
        circuit.apply_gate('RZ', 2 * params[idx], i)
        idx += 1

    for i in range(0, n_sites - 1, 2):
        circuit.apply_gate('RZZ', 2 * params[idx], i, i + 1)
        idx += 1

    for i in range(1, n_sites - 1, 2):
        circuit.apply_gate('RZZ', 2 * params[idx], i, i + 1)
        idx += 1

    return circuit


def Trotter(circuit, couplings, t, nsteps=2):
    # To do: implement second order
    n_sites = circuit.N

    Jx, Jy, Jz, h = couplings
    dt = t / nsteps  # time step for each Trotter step

    # Apply Trotter steps
    for _ in range(nsteps):
        # Apply magnetic field (RZ gates for each site)
        for i in range(n_sites):
            if abs(h) > 0:
                circuit.apply_gate('RZ', 2 * h * dt, i)

        # Apply interaction terms (RXX, RYY, RZZ gates for coupling terms)
        for i in range(0, n_sites - 1, 2):  # even-indexed sites
            if abs(Jz) > 0:
                circuit.apply_gate('RZZ', 2 * Jz * dt, i, i + 1)
            if abs(Jx) > 0:
                circuit.apply_gate('RXX', 2 * Jx * dt, i, i + 1)
            if abs(Jy) > 0:
                circuit.apply_gate('RYY', 2 * Jy * dt, i, i + 1)

        for i in range(1, n_sites - 1, 2):  # odd-indexed sites
            if abs(Jz) > 0:
                circuit.apply_gate('RZZ', 2 * Jz * dt, i, i + 1)
            if abs(Jx) > 0:
                circuit.apply_gate('RXX', 2 * Jx * dt, i, i + 1)
            if abs(Jy) > 0:
                circuit.apply_gate('RYY', 2 * Jy * dt, i, i + 1)

    return circuit


if __name__ == '__main__':
    from src.boostvqe.new_code.vqe.vqe_ansatz import double_ladder_ansatz
    from src.boostvqe.new_code.hamiltonian import build_xxz_hamiltonian
    import time

    start = time.time()

    # Define parameters for the XXZ Hamiltonian
    n_sites = 4  # Number of sites in the chain
    Jx = 1  # Coupling in the x-direction
    Jy = 1  # Coupling in the y-direction
    Jz = 0.6  # Coupling in the z-direction
    h = +0.4  # Transverse field strength
    couplings = np.array([Jx, Jy, Jz, h]) / n_sites

    # Build the Hamiltonian
    hamiltonian, H = build_xxz_hamiltonian(n_sites, couplings)

    # Train the VQE
    layers = 4
    optimal_params, optimal_energy = train_vqe(double_ladder_ansatz, n_sites, layers, H)

    circuit_vqe = double_ladder_ansatz(n_sites, layers, optimal_params)

    # # Evaluate the DBI
    # import tqdm
    #
    # dbi_best = 0
    #
    # for s in tqdm.tqdm(np.logspace(-2, 0, 200)):
    #     psi = dbi_circuit(circuit_vqe, s, couplings)
    #     energy = evaluate(psi, H)
    #     dbi_best = min(dbi_best, energy)
    #
    # print(optimal_energy, dbi_best)

    dbi_params, dbi_energy = train_dbi(couplings, hamiltonian, circuit_vqe)
    print(dbi_energy)

    print("Time taken:", time.time() - start)