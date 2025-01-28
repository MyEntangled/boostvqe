from boostvqe.src.boostvqe.new_code.vqe.vqe_training import train_vqe, evaluate
from boostvqe.src.boostvqe.new_code.vqe.vqe_ansatz import double_ladder_ansatz
from .helper import append_circuits, invert_circuit
from boostvqe.src.boostvqe.new_code.hamiltonian import build_xxz_hamiltonian
import quimb.tensor as qtn
import numpy as np

def dbi_circuit(warmstart_circuit, s, couplings):
    n_sites = warmstart_circuit.N
    warmstart_circuit_inv = invert_circuit(warmstart_circuit)

    params_d = np.random.uniform(0., 2 * np.pi, 2 * n_sites - 1)

    circuit = qtn.CircuitMPS(n_sites, max_bond=1024, cutoff=10 ** -8)
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

    # Evaluate the DBI
    import tqdm

    dbi_best = 0

    for s in tqdm.tqdm(np.logspace(-2, 0, 200)):
        psi = dbi_circuit(circuit_vqe, s, couplings)
        energy = evaluate(psi, H)
        dbi_best = min(dbi_best, energy)

    print(optimal_energy, dbi_best)