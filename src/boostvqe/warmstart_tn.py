import qibo
from qibo import gates
from qibo.models import Circuit


import quimb as qu
import quimb.tensor as qtn

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation

import numpy as np

import pennylane as qml
from functools import partial


def mps_circuit(nqubits:int, bond_dim:int=16):
    # create a random MPS as our initial target to optimize
    psi = qtn.MPS_rand_state(nqubits, bond_dim=bond_dim, cyclic=False)

    # create the hamiltonian MPO, this is a constant TN not to be optimized.
    # second line is the equivalent qibo hamiltonian
    ham_tn = qtn.MPO_ham_ising(nqubits, j=-4.0, bx=2.0, cyclic=False)
    ham = qibo.hamiltonians.TFIM(nqubits, h=1, dense=False)

    # create a TN optimizer
    tnopt = qtn.TNOptimizer(
        psi,
        loss_fn=loss_fn,
        norm_fn=norm_fn,
        loss_constants={"ham": ham_tn},
        optimizer="adam",
        autodiff_backend="jax",
    )

    psi_opt = tnopt.optimize(1000)
    statevector = (psi_opt ^ ...).data
    #print(statevector)
    statevector = statevector.reshape(-1)
    statevector /= np.sqrt(np.sum(np.abs(statevector)**2))
    print(statevector.shape)
    print(np.linalg.norm(statevector))
    #print(statevector)


    gate_list = []
    for op in qml.MottonenStatePreparation(statevector, wires=range(nqubits)).decomposition():
        if op.name != 'GlobalPhase':
            gate_list.append(qtn.Gate(op.name, params=op.data, qubits=op.wires.labels))

    print(len(gate_list))
    circuit = qtn.CircuitMPS.from_gates(
        gates=gate_list,
        max_bond=None,
        cutoff=1e-6,
        progbar=True,
    )
    #print((circuit.psi ^ ...).data)
    return circuit


def norm_fn(psi):
    nfact = (psi.H @ psi)**(1/2)
    return psi.multiply(1 / nfact, spread_over='all')


def loss_fn(psi, ham):
    b, h, k = qtn.tensor_network_align(psi.H, ham, psi)
    energy_tn = b | h | k
    return energy_tn ^ ...




if __name__ == '__main__':
    print('start')
    nqubits = 14
    bond_dim = 16
    circ = mps_circuit(nqubits, bond_dim=bond_dim)
    print(circ.draw())