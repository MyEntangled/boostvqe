import argparse
import logging
import pathlib

import numpy as np
import qibo
from qibo import hamiltonians
from qibo.models.variational import VQE

from ansatze import build_circuit
from plotscripts import plot_results
from utils import (
    FLUCTUATION_FILE,
    LOSS_FILE,
    HAMILTONIAN_FILE,
    create_folder,
    generate_path,
    results_dump,
)

logging.basicConfig(level=logging.INFO)
SEED = 42
TOL = 1e-2


def loss(params, circuit, hamiltonian):
    circuit.set_parameters(params)
    result = hamiltonian.backend.execute_circuit(circuit)
    final_state = result.state()
    return hamiltonian.expectation(final_state), hamiltonian.energy_fluctuation(
        final_state
    )


def main(args):
    """VQE training."""
    # set backend and number of classical threads
    qibo.set_backend(backend=args.backend, platform=args.platform)
    qibo.set_threads(args.nthreads)

    # setup the results folder
    logging.info("Set VQE")
    path = create_folder(generate_path(args))

    # build hamiltonian and variational quantum circuit
    ham = hamiltonians.XXZ(nqubits=args.nqubits)
    circ = build_circuit(nqubits=args.nqubits, nlayers=args.nlayers)

    # print the circuit
    logging.info("\n" + circ.draw())
    nparams = len(circ.get_parameters())

    # initialize VQE
    params_history = []
    loss_list = []
    fluctuations = []
    vqe = VQE(circuit=circ, hamiltonian=ham)

    def callbacks(
        params,
        vqe=vqe,
        loss_list=loss_list,
        loss_fluctuation=fluctuations,
        params_history=params_history,
    ):
        """
        Callback function that updates the energy, the energy fluctuations and
        the parameters lists.
        """
        energy, energy_fluctuation = loss(params, vqe.circuit, vqe.hamiltonian)
        loss_list.append(energy)
        loss_fluctuation.append(energy_fluctuation)
        params_history.append(params)

    # fix numpy seed to ensure replicability of the experiment
    logging.info("Minimize the energy")
    np.random.seed(SEED)
    initial_parameters = np.random.randn(nparams)
    results = vqe.minimize(
        initial_parameters,
        method=args.optimizer,
        callback=callbacks,
        tol=TOL,
    )
    opt_results = results[2]
    # save final results
    output_dict = {
        "nqubits": args.nqubits,
        "nlayers": args.nlayers,
        "optimizer": args.optimizer,
        "best_loss": float(opt_results.fun),
        "true_ground_energy": min(ham.eigenvalues()),
        "success": opt_results.success,
        "message": opt_results.message,
        "backend": args.backend,
        "platform": args.platform,
    }
    np.save(file=f"{path}/{LOSS_FILE}", arr=loss_list)
    np.save(file=f"{path}/{FLUCTUATION_FILE}", arr=fluctuations)
    np.save(file=f"{path}/{HAMILTONIAN_FILE}", arr=ham.matrix)

    logging.info("Dump the results")
    results_dump(path, params_history, output_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQE training hyper-parameters.")
    parser.add_argument("--nqubits", default=6, type=int)
    parser.add_argument("--nlayers", default=5, type=int)
    parser.add_argument("--optimizer", default="Powell", type=str)
    parser.add_argument("--output_folder", default=None, type=str)
    parser.add_argument("--backend", default="qibojit", type=str)
    parser.add_argument("--platform", default="dummy", type=str)
    parser.add_argument("--nthreads", default=1, type=int)

    args = parser.parse_args()
    main(args)
    path = generate_path(args)
    plot_results(pathlib.Path(path))
