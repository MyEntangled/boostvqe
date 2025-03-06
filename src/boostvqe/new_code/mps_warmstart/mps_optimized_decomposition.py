from src.boostvqe.new_code.mps_warmstart.compile_gates import unitary_to_gates
from src.boostvqe.new_code.mps_warmstart.helper import flatten_list, apply_circuit_mps, is_unitary
from src.boostvqe.new_code.mps_warmstart.outerprod_ptrace import outer_product_mps, partial_trace_mpo
from src.boostvqe.new_code.mps_warmstart.create_mps_circuit import generate_gates_from_unitaries

import numpy as np
import quimb.tensor as qtn
import scipy

def overall_fidelity_mps(unitaries, qargs, psi_target, n):
    """
    Compute the fidelity f = |<psi_target | psi_approx>|,
    where psi_approx is obtained by applying the circuit gates to |0...0>.
    """
    circuit_gates = generate_gates_from_unitaries(unitaries, qargs)
    circ = apply_circuit_mps(n, circuit_gates)
    psi_approx = circ.psi

    return np.abs(psi_target.H @ psi_approx)

def optimized_decomposition(psi_target:qtn.MatrixProductState | qtn.CircuitMPS, unitaries, qargs, opt_sweeps=100, fid_target=0.99, r=0.6):
    """
    Optimization loop (Algorithm 2). For a circuit of M gates (acting on n=M+1 qubits),
    we update each gate to locally maximize the fidelity with psi_target.

    The update for gate m is done as:
      U_m ← U_m · (U_m† U_new)^(r)
    where U_new is obtained from the SVD of the 4×4 environment tensor Fₘ.

    The overall circuit fidelity is f = |⟨psi_target|(U_M ... U_0)|0^n⟩|.
    """
    assert len(unitaries) == len(qargs)
    M = len(unitaries)

    if isinstance(psi_target, qtn.MatrixProductState):
        n = psi_target.num_tensors
        psi_target_circ = qtn.CircuitMPS(N=n, psi0=psi_target)
    else:
        n = psi_target.N
        psi_target_circ = psi_target.copy()

    fid = overall_fidelity_mps(unitaries, qargs, psi_target_circ.psi, n)
    #print("Initial fidelity:", fid)

    gate_list = [list(generate_gates_from_unitaries([unitaries[i]], [qargs[i]])) for i in range(M)]
    t = 0

    print('Time complexity', opt_sweeps * M)

    while t < opt_sweeps and fid < fid_target:
        head_circ = qtn.CircuitMPS(n)  # to apply U_{m-1} ... U_1 U_0|0^n>
        tail_circ = psi_target_circ.copy()  # to apply U^dg_{m+1}... U^dg_{M-2} U^dg_{M-1}|psi_target>
        tail_circ = apply_circuit_mps(tail_circ, flatten_list(gate_list), apply_inverse=True)

        for m in range(M):
            # Compute left state: U'_{m-1} U_{m-2}... U_1 U_0|0^n> = U'_{m-1} |prev. left state>.
            if m > 0:
                head_circ = apply_circuit_mps(head_circ, flatten_list([gate_list[m - 1]]))
                #head_circ = apply_circuit_mps(head_circ, flatten_list(gate_list[:m]))

            # Compute right state: U^dg_{m+1}... U^dg_{M-2} U^dg_{M-1}|psi_target> = U_{m} U^dg_m U^dg_{m+1}... U^dg_{M-2} U^dg_{M-1}|psi_target>
            # = U_{m} |prev. right state>
            tail_circ = apply_circuit_mps(tail_circ, gate_list[m])

            # Form the operator O = |right_state><left_state|
            outer_product_tensor = outer_product_mps(tail_circ.psi, head_circ.psi.H)

            # Compute the environment tensor F_m by tracing out all qubits except those affected by the unitary
            F_m = partial_trace_mpo(outer_product_tensor, keep_sites=qargs[m])

            #print(F_m)
            if len(F_m.shape) == 2: # 1-qubit unitary
                F_m = F_m.data
            elif len(F_m.shape) == 4:
                # (1, 3, 0, 2): checked
                # (0, 2, 1, 3): checked
                # (3,1,2,0): checked
                # (2,0,3,1): checked --> works
                F_m = F_m.data.transpose(2,0,3,1).reshape(4,4) # 2-qubit unitary
            else:
                raise Exception('Only 1-qubit and 2-qubit gates are allowed.')

            # SVD of F_m
            U_svd, s, Vh = np.linalg.svd(F_m)
            U_new = U_svd @ Vh

            # Update U_m
            U_current = unitaries[m]
            X = U_current.conj().T @ U_new
            X_r = scipy.linalg.fractional_matrix_power(X, r)
            U_updated = U_current @ X_r
            #assert is_unitary(U_updated)

            # Reunitarize via SVD decomposition (U_updated can be slightly non-unitary due to numerical errors).
            U, _, Vh = np.linalg.svd(U_updated)
            Q = U @ Vh
            assert is_unitary(Q)

            # Update the m-th unitary, but qargs unchanged.
            unitaries[m] = Q
            gate_list[m] = list(generate_gates_from_unitaries([Q], [qargs[m]]))

            #print(overall_fidelity_mps(unitaries, qargs, psi_target_circ.psi, n))

        fid = overall_fidelity_mps(unitaries, qargs, psi_target_circ.psi, n)
        #print(f"Sweep {t + 1}: Fidelity = {fid:.6f}")
        t += 1
    return unitaries, qargs, fid


if __name__ == '__main__':
    np.random.seed(42)
    n = 50
    M = 200

    # Create a "true" circuit to generate the target state.
    true_unitaries = []
    true_qargs = []
    for m in range(M):
        # Generate a random 4x4 unitary via QR decomposition.
        X = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        Q, _ = np.linalg.qr(X)
        true_unitaries.append(Q)
        true_qargs.append((m % n, (m + 1) % n ))

    gate_list = [list(generate_gates_from_unitaries([true_unitaries[i]], [true_qargs[i]])) for i in range(M)]
    print('apply...')
    circ = apply_circuit_mps(n, flatten_list(gate_list))
    print(len(circ.gates))
    print(circ.gates)

    # Create an initial guess circuit.
    guess_unitaries = []
    guess_qargs = []
    for m in range(M):
        X = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        Q, _ = np.linalg.qr(X)
        assert is_unitary(Q)
        guess_unitaries.append(Q)
        guess_qargs.append((m % n, (m + 1) % n))

    print("Initial guess fidelity:", overall_fidelity_mps(guess_unitaries, guess_qargs, circ.psi, n))
    # Optimize the circuit.
    optimized_unitaries, qargs, final_fid = optimized_decomposition(circ, guess_unitaries, guess_qargs, opt_sweeps=10, fid_target=0.999, r=0.6)
    print("Final fidelity:", final_fid)
