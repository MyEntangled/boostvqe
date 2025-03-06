import qiskit.quantum_info
from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import state_fidelity, Statevector

import numpy as np
import scipy

def is_unitary(U):
    #print(np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0])))
    return np.allclose(U.conj().T @ U, np.eye(U.shape[0]))

def partial_trace(rho, keep, dims, optimize=False):
    """Calculate the partial trace

    ρ_a = Tr_b(ρ)

    Parameters
    ----------
    ρ : 2D array
        Matrix to trace
    keep : array
        An array of indices of the spaces to keep after
        being traced. For instance, if the space is
        A x B x C x D and we want to trace out B and D,
        keep = [0,2]
    dims : array
        An array of the dimensions of each space.
        For instance, if the space is A x B x C x D,
        dims = [dim_A, dim_B, dim_C, dim_D]

    Returns
    -------
    ρ_a : 2D array
        Traced matrix
    """
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = [i for i in range(Ndim)]
    idx2 = [Ndim+i if i in keep else i for i in range(Ndim)]
    rho_a = rho.reshape(np.tile(dims,2))
    rho_a = np.einsum(rho_a, idx1+idx2, optimize=optimize)
    return rho_a.reshape(Nkeep, Nkeep)

def ptrace_(rho, dims, axis):
    """
    Takes partial trace over the subsystem defined by 'axis'
    rho: a matrix
    dims: a list containing the dimension of each subsystem
    axis: the index of the subsytem to be traced out
    (We assume that each subsystem is square)
    """
    #axis = [i for i in range(len(dims)) if i not in keeps]

    dims_ = np.array(dims)
    # Reshape the matrix into a tensor with the following shape:
    # [dim_0, dim_1, ..., dim_n, dim_0, dim_1, ..., dim_n]
    # Each subsystem gets one index for its row and another one for its column
    reshaped_rho = rho.reshape(np.concatenate((dims_, dims_), axis=None))

    # Move the subsystems to be traced towards the end
    reshaped_rho = np.moveaxis(reshaped_rho, axis, -1)
    reshaped_rho = np.moveaxis(reshaped_rho, len(dims)+axis-1, -1)

    # Trace over the very last row and column indices
    traced_out_rho = np.trace(reshaped_rho, axis1=-2, axis2=-1)

    # traced_out_rho is still in the shape of a tensor
    # Reshape back to a matrix
    dims_untraced = np.delete(dims_, axis)
    rho_dim = np.prod(dims_untraced)
    return traced_out_rho.reshape([rho_dim, rho_dim])

def partial_trace_new(rho, dims, keeps):
    """
    Takes partial trace over the subsystems defined by 'keeps'
    rho: a matrix
    dims: a list containing the dimension of each subsystem
    keeps: a list of indices of the subsystems to keep
    (We assume that each subsystem is square)
    """
    axis_to_trace = sorted([i for i in range(len(dims)) if i not in keeps])
    print("Axis to trace:", axis_to_trace, "Axis to keep:", list(keeps))
    traced_rho = rho.copy()
    current_dims = dims.copy()

    for k in range(len(axis_to_trace)):
        axis = axis_to_trace[k] - k
        traced_rho = ptrace_(traced_rho, current_dims, axis)
        current_dims = current_dims[:axis] + current_dims[axis + 1:]

    return traced_rho


# def fractional_matrix_power(X, r=None):
#     """
#     Compute X^r via eigendecomposition for unitary X.
#     """
#     if r is None:
#         return X
#
#     eigvals, P = np.linalg.eig(X)
#     D_r = np.diag(eigvals ** r)
#
#     #return P @ D_r @ np.linalg.inv(P)
#     return P @ D_r @ P.conj().T

def apply_circuit(circuit: int | QuantumCircuit, operators, qargs, apply_inverse=False):
    """
    Apply a list of two-qubit gates (each a tuple (U, (q0, q1)))
    using a QuantumCircuit with n qubits.
    """
    assert len(operators) == len(qargs)

    if isinstance(circuit, int):
        circuit = QuantumCircuit(circuit)

    if not apply_inverse:
        for i in range(len(operators)):
            #print(operators[i], qargs[i])
            circuit.unitary(operators[i], qubits=qargs[i])

    else: ## Apply the inverse of the gates (order also reversed)
        #print("Length", len(operators))
        for i in reversed(range(len(operators))):
            circuit.unitary(operators[i].transpose().conjugate(), qubits=qargs[i])
            #print(operators[i].transpose().conjugate())

    return circuit

def overall_fidelity(unitaries_with_qargs, psi_target, n):
    """
    Compute the fidelity of a circuit with a target state.
    """
    unitaries, qargs = zip(*unitaries_with_qargs)
    op_list = [Operator(unitaries[i]) for i in range(len(unitaries))]

    for i in range(len(unitaries)):
        assert is_unitary(unitaries[i]), f"Operator {i} is not unitary"
        #assert op_list[i].is_unitary()

    circ = apply_circuit(n, op_list, qargs)
    psi = Statevector(circ)
    #print(psi, psi_target)
    return np.sqrt(state_fidelity(psi, psi_target))

def optimize_circuit(unitaries_with_qargs, psi_target_circ:QuantumCircuit, T=100, f_target=0.99, r=0.6):
    """
    Optimization loop (Algorithm 2). For a circuit of M gates (acting on n=M+1 qubits),
    we update each gate to locally maximize the fidelity with psi_target.

    The update for gate m is done as:
      U_m ← U_m · (U_m† U_new)^(r)
    where U_new is obtained from the SVD of the 4×4 environment tensor Fₘ.

    The overall circuit fidelity is f = |⟨psi_target|(U_M ... U_0)|0^n⟩|.
    """
    M = len(unitaries_with_qargs)
    n = psi_target_circ.num_qubits
    fid = overall_fidelity(unitaries_with_qargs, Statevector(psi_target_circ), n)
    print("Initial fidelity:", fid)

    unitaries, qargs = zip(*unitaries_with_qargs)
    op_list = [Operator(unitaries[i]) for i in range(M)]
    t = 0

    while t < T and fid < f_target:

        head_circ = QuantumCircuit(n)  # to apply U_{m-1} ... U_1 U_0|0^n>
        tail_circ = psi_target_circ.copy()  # to apply U^dg_{m+1}... U^dg_{M-2} U^dg_{M-1}|psi_target>
        tail_circ = apply_circuit(tail_circ, op_list, qargs, apply_inverse=True)

        for m in range(M):
            print(f"Before op {m}:", overall_fidelity(unitaries_with_qargs, Statevector(psi_target_circ), n))
            # print(head_circ)
            # print(tail_circ)

            # Compute left state: U'_{m-1} U_{m-2}... U_1 U_0|0^n> = U'_{m-1} |prev. left state>.
            if m > 0:
                head_circ = apply_circuit(head_circ, [op_list[m-1]], [qargs[m-1]])
                #head_circ = apply_circuit(n, op_list[:m], qargs[:m]) ## Inefficient

            # Compute right state: U^dg_{m+1}... U^dg_{M-2} U^dg_{M-1}|psi_target> = U_{m} U^dg_m U^dg_{m+1}... U^dg_{M-2} U^dg_{M-1}|psi_target>
            # = U_{m} |prev. right state>
            tail_circ = apply_circuit(tail_circ, [op_list[m]], [qargs[m]])
            #tail_circ = psi_target_circ.copy()
            #tail_circ = apply_circuit(tail_circ, op_list[m+1:], qargs[m+1:], apply_inverse=True)

            # Form the operator O = |right_state><left_state|
            outer_product = np.outer(Statevector(tail_circ).data, Statevector(head_circ).data.conj())

            # Compute the 4x4 environment tensor F_m by tracing out all qubits except m and m+1.
            to_keep = [n-1 - qargs[m][i] for i in range(len(qargs[m]))] ## n-1 - index to follow little-endian notation in qiskit
            F_m = partial_trace_new(outer_product, dims=[2]*n, keeps=to_keep)
            #print(F_m)

            # SVD of F_m
            U_svd, s, Vh = scipy.linalg.svd(F_m)
            U_new = U_svd @ Vh

            # Update U_m
            U_current = unitaries_with_qargs[m][0]
            X = U_current.conj().T @ U_new
            X_r = scipy.linalg.fractional_matrix_power(X, r)
            U_updated = U_current @ X_r

            # Reunitarize via SVD decomposition (U_updated can be slightly non-unitary due to numerical errors).
            Q, _ = np.linalg.qr(U_updated)
            U, _, Vh = scipy.linalg.svd(U_updated)
            Q = U @ Vh

            assert np.linalg.norm(U_updated - Q) < 1e-10
            assert is_unitary(Q)

            # Update the m-th gate, keeping its qubit indices unchanged.
            unitaries_with_qargs[m] = (Q, unitaries_with_qargs[m][1])
            op_list[m] = Operator(Q)

            print(f"After op {m}:", overall_fidelity(unitaries_with_qargs, Statevector(psi_target_circ), n))
            print('--')

        fid = overall_fidelity(unitaries_with_qargs, Statevector(psi_target_circ), n)
        print(f"Sweep {t + 1}: Fidelity = {fid:.6f}")
        t += 1
    return unitaries_with_qargs, fid


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    M = 4  # number of two-qubit gates; total qubits n = M+1
    n = M + 1

    # Create a "true" circuit to generate the target state.
    true_unitaries = []
    for m in range(M):
        # Generate a random 4x4 unitary via QR decomposition.
        # X = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        # Q, _ = np.linalg.qr(X)
        Q = qiskit.quantum_info.random_unitary(4, seed=seed).data
        assert is_unitary(Q)
        true_unitaries.append((Q, (m, m + 1)))

    unitaries, qargs = zip(*true_unitaries)
    op_list = [Operator(unitaries[i]) for i in range(M)]
    circ = apply_circuit(n, op_list, qargs)

    # zero_state = Statevector.from_int(0, 2**n)
    # disentangled_state = circ.copy()
    # apply_circuit(disentangled_state, op_list, qargs, apply_inverse=True)
    # print("Disentangled fidelity:", state_fidelity(zero_state, Statevector(disentangled_state)))


    # Create an initial guess circuit.
    guess_unitaries = []
    for m in range(M):
    #     X = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    #     Q, _ = np.linalg.qr(X)
        Q = qiskit.quantum_info.random_unitary(4, seed=seed+2).data
        assert is_unitary(Q)
        guess_unitaries.append((Q, (m, m + 1)))


    print("Initial guess fidelity:", overall_fidelity(guess_unitaries, Statevector(circ), n))
    # Optimize the circuit.
    optimized_gates, final_fid = optimize_circuit(guess_unitaries.copy(), circ, T=10, f_target=0.999, r=0.6)
    print("Final fidelity:", final_fid)
