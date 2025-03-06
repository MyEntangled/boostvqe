import numpy as np
import qiskit.quantum_info

from src.boostvqe.new_code.mps_warmstart.helper import is_unitary


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

def partial_trace(rho, dims, keeps):
    """
    Takes partial trace over the subsystems defined by 'keeps'
    rho: a matrix
    dims: a list containing the dimension of each subsystem
    keeps: a list of indices of the subsystems to keep
    (We assume that each subsystem is square)
    """
    axis_to_trace = sorted([i for i in range(len(dims)) if i not in keeps])
    print("Axis to trace", axis_to_trace)
    traced_rho = rho.copy()
    current_dims = dims.copy()

    for k in range(len(axis_to_trace)):
        axis = axis_to_trace[k] - k
        traced_rho = ptrace_(traced_rho, current_dims, axis)
        current_dims = current_dims[:axis] + current_dims[axis + 1:]

    return traced_rho


"""
Test out the partial_trace numpy module by creating a matrix
rho_ABC = rho_A \otimes rho_B \otimes rho_C
Each rho_i is normalized, i.e. Tr(rho_i) = 1
"""

# Generate the results we want
rho_A = np.random.rand(4, 4) + 1j*np.random.rand(4, 4)
rho_A /= np.trace(rho_A)
rho_B = np.random.rand(3, 3) + 1j*np.random.rand(3, 3)
rho_B /= np.trace(rho_B)
rho_C = np.random.rand(2, 2) + 1j*np.random.rand(2, 2)
rho_C /= np.trace(rho_C)

rho_AB = np.kron(rho_A, rho_B)
rho_BC = np.kron(rho_B, rho_C)
rho_AC = np.kron(rho_A, rho_C)
rho_ABC = np.kron(rho_AB, rho_C)

# Try to get the results by doing partial_trace's from rho_ABC
# rho_AB_test = ptrace_(rho_ABC, [4, 3, 2], axis=2)
# rho_AC_test = ptrace_(rho_ABC, [4, 3, 2], axis=1)
# rho_A_test = ptrace_(rho_AB_test, [4, 3], axis=1)
# rho_B_test = ptrace_(rho_AB_test, [4, 3], axis=0)
# rho_C_test = ptrace_(rho_AC_test, [4, 2], axis=0)

# rho_AB_test = partial_trace(rho_ABC, [4, 3, 2], keeps=[0,1])
# rho_AC_test = partial_trace(rho_ABC, [4, 3, 2], keeps=[0,2])
# rho_A_test = partial_trace(rho_AB_test, [4, 3], keeps=[0])
# rho_B_test = partial_trace(rho_AB_test, [4, 3], keeps=[1])
# rho_C_test = partial_trace(rho_AC_test, [4, 2], keeps=[1])

rho_AB_test = partial_trace(rho_ABC, [4, 3, 2], keeps=[0,1])
rho_AC_test = partial_trace(rho_ABC, [4, 3, 2], keeps=[0,2])
rho_A_test = partial_trace(rho_ABC, [4, 3, 2], keeps=[0])
rho_B_test = partial_trace(rho_ABC, [4, 3, 2], keeps=[1])
rho_C_test = partial_trace(rho_ABC, [4, 3, 2], keeps=[2])

# See if the outputs of partial_trace are correct
print("rho_AB test correct? ", np.allclose(rho_AB_test, rho_AB))
print("rho_AC test correct? ", np.allclose(rho_AC_test, rho_AC))
print("rho_A test correct? ", np.allclose(rho_A_test, rho_A))
print("rho_B test correct? ", np.allclose(rho_B_test, rho_B))
print("rho_C test correct? ", np.allclose(rho_C_test, rho_C))

n = 10
i = 0
j = 1 # i < j
rho_comps = [qiskit.quantum_info.random_unitary(2).data for _ in range(n)]
rho_comps = [rho / np.trace(rho) for rho in rho_comps]
rho = 1
for l in range(n):
    rho = np.kron(rho, rho_comps[l])
print('--')

rho_i = rho_comps[i]
rho_i_traced = partial_trace(rho, [2]*n, keeps=[i])
print(np.allclose(rho_i, rho_i_traced))

rho_ij_seq = 1
rho_ij_seq = np.kron(rho_ij_seq, rho_comps[i])
rho_ij_seq = np.kron(rho_ij_seq, rho_comps[j])
rho_ij = np.kron(rho_comps[i], rho_comps[j])
rho_ij_traced = partial_trace(rho, [2]*n, keeps=[i,j])

print(np.allclose(rho_ij, rho_ij_traced))

for k in range(n-1, 0, -1):
    rho_1tok = 1
    for i in range(k):
        rho_1tok = np.kron(rho_1tok, rho_comps[i])
    rho_1tok_traced = partial_trace(rho, [2]*n, keeps=range(k))
    print(np.allclose(rho_1tok, rho_1tok_traced))


import qiskit
from qiskit.quantum_info import Operator

circ = qiskit.QuantumCircuit(2)
circ.x(0)

XI = Operator(circ).data

X_data = np.array([[0, 1], [1, 0]])
I_data = np.eye(2)
XI_data = np.kron(I_data, X_data)
print(XI)
print(XI_data)