import numpy as np
import scipy
import quimb.tensor as qtn
from quimb.tensor import MatrixProductState

from typing import List

# qtn.set_contract_backend('jax')
# qtn.set_tensor_linop_backend('jax')

#autoray.register_function('quimb', 'transpose', qtn.Tensor.transpose)

def is_unitary(U):
    return np.allclose(U.conj().T @ U, np.eye(U.shape[0]))

def canonicalize(mps:qtn.MatrixProductState):
    mps.right_canonize()
    return

def check_canonicalization(mps:qtn.MatrixProductState):
    for i in range(mps.num_tensors):
        tensor = mps[i]
        if i == 0:
            norm = mps.norm()
            print(f"Site {i}: Normalization check -> {np.isclose(norm, 1.)}")

        else:
            left_idx = tensor.inds[0]
            new_idx = left_idx + '_prime'
            tensor_conj = tensor.conj().reindex({left_idx: new_idx})

            to_be_contracted = (tensor | tensor_conj)
            left_bond = to_be_contracted.contract(output_inds=[left_idx, new_idx])
            #print(left_bond.data)
            print(f"Site {i}: Orthogonality check -> {is_unitary(left_bond.data)}")
    return


def nullspace_qr(A, tol=1e-12):
    """
    Compute an orthonormal basis for the nullspace of A using QR decomposition.

    Parameters:
    A (ndarray): Input matrix.
    tol (float): Tolerance for detecting rank deficiency.

    Returns:
    ndarray: Orthonormal basis for the nullspace of A.
    """
    # QR decomposition of A^T
    Q, R = scipy.linalg.qr(A.conj().T, mode='full')

    # Find the rank of A based on the tolerance
    rank = np.sum(np.abs(np.diag(R)) > tol)

    # The nullspace basis corresponds to the last columns of Q
    nullspace_basis = Q[:, rank:]
    return nullspace_basis

def truncated_mps_to_circuit(mps:qtn.MatrixProductState):
    canonicalize(mps)
    G_gate_list = []
    G_tensor_list = []

    for n in range(mps.num_tensors):
        if n == mps.num_tensors - 1:
            G = mps[n].data

            #print(n, is_unitary(G))
            #G_gate_list.append(G)
            G_gate_list.append(np.transpose(G, axes=(1,0)))

            G_tensor = qtn.Tensor(G, inds=(f'link_{n-1}_{n}', mps[n].inds[-1]), tags=f'I{n}')
            G_tensor_list.append(G_tensor)

        elif n > 0:
            G = np.zeros(shape=(2,2,2,2), dtype='complex')
            A = mps[n].data

            G[0,:,:,:] = A

            #nullspace = scipy.linalg.null_space(A.reshape(A.shape[1], -1)).T
            nullspace = nullspace_qr(A.reshape(A.shape[1], -1)).T

            #print(f'n={n} nullspace', nullspace.shape)
            if nullspace.shape != (2,4):
                print(A.shape, nullspace.shape)
                raise Exception('Nullspace does not have enough dimension. Tolerance might have been reached.')
                return Exception('Nullspace does not have enough dimension. Tolerance might have been reached.')

            G[1,:,:,:] = nullspace.reshape(2,2,2)

            # G_gate = G.reshape(4,4)
            # #print(n, is_unitary(G_gate))
            # G_gate_list.append(G_gate)

            G_gate = np.transpose(G, axes=(2,3,0,1))
            G_gate_list.append(G_gate.reshape(4,4))

            G_tensor = qtn.Tensor(G, inds=(f'b{n+1}', f'link_{n-1}_{n}', f'link_{n}_{n+1}', mps[n].inds[-1]), tags=f'I{n}')
            G_tensor_list.append(G_tensor)

        else: # n = 0
            G = np.zeros(shape=(2,2,2,2), dtype='complex')
            A = mps[n].data
            #print(A.shape)
            G[0,0,:,:] = A

            #nullspace = scipy.linalg.null_space(A.reshape(1, -1)).T
            nullspace = nullspace_qr(A.reshape(1, -1)).T

            #print('n=0 nullspace', nullspace.shape)
            if nullspace.shape != (3,4):
                print(A.shape, nullspace.shape)
                raise Exception('Nullspace does not have enough dimension. Tolerance might have been reached.')
                return Exception('Nullspace does not have enough dimension. Tolerance might have been reached.')

            G[0,1,:,:] = nullspace[0].reshape(2,2)
            G[1,:,:,:] = nullspace[1:].reshape(-1,2,2)

            # G_gate = G.reshape(4,4)
            # #print(n, is_unitary(G_gate))
            # G_gate_list.append(G_gate)

            G_gate = np.transpose(G, axes=(2,3,0,1))
            G_gate_list.append(G_gate.reshape(4,4))

            G_tensor = qtn.Tensor(G, inds=(f'b{n+1}', f'b{n}', f'link_{n}_{n + 1}', mps[n].inds[-1]), tags=f'I{n}')
            G_tensor_list.append(G_tensor)

    return G_gate_list, G_tensor_list

def output_to_mps(psi:qtn.Tensor, bond_dim:int=None) -> MatrixProductState:
    # The input tensor is assumed to have shape [2]*n
    n = len(psi.inds)
    site_ind_id = 'k{}'
    site_tag_id = 'I{}'

    # Initialize an empty MPS object
    mps = qtn.MatrixProductState.new(L=n,
                                     cyclic=False,
                                     site_ind_id=site_ind_id,
                                     site_tag_id=site_tag_id)

    # Start with the initial tensor
    current_tensor = psi

    left_bond = None  # To track the previous virtual bond index

    for i in range(n - 1):
        left_inds = [f'b{i}']  # Physical index for the current node

        # Add the previous bond index if it exists
        if left_bond is not None:
            left_inds += left_bond

        #print(f'Spitting {i}', left_inds, right_inds)
        # Perform SVD splitting
        T1, current_tensor = current_tensor.split(
            left_inds=left_inds,
            #right_inds=right_inds,
            method='eig',
            absorb='right',
            max_bond=bond_dim,
            bond_ind = f'link_{i}_{i+1}'
        )
        T1.tags.clear()
        T1.add_tag(f'I{i}')

        left_bond = list(T1.bonds(current_tensor))  ## 'link_{i}_{i+1}'

        # Update the MPS
        mps.add_tensor(T1)

    # Add the final tensor (last physical index and last virtual bond)
    final_inds = [f'b{n - 1}']
    if left_bond is not None:
        final_inds += left_bond
    current_tensor.tags.clear()
    current_tensor.add_tag(f'I{n-1}')
    mps.add_tensor(current_tensor)

    reorder_indices(mps)
    return mps

def output_to_mps_new(tensor_list:List[qtn.Tensor], state_mps:qtn.MatrixProductState, max_bond:int) -> MatrixProductState:
    mps = qtn.MatrixProductState.new(L=len(tensor_list),
                                     cyclic=False,
                                     site_ind_id='k{}',
                                     site_tag_id='I{}')

    #mps = qtn.TensorNetwork.new()

    ## Create the MPS
    for n in range(len(tensor_list)):
        G_tensor = tensor_list[n]
        state_tensor = state_mps[n]


        # Contract gate with corresponding qubit state
        #print('-----')
        #print(f'n={n}')
        #print('Gate:', G_tensor)
        #print('State:', state_tensor)
        tensor = (G_tensor | state_tensor)^...


        if n == 0:
            tensor.reindex_({'link_0_1': 'link_1_2'})
            T1, T2 = tensor.split(left_inds=['b0'], method='svd', absorb='right', bond_ind='link_0_1')

            T1.tags.clear()
            T1.add_tag(f'I0')
            T2.tags.clear()
            T2.add_tag(f'I1')

            mps.add_tensor(T1)
            mps.add_tensor(T2)
        elif n == len(tensor_list) - 2:
            to_be_contracted = tensor
        elif n == len(tensor_list) - 1:
            merge_last_two = tensor.contract(to_be_contracted)

            merge_last_two.reindex_({f'link_{n-2}_{n-1}': f'link_{n-1}_{n}'})
            merge_last_two.tags.clear()
            merge_last_two.add_tag(f'I{n}')

            mps.add_tensor(merge_last_two)
        else:
            tensor.reindex_({f'link_{n-1}_{n}': f'link_{n}_{n+1}', f'link_{n}_{n+1}':f'link_{n+1}_{n+2}'})
            tensor.tags.clear()
            tensor.add_tag(f'I{n+1}')

            mps.add_tensor(tensor)

        #print('Tensor:', tensor)

    #print('Prefuse', mps)
    mps.fuse_multibonds_()
    #print('Postfuse', mps)

    for n in range(len(tensor_list)-1):
        bond_ind = mps[n].bonds(mps[n + 1]).pop()
        mps[n].reindex_({bond_ind: f'slink_{n}_{n + 1}'})
        mps[n + 1].reindex_({bond_ind: f'slink_{n}_{n + 1}'})

    #print('-----')
    #print('Full', mps)

    ## Reorder indices in "lrp" order and compress virtual bonds.
    reorder_indices(mps)
    mps.compress_all_1d_(max_bond=max_bond)
    #print('Standardized', mps)
    return mps


def reorder_indices(mps:qtn.MatrixProductState) -> qtn.MatrixProductState:
    # Standardize indices for each tensor
    for i in range(mps.num_tensors):
        tensor = mps[i]

        # Define the expected order
        inds = tensor.inds

        left = next((ind for ind in inds if 'slink' in ind and int(ind.split('_')[-1]) == i), None)
        right = next((ind for ind in inds if 'slink' in ind and int(ind.split('_')[-1]) == i+1), None)
        physical = next((ind for ind in inds if ('k' in ind or 'b' in ind) and ('slink' not in ind)), None)
        rest = [ind for ind in inds if ind not in [left, right, physical]]

        # Construct the desired order
        desired_order = tuple(filter(None, [left, right, physical, *rest]))
        # Reorder indices
        tensor.transpose_(*desired_order)

def disentangling_gates(input_mps:qtn.MatrixProductState, num_layers:int=1, hamiltonian=None):
    mps_list = []
    temp = input_mps.copy()
    for n in range(temp.num_tensors-1):
        bond_ind = temp[n].bonds(temp[n+1]).pop()
        temp[n].reindex_({bond_ind: f'slink_{n}_{n+1}'})
        temp[n+1].reindex_({bond_ind: f'slink_{n}_{n+1}'})
    mps_list.append(temp)

    max_bond = mps_list[0].max_bond()
    circuit_unitaries = []

    # zero_state = np.zeros(2 ** input_mps.num_tensors)
    # zero_state[0] = 1
    zero_mps = qtn.MPS_computational_state('0'*input_mps.num_tensors)

    if hamiltonian is not None:
        from boostvqe.src.boostvqe.new_code.mps_warmstart.create_mps_circuit import create_circuit_from_gate_unitaries ## HACKY TRICK, ONLY USED TO SHOW ENERGY


    for k in range(num_layers):
        mps = mps_list[k]
        print(f'MPS layer {k+1}')

        ## Truncating virtual bonds
        truncated_mps = qtn.tensor_1d_compress.tensor_network_1d_compress_dm(mps, max_bond=2, normalize=True)

        ## Making circuit
        output = truncated_mps_to_circuit(truncated_mps)
        if isinstance(output, Exception):
            return circuit_unitaries
        else:
            unitary_list, tensor_list = output

        output_mps = output_to_mps_new(tensor_list, state_mps=mps, max_bond=min(max_bond*num_layers,128))

        reindex_map = {f'b{i}': f'k{i}' for i in range(output_mps.num_tensors)}

        # Apply reindexing to each tensor in the network
        for tensor in output_mps:
            tensor.reindex_(reindex_map)

        mps_list.append(output_mps)
        circuit_unitaries.append(unitary_list)

        print('Output quality', abs(zero_mps.H @ output_mps)**2)

        if hamiltonian is not None:
            circ = qtn.CircuitMPS(input_mps.num_tensors)
            gates_to_append = list(create_circuit_from_gate_unitaries(circuit_unitaries))
            circ.apply_gates(gates_to_append,)

            energy = (circ.psi.conj().reindex_(
                {f'k{n}': f'b{n}' for n in range(circ.psi.num_tensors)}) | hamiltonian | circ.psi) ^ ...

            print('Circuit gates:', circ.num_gates)
            print('Energy', energy)
        print('--')
    return circuit_unitaries


if __name__ == "__main__":

    nqubits = 30
    bond_dim = 10
    psi = qtn.MPS_rand_state(nqubits, bond_dim=bond_dim, phys_dim=2, cyclic=False, normalize=True)

    circuit_unitaries = disentangling_gates(psi, num_layers=100)
    print(len(circuit_unitaries))
    #