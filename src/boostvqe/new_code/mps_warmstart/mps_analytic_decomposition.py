import numpy as np
import scipy
import quimb.tensor as qtn
import quimb as qu
from quimb.tensor import MatrixProductState

from typing import List

from src.boostvqe.new_code.mps_warmstart.create_mps_circuit import generate_gates_from_unitaries
from src.boostvqe.new_code.mps_warmstart.helper import is_unitary, apply_circuit_mps


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

def output_to_mps_new(tensor_list:List[qtn.Tensor], state_mps:qtn.MatrixProductState, max_bond:int) -> MatrixProductState:
    mps = qtn.MatrixProductState.new(L=len(tensor_list),
                                     cyclic=False,
                                     site_ind_id='k{}',
                                     site_tag_id='I{}')

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

def analytic_decomposition(psi_target:qtn.MatrixProductState | qtn.CircuitMPS,
                           num_layers:int=1,
                           hamiltonian=None,
                           fid_target=0.99,
                           use_raw_gates=True):
    
    if isinstance(psi_target, qtn.CircuitMPS):
        psi_target = psi_target.psi

    ## Create a zero MPS state that is the desired output after disentangling.
    zero_mps = qtn.MPS_computational_state('0' * psi_target.num_tensors)

    mps_list = []
    temp = psi_target.copy()
    for n in range(temp.num_tensors-1):
        bond_ind = temp[n].bonds(temp[n+1]).pop()
        temp[n].reindex_({bond_ind: f'slink_{n}_{n+1}'})
        temp[n+1].reindex_({bond_ind: f'slink_{n}_{n+1}'})
    mps_list.append(temp)

    max_bond = mps_list[0].max_bond()

    circuit_unitaries = []
    qargs = []
    if hamiltonian is not None:
        circ = qtn.CircuitMPS(psi_target.num_tensors, max_bond=4096, cutoff=1e-8)

    energy = None
    fid = np.abs(zero_mps.H @ psi_target)

    for k in range(num_layers):
        mps = mps_list[k]
        #print(f'MPS layer {k+1}')

        ## Truncating virtual bonds
        truncated_mps = qtn.tensor_1d_compress.tensor_network_1d_compress_dm(mps, max_bond=2, normalize=True)

        ## Making circuit
        output = truncated_mps_to_circuit(truncated_mps)
        if isinstance(output, Exception):
            return circuit_unitaries
        else:
            unitary_list, tensor_list = output

        output_mps = output_to_mps_new(tensor_list, state_mps=mps, max_bond=min(max_bond*num_layers,128))
        output_mps.normalize()

        # Apply reindexing to each tensor in the network
        reindex_map = {f'b{i}': f'k{i}' for i in range(output_mps.num_tensors)}
        for tensor in output_mps:
            tensor.reindex_(reindex_map)

        # Update the circuit (unitaries, qargs) with the new layer
        mps_list.append(output_mps)
        circuit_unitaries = [unitary_list] + circuit_unitaries ## LATEST LAYER FIRST
        layer_unitaries = circuit_unitaries[0]  # latest layer
        layer_qargs = [(n, n + 1) for n in range(len(layer_unitaries) - 1)]
        layer_qargs.append((len(layer_unitaries) - 1,))
        qargs = [layer_qargs] + qargs

        fid = np.abs(zero_mps.H @ output_mps)
        print('Output quality', fid)

        if hamiltonian is not None:
            layer_gates = list(generate_gates_from_unitaries(circuit_unitaries[0], qargs[0], use_raw_gates=use_raw_gates))

            full_circ = apply_circuit_mps(psi_target.num_tensors, layer_gates + list(circ.gates))

            energy = (full_circ.psi.conj().reindex_(
                {f'k{n}': f'b{n}' for n in range(full_circ.psi.num_tensors)}) | hamiltonian | full_circ.psi) ^ ...

            circ = full_circ ## Update the circuit
            print('Circuit gates:', circ.num_gates)
            print('Energy', energy)
        #print('--')

        if fid > fid_target:
            break

    #print(len(circ.gates))

    return circuit_unitaries, qargs, fid, energy


if __name__ == "__main__":
    nqubits = 30
    bond_dim = 64

    from src.boostvqe.new_code.hamiltonian import build_xxz_hamiltonian
    Jx = 1  # Coupling in the x-direction
    Jy = 1  # Coupling in the y-direction
    Jz = 0  # Coupling in the z-direction
    h = +0.5  # Transverse field strength

    # Build the Hamiltonian
    #ham_build, H = build_xxz_hamiltonian(nqubits, [Jx, Jy, Jz, h])
    ham = qtn.MPO_ham_heis(nqubits, (4*Jx, 4*Jy, 4*Jz), bz=-2*h)

    #print((ham_build - ham).norm())
    bond_dims = [4]
    dmrg = qtn.DMRG2(ham, bond_dims=bond_dims, cutoffs=1e-6)
    res = dmrg.solve(verbosity=0)
    dmrg.solve()
    psi = dmrg.state

    # print(psi.max_bond())
    # print(psi.shape)

    circuit_unitaries, qargs, _, _ = analytic_decomposition(psi, hamiltonian=ham, num_layers=5)
    print(len(circuit_unitaries), len(qargs))

