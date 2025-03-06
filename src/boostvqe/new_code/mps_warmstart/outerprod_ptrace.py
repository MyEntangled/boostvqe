import quimb.tensor as qtn
import numpy as np

def reorder_indices(mps:qtn.MatrixProductState) -> qtn.MatrixProductState:
    # Standardize indices for each tensor
    for i in range(mps.num_tensors):
        tensor = mps[i]

        # Define the expected order
        inds = tensor.inds

        left = next((ind for ind in inds if 'link' in ind and int(ind.split('_')[-1]) == i), None)
        right = next((ind for ind in inds if 'link' in ind and int(ind.split('_')[-1]) == i+1), None)
        physical = next((ind for ind in inds if ('k' in ind or 'b' in ind) and ('link' not in ind)), None)
        rest = [ind for ind in inds if ind not in [left, right, physical]]

        # Construct the desired order
        desired_order = tuple(filter(None, [left, right, physical, *rest]))
        # Reorder indices
        tensor.transpose_(*desired_order)

def outer_product_mps(mps1:qtn.MatrixProductState, mps2:qtn.MatrixProductState) -> qtn.MatrixProductOperator:
    """
    Create the outer product of two MPS while maintaining the tensor network structure.

    Parameters:
    mps1 (qtn.MPS): The first MPS tensor network.
    mps2 (qtn.MPS): The second MPS tensor network.

    Returns:
    qtn.MatrixProductOperator: The MPO representing |mps1><mps2|.
    """
    # Ensure the MPS objects have the same number of sites
    if mps1.nsites != mps2.nsites:
        raise ValueError("Both MPS must have the same number of sites.")

    L = mps1.nsites

    ## Renaming virtual bonds (identical virtual bonds' naming for mps1 and mps2 for later fusing)
    for n in range(L-1):
        bond_ind = mps1[n].bonds(mps1[n + 1]).pop()
        mps1[n].reindex_({bond_ind: f'link_{n}_{n + 1}'})
        mps1[n + 1].reindex_({bond_ind: f'link_{n}_{n + 1}'})

        bond_ind = mps2[n].bonds(mps2[n + 1]).pop()
        mps2[n].reindex_({bond_ind: f'link_{n}_{n + 1}'})
        mps2[n + 1].reindex_({bond_ind: f'link_{n}_{n + 1}'})

    # Change lower physical bonds
    for n in range(L):
        mps2[n].reindex_({f'k{n}': f'b{n}'})

    reorder_indices(mps1)
    reorder_indices(mps2)

    # Create a new tensor network to store the outer product
    mpo = qtn.MatrixProductOperator.new(L=L, cyclic=False, upper_ind_id='k{}', lower_ind_id='b{}', site_tag_id='I{}')

    # Iterate through the tensors in the MPS
    for n in range(L):
        ta = mps1[n]
        tb = mps2[n]

        if n == 0 and L == 1:
            p1, = ta.shape
            p2, = tb.shape

            c = np.tensordot(ta.data, tb.data, axes=0) # shape: (p1, p2)

            # Create a new tensor with the merged data
            new_tensor = qtn.Tensor(
                data=c,
                inds=[f'k{n}', f'b{n}'],
                tags=[f'I{n}']
            )

        elif n == 0 and L > 1:
            r1, p1 = ta.shape
            r2, p2 = tb.shape

            c = np.tensordot(ta.data, tb.data, axes=0)
            c_transposed = np.transpose(c, (0, 2, 1, 3))  # new shape: (r1, r2, p1, p2)
            c_merged = c_transposed.reshape(r1 * r2, p1, p2)

            # Create a new tensor with the merged data
            new_tensor = qtn.Tensor(
                data=c_merged,
                inds=[f'link_{n}_{n+1}', f'k{n}', f'b{n}'],
                tags=[f'I{n}']
            )

        elif n == L-1:
            l1, p1 = ta.shape
            l2, p2 = tb.shape

            c = np.tensordot(ta.data, tb.data, axes=0)
            c_transposed = np.transpose(c, (0, 2, 1, 3))  # new shape: (l1, l2, p1, p2)
            c_merged = c_transposed.reshape(l1 * l2, p1, p2)

            # Create a new tensor with the merged data
            new_tensor = qtn.Tensor(
                data=c_merged,
                inds=[f'link_{n - 1}_{n}', f'k{n}', f'b{n}'],
                tags=[f'I{n}']
            )
        else:
            l1, r1, p1 = ta.shape
            l2, r2, p2 = tb.shape

            c = np.tensordot(ta.data, tb.data, axes=0)
            c_transposed = np.transpose(c, (0, 3, 1, 4, 2, 5))  # new shape: (l1, l2, r1, r2, p1, p2)
            c_merged = c_transposed.reshape(l1 * l2, r1 * r2, p1, p2)

            # Create a new tensor with the merged data
            new_tensor = qtn.Tensor(
                data=c_merged,
                inds=[f'link_{n-1}_{n}', f'link_{n}_{n+1}', f'k{n}', f'b{n}'],
                tags=[f'I{n}']
            )

        mpo.add_tensor(new_tensor)

    return mpo

def partial_trace_mpo(mpo:qtn.MatrixProductOperator, keep_sites) -> qtn.MatrixProductOperator:
    """
    Compute the partial trace of an MPO.

    Parameters:
    mpo (qtn.MatrixProductOperator): The MPO.
    keep_nodes (list): The indices to keep.

    Returns:
    qtn.MatrixProductOperator: The MPO representing the partial trace.
    """
    # Get indices to be contracted
    L = mpo.num_tensors
    left_inds = [f'k{n}' for n in range(L) if n not in keep_sites]
    right_inds = [f'b{n}' for n in range(L) if n not in keep_sites]
    res = mpo.trace(left_inds=left_inds, right_inds=right_inds)

    return res


# Example usage
if __name__ == "__main__":
    # Create two MPS objects
    zero_mps = qtn.MPS_computational_state('000')
    one_mps = qtn.MPS_computational_state('011')


    x_state = np.array([1, 1]) / np.sqrt(2)
    y_state = np.array([1, 1j])/np.sqrt(2)

    xx_state = np.kron(x_state, x_state)
    yy_state = np.kron(y_state, y_state)
    xy_state = np.kron(x_state, y_state)
    yx_state = np.kron(y_state, x_state)

    mps1 = qtn.MatrixProductState.from_dense(np.array(xy_state, dtype=np.complex128))
    mps2 = qtn.MatrixProductState.from_dense(np.array(yx_state, dtype=np.complex128))

    #print(mps1, mps2)

    # Compute the outer product
    outer_product_tn = outer_product_mps(zero_mps, one_mps.H)
    print("Outer Product MPO:", outer_product_tn.shape, outer_product_tn)
    #print((outer_product_tn ^...).data)
    print('----')

    ptrace = partial_trace_mpo(outer_product_tn, keep_sites=[1,2])
    # print(ptrace)
    # print(ptrace.data)

    data = ptrace.data.transpose(0, 2, 1, 3)
    print(data.reshape(4,4))

    # res = outer_product_tn.trace(left_inds=[f'k{n}' for n in [0]], right_inds=[f'b{n}' for n in [0]])
    # print(res)
    # print(res.data)