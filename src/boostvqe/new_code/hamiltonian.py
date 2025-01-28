import quimb.tensor as qtn

# Function to build the XXZ Hamiltonian with transverse field
def build_xxz_hamiltonian(n_sites, couplings):
    Jx, Jy, Jz, h = couplings
    paulis = []
    coeffs = []
    sites = []

    mpo_terms = []

    for i in range(n_sites - 1):
        if abs(Jx) > 0:
            mpo_terms.append((4 * Jx, i, i + 1, 'X', 'X'))
            paulis.append('XX')
            coeffs.append(Jx)
            sites.append([i, i + 1])
        if abs(Jy) > 0:
            mpo_terms.append((4 * Jy, i, i + 1, 'Y', 'Y'))
            paulis.append('YY')
            coeffs.append(Jy)
            sites.append([i, i + 1])
        if abs(Jz) > 0:
            mpo_terms.append((4 * Jz, i, i + 1, 'Z', 'Z'))
            paulis.append('ZZ')
            coeffs.append(Jz)
            sites.append([i, i + 1])

            # Add transverse field term
    for i in range(n_sites):
        mpo_terms.append((2 * h, i, 'Z'))
        paulis.append('Z')
        sites.append([i])
        coeffs.append(h)

    H = [coeffs, paulis, sites]

    # Builtd the MPO
    mpo = qtn.SpinHam1D(S=0.5)
    for term in mpo_terms:
        if len(term) == 5:
            coeff, site1, site2, op1, op2 = term
            mpo[site1, site2] += coeff, op1, op2
        elif len(term) == 3:
            coeff, site, op = term
            mpo[site] += coeff, op

    return mpo.build_mpo(n_sites), H


if __name__ == '__main__':
    import numpy as np
    import scipy

    # Define parameters for the XXZ Hamiltonian
    n_sites = 4  # Number of sites in the chain
    Jx = 1  # Coupling in the x-direction
    Jy = 1  # Coupling in the y-direction
    Jz = 0.6  # Coupling in the z-direction
    h = +0.4  # Transverse field strength
    couplings = np.array([Jx, Jy, Jz, h]) / n_sites

    # Build the MPO Hamiltonian
    hamiltonian, H = build_xxz_hamiltonian(n_sites, couplings)

    if n_sites < 12:
        dense_hamiltonian = hamiltonian.to_dense()

        ew, ev = scipy.linalg.eigh(dense_hamiltonian)
        print(ew)