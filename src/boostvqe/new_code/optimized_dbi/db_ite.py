import quimb as qu
import quimb.tensor as qtn
import numpy as np


def build_hamiltonian_mpo(n_sites:int):
    mpo = qtn.SpinHam1D(S=0.5, cyclic=None)
    mpo += 0.5, '+', '-'
    mpo += 0.5, '-', '+'
    mpo += 0.5, 'Z', 'Z'
    return mpo.build_mpo(n_sites)

def linear_ite_mpo(H, tau):
    return qtn.MPO_identity_like(H) - tau * H

def get_exact_energy(H):
    """
    Get the exact energy of the Hamiltonian via DMRG.
    """
    bond_dims = [16,32,128,512,1024]
    dmrg = qtn.DMRG2(H, bond_dims=bond_dims, cutoffs=1e-10)
    res = dmrg.solve(verbosity=0, max_sweeps=5)
    print("DMRG converged: ", res)
    return dmrg.energy, dmrg.state

def approx_ground_state(H, max_bond=16, nsweeps=1):
    bond_dims = [max_bond]  # gradually increase states kept

    dmrg = qtn.DMRG2(H, bond_dims=bond_dims, cutoffs=1e-10)
    res = dmrg.solve(verbosity=0, max_sweeps=nsweeps)
    return dmrg.state, dmrg.energy


def get_energy_and_variance(psi:qtn.MatrixProductState, hamiltonian:qtn.MatrixProductOperator,
                            contract_tree=None):

    psi_ket = psi.copy()
    psi_bra = psi.H.reindex_sites(new_id="b{}", inplace=False)
    energy = (psi_bra | hamiltonian | psi_ket).contract(all, optimize=contract_tree)

    hamiltonian_ket = hamiltonian.reindex_lower_sites("link{}", inplace=False)
    hamiltonian_bra = hamiltonian.reindex_upper_sites("link{}", inplace=False)

    M2 = (psi_bra | hamiltonian_bra | hamiltonian_ket | psi_ket).contract(all, optimize=contract_tree)
    return energy, M2-energy**2

def get_third_moment(psi:qtn.MatrixProductState, hamiltonian:qtn.MatrixProductOperator, contract_tree=None):
    psi_ket = psi.copy()
    psi_bra = psi.H.reindex_sites(new_id="b{}", inplace=False)

    hamiltonian_ket = hamiltonian.reindex_lower_sites("link_km{}", inplace=False)
    hamiltonian_mid = hamiltonian.reindex_upper_sites("link_km{}", inplace=False)
    hamiltonian_mid.reindex_lower_sites_("link_mb{}")
    hamiltonian_bra = hamiltonian.reindex_upper_sites("link_mb{}", inplace=False)

    third_moment = (psi_bra | hamiltonian_bra | hamiltonian_mid | hamiltonian_ket | psi_ket).contract(all, optimize=contract_tree)
    return third_moment


def compute_parameters_monomial(psi:qtn.MatrixProductState, hamiltonian:qtn.MatrixProductOperator,
                                x:float, y:float, perform_check:bool=True,
                                contract_tree=None):
    """
    Compute s_phi, a(s_phi), and b(s_phi) for operators of the form xI + yH
    where (xI + yH)|phi>/||(xI + yH)|phi>|| = [a(s_phi)I + b(s_phi)H] |phi>
    """

    E, V = get_energy_and_variance(psi, hamiltonian, contract_tree=contract_tree)
    M2 = V + E**2
    V_sqrt = np.sqrt(V)

    norm = np.sqrt(x**2 + y**2 * M2 + 2*x*y*E)

    s_phi = -np.sign(y) / V_sqrt * np.arccos((x+y*E)/norm)

    angle = s_phi * V_sqrt
    sin = np.sin(angle)
    cos = np.cos(angle)
    a = E/V_sqrt * sin + cos
    b = -1./V_sqrt * sin

    if perform_check:
        ## Check the norm of (aI + bH)|phi>
        ## <phi|(aI + bH)(aI + bH)|phi> = a^2 <phi|phi> + b^2*<phi|H^2|phi> + 2ab*<phi|H|phi>
        norm_check = np.sqrt(a**2 + b**2 * M2 + 2*a*b*E)
        assert np.isclose(norm_check, 1), f"Norm check failed: {norm_check} != 1"

        ## Check the energy of (aI + bH)|phi>
        ## <phi|(aI + bH) H (aI + bH)|phi> = a^2 <phi|H|phi> + b^2 <phi|H^3|phi> + 2ab <phi|H^2|phi>
        M3 = get_third_moment(psi, hamiltonian, contract_tree=contract_tree)
        energy_ab = a**2 * E + b**2 * M3 + 2*a*b*M2

        ## Energy by (xI + yH)|phi>/||xI + yH|phi>||
        expval_xy = x**2 * E + y**2 * M3 + 2*x*y*M2
        energy_xy = expval_xy / (x**2 + y**2 * M2 + 2*x*y*E)

        # print("Energy with aI + bH", energy_ab)
        # print("Energy with xI + yH", energy_xy)
        assert np.isclose(energy_ab, energy_xy), f"Energy check failed: {energy_ab} != {energy_xy}"
    return E, V, s_phi, a, b

def compute_optimal_time(psi:qtn.MatrixProductState, hamiltonian:qtn.MatrixProductOperator, k:int=0,
                         contract_tree=None):
    """
    Find the best tau in F=I-tau*H to minimize the energy <psi_new|H|psi_new> for |psi_new> = F|psi> / ||F|psi>||
    """

    E, V = get_energy_and_variance(psi, hamiltonian, contract_tree=contract_tree)
    M2 = V + E**2
    V_sqrt = np.sqrt(V)

    M3 = get_third_moment(psi, hamiltonian, contract_tree=contract_tree)
    skew = (M3 - 3 * E * M2 + 3 * E**2 * E - E**3) / V_sqrt**3 # skewness

    a = np.arctan2(skew, 2)

    ## Find the optimal solution:
    ## 2 * s_opt * V_sqrt + a := np.pi/2 + k * 2*np.pi
    ## Rearranging gives us
    ## s_opt = (np.pi/2 + k * 2*np.pi - a) / (2*V_sqrt)

    s_opt = (np.pi/2 + k * 2 * np.pi - np.arctan2(skew, 2)) / (2 * V_sqrt)  # Eq 6
    tau_opt = np.tan(abs(s_opt) * V_sqrt) / (E * np.tan(abs(s_opt) * V_sqrt) + V_sqrt)

    ## Predict energy of the optimal time
    ## Recall <psi| (I - tau * H) (I - tau * H) |psi> = <psi|psi> - 2tau*<psi|H|psi> + tau^2*<psi|H^2|psi>

    s_tau_func = lambda tau: np.sign(tau) / V_sqrt * np.arccos((1. - tau * E) / np.sqrt(1. - 2*tau*E + tau**2 * M2))  ## Eq 7
    energy_gain_func = lambda s: (skew * np.sin(s * V_sqrt)**2 - np.sin(2 * s * V_sqrt)) * V_sqrt   ## Eq 8
    #energy_gain_func = lambda s: (skew - np.sqrt(skew**2 + 4) * np.sin(2*s*V_sqrt + a)) * V_sqrt/2 ## Eq 8 (alternative form)

    opt_gain = energy_gain_func(s_tau_func(tau_opt))
    opt_gain_ideal = -V_sqrt * (np.sqrt(1. + 0.25 * skew**2) - 0.5 * skew)  ## Eq 10

    assert np.isclose(np.sin(2 * s_opt * V_sqrt + a), 1), f"Sine factor {np.sin(2 * s_opt * V_sqrt + a)} is not close to 1"
    assert np.isclose(opt_gain, opt_gain_ideal), f"Predicted gain {opt_gain} does not match ideal gain {opt_gain_ideal}"

    opt_energy = E + opt_gain

    return tau_opt, opt_gain, opt_energy


# Example usage
if __name__ == "__main__":
    from tqdm import tqdm
    import pickle
    import cotengra as ctg

    opt = ctg.ReusableHyperOptimizer(methods=['greedy'],reconf_opts={},max_repeats=32,max_time="rate:1e6",parallel=True)

    overlap_fn = lambda psi, phi: abs(psi.H @ phi)

    n_sites = 50
    max_bond = 256
    simulate_steps = 50
    init = 'hadamard' # 'dmrg' or 'hadamard'
    filename = f'simulate_db_ite_n{n_sites}_{init}.pkl'

    H = build_hamiltonian_mpo(n_sites)
    exact_energy, exact_state = get_exact_energy(H)
    print(f"Exact energy: {exact_energy}")

    print(overlap_fn(exact_state, exact_state))

    step_sizes = np.round(np.arange(-0.25, 0.2501, step=0.01), 3)
    step_sizes = step_sizes[step_sizes != 0.0]

    try:
        with open(filename, 'rb') as f:
            record = pickle.load(f)
            record['step_sizes'] = list(record['step_sizes'])

    except FileNotFoundError:
        record = {'n_sites': n_sites, 'E_exact': exact_energy, 'psi_exact': exact_state,
                  'psi_init': None, 'E_init': None, 'F_init': None,
                  'step_sizes': [], 'fixed_sizes_energy': [], 'fixed_sizes_fidelity': [],
                  'optimal_time': [], 'optimal_time_energy': [], 'optimal_time_fidelity': []}

    ## Initialize the MPS
    if record['psi_init'] is None or record['E_init'] is None or record['F_init'] is None:

        if init == 'hadamard':
            circ_init = qtn.CircuitMPS(n_sites, max_bond=max_bond)
            for i in range(n_sites):
                circ_init.h(i)

            psi_init = circ_init.psi
            E_init = get_energy_and_variance(psi_init, H)[0]

        elif init == 'dmrg':
            psi_init, E_init = approx_ground_state(H)

        F_init = overlap_fn(exact_state, psi_init)

        record['psi_init'] = psi_init
        record['E_init'] = E_init
        record['F_init'] = F_init

        print(f"Initial energy ({init} initialized): {E_init}, fidelity: {F_init}")

    else:
        psi_init = record['psi_init']
        E_init = record['E_init']
        F_init = record['F_init']
        print(f"Using precomputed initial state with initial energy {E_init} and fidelity {F_init}")


    if len(record['optimal_time']) == 0:
        print("Starting simulation with optimal time steps...")
        psi = psi_init.copy()
        E, V = get_energy_and_variance(psi, H)
        F = overlap_fn(exact_state, psi)
        record['optimal_time_energy'].append(E)
        record['optimal_time_fidelity'].append(F)

        ## Compute optimal tau
        tau, energy_gain, energy = compute_optimal_time(psi, H, contract_tree=opt)


        for step in tqdm(range(simulate_steps), desc="Iterations"):
            psi = qtn.tensor_1d_compress.mps_gate_with_mpo_dm(psi, linear_ite_mpo(H, tau), max_bond=max_bond)
            psi.normalize()
            E, V = get_energy_and_variance(psi, H)
            F = overlap_fn(exact_state, psi)
            assert np.isclose(E, energy), f"Prediction {energy} does not match actual energy {E}"
            print(f"tau = {tau:.2f}, energy = {E:.3f}, gain = {energy_gain:.5f}, fidelity = {F:.3f}")

            ## Record
            record['optimal_time'].append(tau)
            record['optimal_time_energy'].append(E)
            record['optimal_time_fidelity'].append(F)

            ## Compute the optimal time for the next step
            tau, energy_gain, energy = compute_optimal_time(psi, H, contract_tree=opt)

        print("Optimal time steps simulation completed.")
        print('---------------')


    for i, tau in enumerate(step_sizes):
        if tau not in record['step_sizes']:
            record['step_sizes'].append(tau)
            record['fixed_sizes_energy'].append([])
            record['fixed_sizes_fidelity'].append([])
        else:
            continue

        print(f"Simulating with fixed tau = {tau:.4f}")

        psi = psi_init.copy()
        E, V = get_energy_and_variance(psi, H)
        F = overlap_fn(exact_state, psi)
        record['fixed_sizes_energy'][-1].append(E)
        record['fixed_sizes_fidelity'][-1].append(F)

        for step in tqdm(range(simulate_steps), desc="Iterations"):
            psi = qtn.tensor_1d_compress.mps_gate_with_mpo_dm(psi, linear_ite_mpo(H, tau), max_bond=max_bond)
            psi.normalize()
            E,V = get_energy_and_variance(psi, H)
            F = overlap_fn(exact_state, psi)
            print(f"E = {E:.3f}, F = {F:.3f}")
            record['fixed_sizes_energy'][-1].append(E)
            record['fixed_sizes_fidelity'][-1].append(F)

            # Break if E is too large (divergence)
            if E > max(0, E_init + 2):
                break

        with open(filename, 'wb') as f:
            pickle.dump(record, f)
        print('---------------')



    with open(filename, 'wb') as f:
        pickle.dump(record, f)

