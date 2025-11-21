import numpy as np

def omp(A, y, epsilon, max_iterations=np.inf):
    """
    Orthogonal Matching Pursuit (OMP) algorithm for sparse signal recovery.
    
    Parameters:
    D : np.ndarray
        The sensing matrix (size: m x n).
    y : np.ndarray
        The observed vector (size: m x 1).
    epsilon: float
        The error tolerance for stopping criterion.
    max_iterations : int
        Maximum number of iterations to perform.
        
    Returns:
    x_hat : np.ndarray
        The recovered sparse signal (size: n x 1).
    """
    m, n = A.shape
    x_hat = np.zeros((n, 1))
    residual = y.copy()
    index_set = []
    max_iterations = min(max_iterations, n)
    err = np.inf
    while err > epsilon and len(index_set) < max_iterations:
        # Step 1: Find the index of the atom that best correlates with the residual
        correlations = A.T @ residual
        correlations[index_set] = 0
        best_index = np.argmax(np.abs(correlations))
        index_set.append(best_index)

        # Step 2: Solve the least squares problem to update the coefficients
        A_subset = A[:, index_set]
        x_subset, _, _, _ = np.linalg.lstsq(A_subset, y, rcond=None)

        # Step 3: Update the residual
        residual = y - A_subset @ x_subset
        err = np.linalg.norm(residual)

    # Step 4: Construct the full solution vector
    for i, idx in enumerate(index_set):
        x_hat[idx] = x_subset[i]

    return x_hat


def psomp(A, y, K, sigma2=None):
    """
    Paired-Support Orthogonal Matching Pursuit (PSOMP)
    Based on Algorithm 1 in Masoumi & Myers (2023).

    Inputs:
        A      : sensing matrix (M x N)
        y      : measurement vector (M,)
        K      : sparsity level of x
        sigma2 : noise variance (optional for stopping rule)

    Outputs:
        z_hat  : estimated augmented sparse vector (2N,)
    """

    M, N = A.shape

    # Build augmented matrix
    A_aug = np.hstack([A, np.conjugate(A)])

    # Initialize
    r = y.copy()
    Q = []     # support set
    z_hat = np.zeros(2*N, dtype=complex)

    max_iter = 2*K
    err = np.inf
    while len(Q) < max_iter and (sigma2 is None or err > sigma2):

        # --- Step 1: support detection (paired) ---
        # Compute both matching terms
        match1 = np.abs(np.conjugate(A).T @ r)     # |a_j^* r|
        match2 = np.abs(A.T @ r)                   # |a_j^T r|

        # Choose best index from first N entries
        j = np.argmax(np.maximum(match1[:N], match2[:N]))

        # Paired support structure
        pair = [j, j+N]
        Q.extend(pair)

        # --- Step 2: least squares on selected support ---
        A_sub = A_aug[:, Q]
        z_sub, *_ = np.linalg.lstsq(A_sub, y, rcond=None)

        # assign
        for ii, idx in enumerate(Q):
            z_hat[idx] = z_sub[ii]

        # --- Step 3: update residual ---
        r = y - A_sub @ z_sub


    return z_hat

def find_x_xi(z : np.ndarray):
    """
    Function to recover the original signal and IQ imbalance parameter from the IQ imbalanced signal.
    Parameters:
        z : np.ndarray
            IQ imbalanced signal (size: 2n x 1).
            
        Returns:
        x : np.ndarray
            Recovered original signal (size: n x 1).
        xi : float
            Estimated IQ imbalance parameter.
    """
    z_1,z_2 = np.split(z, 2)
    alpha = np.linalg.norm(z_1)**2
    beta = np.linalg.norm(z_2)**2
    gamma = z_1.T @ z_2
    xi_hat = (alpha - beta - 2*gamma + np.sqrt( (alpha - beta)**2 + 4*np.abs(gamma)**2))/(2*(alpha - beta + np.conj(gamma) - gamma))
    x_hat = z_1/xi_hat
    return x_hat.reshape(-1,1), xi_hat