import numpy as np

def dare(A, B, Q, R, max_iter=1000):
    """Solve the DARE by square-root Riccati iteration."""
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    nu = B.shape[1]
    Lp = np.linalg.cholesky(Q)
    for k in range(max_iter):
        ALp = A.T @ Lp
        BLp = B.T @ Lp
        t22 = Q + ALp @ ALp.T
        t12 = BLp @ ALp.T
        t11 = R + BLp @ BLp.T
        L = np.linalg.cholesky(np.block([
            [t11, t12],
            [t12.T, t22]
        ]))
        Lp_new = L[nu:, nu:]
        res = np.linalg.norm(Lp_new - Lp)
        Lp = Lp_new
        if res < 1e-9:
            return Lp @ Lp.T
    raise RuntimeError(f"DARE did not converge after {max_iter} iterations")


# Condensation helpers from ~/Desktop/mpc/mpc_benchmarks.py/utils.py.
def condense_input_constrained(A, B, Q, R, P, lb_u, ub_u, N, K = None):
    nx = A.shape[1]
    nu = B.shape[1]
    if K is None:
        K = np.zeros((nu, nx))
    
    Q_bar = np.kron(np.eye(N), Q)
    Q_bar[-nx:, -nx:] = P
    R_bar = np.kron(np.eye(N), R)
    
    # Input to states map
    F = np.zeros((N*nx, N*nu))
    F_bar = np.eye(N * nu)
    A2k_B = B
    K_A2k_B = K @ B
    for k in range(N):
        for i in range(k, N):
            F[i*nx : (i+1)*nx, (i-k)*nu : (i-k+1)*nu] = A2k_B
            if i > k:
                F_bar[i*nu : (i+1)*nu, (i-k-1)*nu : (i-k)*nu] = -K_A2k_B
        A2k_B = (A - B @ K) @ A2k_B
        K_A2k_B = K @ A2k_B

    # Hessian
    H = F_bar.T @ R_bar @ F_bar + F.T @ Q_bar @ F
    # Constraints matrix
    G = F_bar

    L = np.zeros((nx * N, nx))
    L_bar = np.zeros((nu * N, nx))
    A2k = np.eye(nx)
    for k in range(N):
        L_bar[k*nu : (k+1)*nu, :] = - K @ A2k
        A2k = (A - B @ K) @ A2k
        L[k*nx : (k+1)*nx, :] = A2k
    # Linear term
    g_x0 = F_bar.T @ R_bar @ L_bar + F.T @ Q_bar @ L

    # Input constraints
    lu_x0 = L_bar
    low = np.tile(lb_u, N)
    upp = np.tile(ub_u, N)

    # q.T X = q.T (L x0 + F dU) = q.T L x0 + q.T F dU
    # r.T U = r.T (L_bar x0 + F_bar dU) = r.T L_bar x0 + r.T F_bar dU
    g_q = F.T
    g_r = F_bar.T

    return H, G, g_x0, g_q, g_r, lu_x0, low, upp

def get_parametric_qp_terms(g_x0, lu_x0, low, upp, x0):
    const = lu_x0 @ x0
    return g_x0 @ x0, low - const, upp - const

