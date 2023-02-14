import pickle, sys, time, pdb, os, sys

import matplotlib.pyplot as plt, numpy as np, scipy.sparse as sp
import cvxpy as cp, scipy.sparse.linalg

dirname = os.path.abspath(os.path.dirname(__file__))
sys.path = [os.path.join(dirname, "..")] + sys.path
import pmpc

dirname = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(dirname, "..", ".."))
from dubins_car import f_np as f_fn, fx_np as fx_fn, fu_np as fu_fn


def f_fx_fu_fn(X_prev, U_prev):
    x, u, p = X_prev, U_prev, np.array([1.0, 1.0, 0.3])
    return f_fn(x, u, p), fx_fn(x, u, p), fu_fn(x, u, p)


if __name__ == "__main__":
    M, N, xdim, udim = 1, 30, 4, 2

    Q = np.tile(np.eye(xdim), (N, 1, 1))
    R = np.tile(1e-2 * np.eye(udim), (N, 1, 1))
    x0 = np.tile(np.ones(xdim), (1,))
    X_ref, U_ref = np.zeros((N, xdim)), np.zeros((N, udim))
    X_prev, U_prev = np.zeros((N, xdim)), np.zeros((N, udim))
    u_l, u_u = -1 * np.ones((N, udim)), 1 * np.ones((N, udim))

    opts = dict(verbose=True, u_l=u_l, u_u=u_u)
    args = (f_fx_fu_fn, Q, R, x0, X_ref, U_ref, X_prev, U_prev)

    for solver in ["ecos", "osqp", "cosmo"]:
        #ret = pmpc.tune_scp(*args, **opts)
        #opts["rho_res_x"], opts["rho_res_u"] = ret
        opts["solver_settings"] = dict(solver=solver)
        X, U, data = pmpc.solve(*args, max_it=100, **opts)
        #X, U = X[0], U[0]

        #ret = pmpc.tune_scp(*args, solve_fn=pmpc.accelerated_scp_solve, **opts)
        #opts["rho_res_x"], opts["rho_res_u"] = ret
        #X, U, data = pmpc.accelerated_scp_solve(*args, max_iters=100, **opts)
        #X, U = X[0], U[0]

        plt.figure()
        plt.title(solver)
        for r in range(xdim):
            plt.plot(X[:, r], label="$x_%d$" % (r + 1))
        plt.legend()
        plt.tight_layout()

        plt.figure()
        plt.title(solver)
        for r in range(udim):
            plt.plot(U[:, r], label="$u_%d$" % (r + 1))
        plt.legend()
        plt.tight_layout()

    plt.show()
