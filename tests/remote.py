import sys, os, sys

import matplotlib.pyplot as plt, numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import pmpc
#from pmpc.remote import solve
from pmpc.remote import call
solve = lambda *args, **kw: call("solve", None, 5678, True, *args, **kw)

from dubins_car import f_np as f_fn, fx_np as fx_fn, fu_np as fu_fn


def f_fx_fu_fn(X_prev, U_prev):
    x, u, p = X_prev, U_prev, np.array([1.0, 1.0, 0.3])
    return f_fn(x, u, p), fx_fn(x, u, p), fu_fn(x, u, p)


if __name__ == "__main__":
    M, N, xdim, udim = 1, 30, 4, 2

    # Q = np.tile(np.eye(xdim), (M, N, 1, 1))
    # R = np.tile(1e-2 * np.eye(udim), (M, N, 1, 1))
    # x0 = np.tile(np.ones(xdim), (M, 1))
    # X_ref, U_ref = np.zeros((M, N, xdim)), np.zeros((M, N, udim))
    # X_prev, U_prev = np.zeros((M, N, xdim)), np.zeros((M, N, udim))
    # u_l, u_u = -1 * np.ones((M, N, udim)), 1 * np.ones((M, N, udim))

    Q = np.tile(np.eye(xdim), (N, 1, 1))
    R = np.tile(1e-2 * np.eye(udim), (N, 1, 1))
    x0 = np.tile(np.ones(xdim), (1,))
    X_ref, U_ref = np.zeros((N, xdim)), np.zeros((N, udim))
    X_prev, U_prev = np.zeros((N, xdim)), np.zeros((N, udim))
    u_l, u_u = -1 * np.ones((N, udim)), 1 * np.ones((N, udim))

    opts = dict(verbose=True, u_l=u_l, u_u=u_u)
    args = (f_fx_fu_fn, Q, R, x0, X_ref, U_ref, X_prev, U_prev)

    opts["reg_x"], opts["reg_u"] = 1e0, 1e0
    # ret = pmpc.tune_scp(*args, **opts)
    # opts["reg_x"], opts["reg_u"] = ret
    X, U, data = solve(*args, max_it=100, **opts)

    # ret = pmpc.tune_scp(*args, solve_fn=pmpc.accelerated_scp_solve, **opts)
    # opts["rho_res_x"], opts["rho_res_u"] = ret
    # X, U, data = pmpc.accelerated_scp_solve(*args, max_it=100, **opts)
    # X, U = X[0], U[0]

    plt.figure()
    for r in range(xdim):
        plt.plot(X[:, r], label="$x_%d$" % (r + 1))
    plt.legend()
    plt.tight_layout()

    plt.figure()
    for r in range(udim):
        plt.plot(U[:, r], label="$u_%d$" % (r + 1))
    plt.legend()
    plt.tight_layout()

    #plt.draw_all()
    #plt.pause(1e-1)
    plt.show()
