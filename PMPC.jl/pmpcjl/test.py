import os
import time
from pathlib import Path
import ctypes

import numpy as np
import matplotlib.pyplot as plt

for path in os.listdir("lib"):
    path = Path("lib") / path
    if path.is_file():
        ctypes.cdll.LoadLibrary(str(path))

import pmpcjl

####################################################################################################
####################################################################################################
####################################################################################################


def lqp_solve(
    x0,
    f,
    fx,
    fu,
    X_prev,
    U_prev,
    Q,
    R,
    X_ref,
    U_ref,
    lx,
    ux,
    lu,
    uu,
    reg_x,
    reg_u,
    slew_reg,
    slew_reg0,
    slew_um1,
    verbose=False,
):
    assert x0.shape == (xdim, M)
    assert f.shape == (xdim, N, M)
    assert fx.shape == (xdim, xdim, N, M)
    assert fu.shape == (xdim, udim, N, M)
    assert X_prev.shape == (xdim, N, M)
    assert U_prev.shape == (udim, N, M)
    assert Q.shape == (xdim, xdim, N, M)
    assert R.shape == (udim, udim, N, M)
    assert X_ref.shape == (xdim, N, M)
    assert U_ref.shape == (udim, N, M)
    assert lx.shape == (xdim, N, M)
    assert ux.shape == (xdim, N, M)
    assert lu.shape == (udim, N, M)
    assert uu.shape == (udim, N, M)
    assert slew_um1.shape == (udim, M)

    x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref, slew_um1 = [
        np.asfortranarray(z) for z in [x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref, slew_um1]
    ]
    X, U = pmpcjl.lqp_solve(
        xdim,
        udim,
        N,
        M,
        Nc,
        x0,
        f,
        fx,
        fu,
        X_prev,
        U_prev,
        Q,
        R,
        X_ref,
        U_ref,
        lx,
        ux,
        lu,
        uu,
        reg_x,
        reg_u,
        slew_reg,
        slew_reg0,
        slew_um1,
        int(verbose),
    )
    X, U = np.reshape(X, (M, N, xdim)), np.reshape(U, (M, N, udim))
    return X, U


def lcone_solve(
    x0,
    f,
    fx,
    fu,
    X_prev,
    U_prev,
    Q,
    R,
    X_ref,
    U_ref,
    lx,
    ux,
    lu,
    uu,
    reg_x,
    reg_u,
    slew_reg,
    slew_reg0,
    slew_um1,
    verbose=False,
):
    assert x0.shape == (xdim, M)
    assert f.shape == (xdim, N, M)
    assert fx.shape == (xdim, xdim, N, M)
    assert fu.shape == (xdim, udim, N, M)
    assert X_prev.shape == (xdim, N, M)
    assert U_prev.shape == (udim, N, M)
    assert Q.shape == (xdim, xdim, N, M)
    assert R.shape == (udim, udim, N, M)
    assert X_ref.shape == (xdim, N, M)
    assert U_ref.shape == (udim, N, M)
    assert lx.shape == (xdim, N, M)
    assert ux.shape == (xdim, N, M)
    assert lu.shape == (udim, N, M)
    assert uu.shape == (udim, N, M)
    assert slew_um1.shape == (udim, M)

    x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref, slew_um1 = [
        np.asfortranarray(z) for z in [x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref, slew_um1]
    ]
    X, U = pmpcjl.lcone_solve(
        xdim,
        udim,
        N,
        M,
        Nc,
        x0,
        f,
        fx,
        fu,
        X_prev,
        U_prev,
        Q,
        R,
        X_ref,
        U_ref,
        lx,
        ux,
        lu,
        uu,
        reg_x,
        reg_u,
        slew_reg,
        slew_reg0,
        slew_um1,
        int(verbose),
        1e1,
    )
    X, U = np.reshape(X, (M, N, xdim)), np.reshape(U, (M, N, udim))
    return X, U


####################################################################################################
####################################################################################################
####################################################################################################

if __name__ == "__main__":
    M = 1
    N, xdim, udim = 30, 2, 1
    Nc = 3
    x0 = np.array([5.0, 5.0])
    fx = np.stack([np.array([[1.0, 0.1], [0.0, 1.0]]) for _ in range(N)], axis=-1)
    fu = np.stack([np.array([[0.0], [1.0]]) for _ in range(N)], axis=-1)
    f = np.stack([fx[:, :, i] @ x0 if i == 0 else np.zeros(xdim) for i in range(N)], axis=-1)
    X_prev, U_prev = np.zeros((xdim, N, M)), np.zeros((udim, N, M))
    X_ref, U_ref = np.zeros((xdim, N, M)), np.zeros((udim, N, M))
    Q = np.stack(
        [np.diag([1e0, 1e0] if i == N - 1 else [1e0, 1.0]) for i in range(N)],
        axis=-1,
    )
    R = np.stack([np.diag([1e0]) for _ in range(N)], axis=-1)
    u_limit = 0.4
    lu, uu = -u_limit * np.ones((udim, N, M)), u_limit * np.ones((udim, N, M))
    x_limit = 20.0
    lx, ux = -x_limit * np.ones((xdim, N, M)), x_limit * np.ones((xdim, N, M))

    x0 = np.expand_dims(x0, axis=-1)
    f = np.expand_dims(f, axis=-1)
    fx = np.ascontiguousarray(np.expand_dims(fx, axis=-1))
    fu = np.expand_dims(fu, axis=-1)
    Q = np.expand_dims(Q, axis=-1)
    R = np.expand_dims(R, axis=-1)

    reg_x, reg_u = 1e0, 1e-1

    slew_reg = 1.0 * np.ones((M,))
    slew_reg0 = 0.0 * np.ones((M,))
    slew_um1 = np.zeros((udim, M))

    smooth_alpha = 1e0

    args = (
        x0,
        f,
        fx,
        fu,
        X_prev,
        U_prev,
        Q,
        R,
        X_ref,
        U_ref,
        lx,
        ux,
        lu,
        uu,
        reg_x,
        reg_u,
        slew_reg,
        slew_reg0,
        slew_um1,
    )

    t = time.time()
    lqp_solve(*args, verbose=False)
    lcone_solve(*args, verbose=False)
    t = time.time() - t
    print(f"First compute time:  {t:.4e} s")

    t = time.time()
    lqp_solve(*args, verbose=False)
    lcone_solve(*args, verbose=False)
    t = time.time() - t
    print(f"Second compute time: {t:.4e} s")

    ###################################################################

    X, U = lcone_solve(*args, verbose=False)
    plt.figure()
    plt.plot(X[0, :, 0], label="x1")
    plt.plot(X[0, :, 1], label="x2")
    plt.legend()

    plt.figure()
    plt.plot(U[0, :, 0], label="u1")
    plt.legend()

    X, U = lqp_solve(*args, verbose=False)
    plt.figure()
    plt.plot(X[0, :, 0], label="x1")
    plt.plot(X[0, :, 1], label="x2")
    plt.legend()

    plt.figure()
    plt.plot(U[0, :, 0], label="u1")
    plt.legend()

    #plt.show()

    # print(X.shape)
    # print(X)
    # print(X.reshape((N + 1, xdim)).T)
    # print("#" * 80)

    # X = np.reshape(X, (xdim, N + 1, M))
    # U = np.reshape(U, (udim, N, M))

    # plt.figure()
    # plt.plot(X[:, 0, 0], label="x1")
    # plt.plot(X[:, 1, 0], label="x2")
    # plt.legend()

    # plt.figure()
    # plt.plot(U[:, 0, 0], label="u1")
    # plt.legend()

    ## plt.show()

    # print(X[:, :, 0])

    # pdb.set_trace()
