import pickle, sys, time

import matplotlib.pyplot as plt, numpy as np, scipy.sparse as sp
import cvxpy as cp, scipy.sparse.linalg

import camelid.controllers.pmpc as pmpc

np.set_printoptions(precision=2, linewidth=200)

if __name__ == "__main__":
    np.random.seed(2020)

    xdim, udim = 2, 1
    fx = np.array([[1, 0.1], [0, 1]])
    fu = np.array([[0], [1]])
    Q = np.eye(2)
    R = np.eye(1)
    x0 = np.array([1, 1])
    u0 = np.zeros(1)

    X_ref = np.array([0, 0])
    U_ref = np.array([0])
    X_prev = x0
    U_prev = u0
    x_l, x_u, u_l, u_u = None, None, None, None

    M, N = 1, 1000

    # f = np.tile(f[None, None, ...], (M, N, 1))
    fx = np.tile(fx[None, None, ...], (M, N, 1, 1))
    fx[:, :, 0, 1] = ((np.random.rand(M) + 0.1) / 1.1 / 2.0).reshape((M, 1))
    # print(fx[:, 0, 0, 1])
    fu = np.tile(fu[None, None, ...], (M, N, 1, 1))
    f = fx @ x0 + fu @ u0
    Q = np.tile(Q[None, None, ...], (M, N, 1, 1))
    R = np.tile(R[None, None, ...], (M, N, 1, 1))
    x0 = np.tile(x0[None, ...], (M, 1))
    X_ref = np.tile(X_ref[None, None, ...], (M, N, 1))
    U_ref = np.tile(U_ref[None, None, ...], (M, N, 1))
    X_prev = np.tile(X_prev[None, None, ...], (M, N, 1))
    U_prev = np.tile(U_prev[None, None, ...], (M, N, 1))

    P, q, A, b, G, l, u = pmpc.to_qp_csc(
        f, fx, fu, x0, X_prev, U_prev, Q, R, X_ref, U_ref, 0, 0
    )
    P, q, A, b, G, l, u = P[0], q[0], A[0], b[0], G[0], l[0], u[0]

    Aa = sp.hstack(
        [
            sp.vstack([P, A], format="csc"),
            sp.vstack(
                [A.transpose(), sp.csc_matrix(2 * (b.size,))], format="csc"
            ),
        ],
        format="csc",
    ) + 1e-9 * sp.eye(q.size + b.size)
    # print(np.linalg.eigvals(Aa.toarray()))
    # print(np.linalg.cond(Aa.toarray()))
    ba = np.concatenate([-q, b])

    t = time.time()
    x1 = sp.linalg.cg(Aa, ba, tol=1e-2)[0]
    print("CG elapsed %e s" % (time.time() - t))
    # print(x1)
    t = time.time()
    x2 = sp.linalg.spsolve(Aa, ba)
    print("SPLU elapsed %e s" % (time.time() - t))
    # print(x2)
    print(np.linalg.norm(x2 - x1))
    sys.exit()

    x = cp.Variable(N * (xdim + udim))
    print("P.shape =", P.shape)
    print("q.shape =", q.shape)
    print("A.shape =", A.shape)
    print("b.shape =", b.shape)
    print("G.shape =", G.shape)
    print("l.shape =", l.shape)
    print("u.shape =", u.shape)
    print("x.shape =", x.shape)
    cstr = [A @ x == b]
    if G.shape[-2] > 0:
        cstr += [G @ x <= u, G @ x >= l]
    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, P) + cp.matmul(q.T, x)), cstr
    )
    prob.solve(solver=cp.GUROBI, verbose=True)

    x_value = x.value
    x = np.concatenate(
        [x0[0, None, :], x_value[N * udim :].reshape((-1, xdim))], -2
    )
    u = x_value[: N * udim].reshape((-1, udim))

    plt.figure(99)
    plt.plot(x[:, 0], label="x0")
    plt.plot(x[:, 1], label="x1")
    plt.plot(u[:, 0], label="u0")
    plt.legend()
    # plt.show()

    """
    t = time.time()
    ret = pmpc.to_qp_csc(f, fx, fu, x0, X_prev, U_prev, Q, R, X_ref, U_ref, 0, 0)
    print(fu)
    print(time.time() - t)
    #print(P)

    #P = sp.csc_matrix((ret[0][0][0], ret[0][1][0], ret[0][2][0]), shape=ret[0][3])
    #A = sp.csc_matrix((ret[2][0][0], ret[2][1][0], ret[2][2][0]), shape=ret[2][3])
    #G = sp.csc_matrix((ret[4][0][0], ret[3][1][0], ret[4][2][0]), shape=ret[4][3])
    """

    """
    f, fu, fx, x0, X_prev, U_prev, Q, R, X_ref, U_ref = [np.copy(x, order="C") 
            for x in [f, fu, fx, x0, X_prev, U_prev, Q, R, X_ref, U_ref]]
    """

    ########################################################################
    for i in range(1):
        t = time.time()
        # X_traj, U_traj, viol = pmpc.admm_solve(f, fx, fu, x0, X_prev, U_prev,
        #        Q, R, X_ref, U_ref, 5e1, -1, 25, 0, 0, False)
        print("Actual time spent is %e" % (time.time() - t))
        t = time.time()
        X_traj, U_traj, viol = pmpc.large_qp_solve(
            f, fx, fu, x0, X_prev, U_prev, Q, 2 * R, X_ref, U_ref, -1, 0, 0
        )
        print("Actual time spent constructing is %e" % (time.time() - t))
        # X_traj, U_traj = ret

    plt.figure()
    plt.plot(X_traj[0][:, 0], label="x_1")
    plt.plot(X_traj[0][:, 1], label="x_2")
    plt.plot(U_traj[0][:, 0], label="u_1")
    plt.legend()

    plt.figure()
    for i in range(min(M, 50)):
        plt.plot(U_traj[i][:, 0], label="u_" + str(i))
    plt.legend()

    plt.show()
