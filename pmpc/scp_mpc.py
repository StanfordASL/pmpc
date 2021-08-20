##^# library imports ###########################################################
import math, os, pdb, time

import matplotlib.pyplot as plt, numpy as np

from . import julia_utils as ju

jl = None


def ensure_julia():
    global jl
    if jl is None:
        jl = ju.load_julia()


##$#############################################################################
##^# affine solve using julia ##################################################
def atleast_nd(x, n):
    return x.reshape((1,) * max(n - x.ndim, 0) + x.shape)


def aff_solve(
    f,
    fx,
    fu,
    x0,
    X_prev,
    U_prev,
    Q,
    R,
    X_ref,
    U_ref,
    rho_res_x,
    rho_res_u,
    slew_rate,
    u_slew,
    x_l,
    x_u,
    u_l,
    u_u,
    method="admm",
    solver_settings={},
):
    ensure_julia()
    f = atleast_nd(f, 3)
    fx, fu = atleast_nd(fx, 4), atleast_nd(fu, 4)
    x0 = atleast_nd(x0, 2)
    X_prev, U_prev = atleast_nd(X_prev, 3), atleast_nd(U_prev, 3)
    Q, R = atleast_nd(Q, 4), atleast_nd(R, 4)
    X_ref, U_ref = atleast_nd(X_ref, 3), atleast_nd(U_ref, 3)
    x_l, x_u, u_l, u_u = [atleast_nd(z, 3) for z in [x_l, x_u, u_l, u_u]]

    t = time.time()
    x_l, x_u, u_l, u_u = [ju.py2jl(z, 1) for z in [x_l, x_u, u_l, u_u]]
    args = [
        ju.py2jl(z, d)
        for (z, d) in zip(
            [x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref],
            [1, 1, 2, 2, 1, 1, 2, 2, 1, 1],
        )
    ]
    if method == "lqp":
        solve_fn = jl.lqp_solve
    elif method == "admm":
        solve_fn = jl.admm_solve
    elif method == "socp":
        solve_fn = jl.lsocp_solve
    else:
        raise ValueError("No method [%s] found" % method)
    ret = solve_fn(
        *args,
        rho_res_x=rho_res_x,
        rho_res_u=rho_res_u,
        lx=x_l,
        ux=x_u,
        lu=u_l,
        uu=u_u,
        **solver_settings,
    )
    X, U = [ju.jl2py(z, d) for (z, d) in zip(ret[:2], [1, 1])]
    X_traj, U_traj = np.concatenate([x0[:, None, :], X], -2), U
    return X_traj, U_traj, ret[2]


##$#############################################################################
##^# cost augmentation #########################################################
def augment_cost(cost_fn, X_prev, U_prev, Q, R, X_ref, U_ref):
    if cost_fn is not None:
        Cxx, cx, Cuu, cu = cost_fn(X_prev, U_prev)
        # augment the state cost #############
        Qp = (Q + Cxx) if Cxx is not None else Q
        xdim = Qp.shape[-1]
        ev = np.linalg.eigvals(Qp)
        reg = (
            -np.minimum(np.min(ev, -1), 0.0)[..., None, None] + 1e-5
        ) * np.eye(xdim)
        Qp = Qp + reg
        if cx is not None:
            X_ref = np.linalg.solve(Qp, Q @ X_ref[..., None] - cx[..., None])[
                ..., 0
            ]
        Q = Qp
        # augment the control cost ###########
        Rp = (R + Cuu) if Cuu is not None else R
        udim = Rp.shape[-1]
        ev = np.linalg.eigvals(Rp)
        reg = (
            -np.minimum(np.min(ev, -1), 0.0)[..., None, None] + 1e-5
        ) * np.eye(udim)
        Rp = Rp + reg
        if cu is not None:
            U_ref = np.linalg.solve(Rp, R @ U_ref[..., None] - cu[..., None])[
                ..., 0
            ]
        R = Rp
    return Q, R, X_ref, U_ref


##$#############################################################################
##^# SCP MPC ###################################################################
norm = lambda x, p=None, dim=None: np.linalg.norm(x, p, dim)
bmv = lambda A, x: (A @ x[..., None])[..., 0]

SCP_TABLE_HEADER = (
    ("+" + "-" * 6)
    + ("+" + "-" * 11) * 5
    + "+"
    + "\n"
    + "|  it. |   elaps.  |    obj.   |   resid.  | rho_res_x | rho_res_u |"
    + "\n"
    + ("+" + "-" * 6)
    + ("+" + "-" * 11) * 5
    + "+"
)
SCP_TABLE_FOOTER = ("+" + "-" * 6) + ("+" + "-" * 11) * 5 + "+"


def scp_solve(
    f_fx_fu_fn,
    Q,
    R,
    x0,
    X_ref=None,
    U_ref=None,
    X_prev=None,
    U_prev=None,
    x_l=None,
    x_u=None,
    u_l=None,
    u_u=None,
    verbose=True,
    debug=False,
    max_iters=100,
    time_limit=1000.0,
    res_tol=1e-5,
    rho_res_x=1e0,
    rho_res_u=1e-2,
    slew_rate=0.0,
    u_slew=None,
    cost_fn=None,
    method="admm",
    solver_settings={},
    solver_state=None,
    **kwargs,
):
    t_elaps = time.time()

    # create variables and reference trajectories ##############################
    Q, R = np.copy(Q), np.copy(R)
    assert x0.ndim == 2 and R.ndim == 4 and Q.ndim == 4
    M, N, xdim, udim = Q.shape[:3] + R.shape[-1:]

    X_ref = np.zeros((M, N, xdim)) if X_ref is None else X_ref
    U_ref = np.zeros((M, N, udim)) if U_ref is None else U_ref
    X_prev = X_prev if X_prev is not None else X_ref
    U_prev = U_prev if U_prev is not None else U_ref
    X_prev, U_prev = X_prev.reshape((M, N, xdim)), U_prev.reshape((M, N, udim))
    X_ref, U_ref = X_ref.reshape((M, N, xdim)), U_ref.reshape((M, N, udim))
    x_l, x_u, u_l, u_u = [
        z if z is not None else np.zeros((0, 0, 0))
        for z in [x_l, x_u, u_l, u_u]
    ]
    data = dict(solver_data=[], hist=[])

    # solve sequentially, linearizing ##########################################
    if verbose:
        # print(("+" + "-" * 6) + ("+" + "-" * 11) * 5 + "+")
        # print(
        #    "|  it. |   elaps.  |    obj.   |   resid.  |"
        #    + " rho_res_x | rho_res_u |"
        # )
        # print(("+" + "-" * 6) + ("+" + "-" * 11) * 5 + "+")
        print(SCP_TABLE_HEADER)
    it = 0
    X, U, solver_data = None, None, None
    while it < max_iters:
        X_ = np.concatenate([x0[..., None, :], X_prev[..., :-1, :]], -2)
        t = time.time()
        f, fx, fu = f_fx_fu_fn(X_, U_prev)
        # print("Linearization takes %e s" % (time.time() - t))
        f = f.reshape((M, N, xdim))
        fx = fx.reshape((M, N, xdim, xdim))
        fu = fu.reshape((M, N, xdim, udim))

        Q_, R_, X_ref_, U_ref_ = augment_cost(
            cost_fn, X_prev, U_prev, Q, R, X_ref, U_ref
        )
        # t = time.time()
        # if debug:
        #    print("f.norm() =  %9.4e" % norm(f.reshape(-1)))
        #    print("fx.norm() = %9.4e" % norm(fx.reshape(-1)))
        #    print("fu.norm() = %9.4e" % norm(fu.reshape(-1)))
        #    if norm(fx.reshape(-1)) > 1e2:
        #        print(np.mean(fx[0], -3))
        t_aff_solve = time.time()
        X, U, solver_data = aff_solve(
            f,
            fx,
            fu,
            x0,
            X_prev,
            U_prev,
            Q_,
            R_,
            X_ref_,
            U_ref_,
            rho_res_x,
            rho_res_u,
            slew_rate,
            u_slew,
            x_l,
            x_u,
            u_l,
            u_u,
            method=method,
            solver_settings=dict(solver_settings, solver_state=solver_state),
        )
        t_aff_solve = time.time() - t_aff_solve
        solver_state = solver_data.get("solver_state", None)
        X, U = X.reshape((M, N + 1, xdim)), U.reshape((M, N, udim))

        if debug:
            # plt.figure(3400 + it)
            plt.figure(345789453)
            plt.clf()
            for i in range(X.shape[-1]):
                plt.plot(X[0, :, i], label="x" + str(i), alpha=0.5)
            plt.title("State, it %03d" % it)
            plt.legend()
            plt.tight_layout()

            # plt.figure(6400 + it)
            plt.figure(4389423733)
            plt.clf()
            for i in range(U.shape[-1]):
                plt.plot(U[0, :, i], label="u" + str(i), alpha=0.5)
            plt.title("Control, it %03d" % it)
            plt.legend()
            plt.tight_layout()

            plt.draw_all()
            plt.pause(1e-2)
        # print("Solving took =", time.time() - t)
        if np.any(np.isnan(X)) or np.any(np.isnan(U)):
            if verbose:
                print("Solver failed...")
            return None, None, None
        X_ = X[..., 1:, :]
        dX, dU = X_ - X_prev, U - U_prev
        max_res = max(np.max(norm(dX, 2, -1)), np.max(norm(dU, 2, -1)))
        dX, dU = X_ - X_ref, U - U_ref
        obj = (np.sum(dX * bmv(Q, dX)) + np.sum(dU * bmv(R, dU))) / N / M

        X_prev, U_prev = X[..., 1:, :], U

        t_run = time.time() - t_elaps
        vals = (it + 1, t_run, obj, max_res, rho_res_x, rho_res_u)
        if verbose:
            print("| %04d | %6.3e | %6.3e | %6.3e | %6.3e | %6.3e |" % vals)
        data["solver_data"].append(solver_data)
        data["hist"].append(
            {
                k: val
                for (k, val) in zip(
                    ["it", "t_run", "obj", "max_res", "rho_res_x", "rho_res_u"],
                    vals,
                )
            }
        )
        data.setdefault("t_aff_solve", [])
        data["t_aff_solve"].append(t_aff_solve)

        if max_res < res_tol:
            break
        it += 1
        if (time.time() - t_elaps) * (it + 1) / it > time_limit:
            break

    if verbose:
        # print(("+" + "-" * 6) + ("+" + "-" * 11) * 5 + "+")
        print(SCP_TABLE_FOOTER)
    if verbose and max_res > 1e-3:
        print("#" * 76)
        print(
            "Bad solution found, the solution is approximate to a "
            + "residual: %5.3e" % max_res
        )
        print("#" * 76)
    return X.reshape((M, N + 1, xdim)), U.reshape((M, N, udim)), data


solve = scp_solve
##$#############################################################################
##^# tuning hyperparameters ####################################################
def tune_scp(*args, **kwargs):
    rho_res_ratio = kwargs.get("rho_res_ratio", 1e-1)

    rho_res_list = kwargs.get("rho_rng", np.logspace(-3, 3, 14))
    max_res_list = []
    for rho_res in rho_res_list:
        rho_res_x, rho_res_u = rho_res, rho_res * rho_res_ratio
        kwargs["rho_res_x"], kwargs["rho_res_u"] = rho_res_x, rho_res_u
        X, U, data = scp_solve(*args, **kwargs)
        max_res_list.append(
            [z["max_res"] for z in data["hist"]] if data is not None else [1e1]
        )

    plt.figure(23423345)
    plt.clf()
    # for (i, max_res) in enumerate(max_res_list):
    #    plt.semilogy(max_res, label="%5.1e" % rho_res_list[i])
    # plt.legend()
    plt.loglog(rho_res_list, [z[-1] for z in max_res_list])
    plt.draw_all()
    plt.pause(1e-2)
    pdb.set_trace()

    return


##$#############################################################################
##^# accelerated SCP ###########################################################
momentum_update = lambda zk, zkm1, it: zk + it / (it + 3) * (zk - zkm1)


def accelerated_scp_solve(
    f_fx_fu_fn,
    Q,
    R,
    x0,
    X_ref=None,
    U_ref=None,
    X_prev=None,
    U_prev=None,
    x_l=None,
    x_u=None,
    u_l=None,
    u_u=None,
    verbose=True,
    debug=False,
    max_iters=100,
    time_limit=1000.0,
    res_tol=1e-5,
    rho_res_x=1e0,
    rho_res_u=1e-2,
    slew_rate=0.0,
    u_slew=None,
    cost_fn=None,
    method="admm",
    solver_settings={},
    solver_state=None,
    **kwargs,
):
    # initialize the SCP variables and reference trajectory ##
    assert x0.ndim == 2 and R.ndim == 4 and Q.ndim == 4
    M, N, xdim, udim = Q.shape[:3] + R.shape[-1:]
    X_ref = np.zeros((M, N, xdim)) if X_ref is None else X_ref
    U_ref = np.zeros((M, N, udim)) if U_ref is None else U_ref
    X_prev = X_prev if X_prev is not None else X_ref
    U_prev = U_prev if U_prev is not None else U_ref
    X_prev, U_prev = X_prev.reshape((M, N, xdim)), U_prev.reshape((M, N, udim))
    X_ref, U_ref = X_ref.reshape((M, N, xdim)), U_ref.reshape((M, N, udim))

    # initialize the Nesterov momentum history ###############
    X_prev_2hist = [X_prev, X_prev]
    U_prev_2hist = [U_prev, U_prev]

    t_start = time.time()
    data = {}
    if verbose:
        print(SCP_TABLE_HEADER)
    for it in range(max_iters):
        X_prev = momentum_update(X_prev_2hist[-1], X_prev_2hist[-2], it)
        U_prev = momentum_update(U_prev_2hist[-1], U_prev_2hist[-2], it)

        X, U, data_ = scp_solve(
            f_fx_fu_fn,
            Q,
            R,
            x0,
            X_ref=X_ref,
            U_ref=U_ref,
            X_prev=X_prev,
            U_prev=U_prev,
            x_l=x_l,
            x_u=x_u,
            u_l=u_l,
            u_u=u_u,
            verbose=False,
            debug=debug,
            max_iters=1,
            time_limit=math.inf,
            res_tol=0.0,
            rho_res_x=rho_res_x,
            rho_res_u=rho_res_u,
            slew_rate=slew_rate,
            u_slew=u_slew,
            cost_fn=cost_fn,
            method=method,
            solver_settings=solver_settings,
            solver_state=solver_state,
            **kwargs,
        )

        X_prev_2hist = [X_prev_2hist[-1], X[..., 1:, :]]
        U_prev_2hist = [U_prev_2hist[-1], U]

        solver_state = data_.get("solver_data", [{}])[-1].get(
            "solver_state", None
        )
        for k in data_.keys():
            data.setdefault(k, [])
            data[k].extend(data_[k])
        if verbose:
            vals = [it + 1, time.time() - t_start] + [
                data_["hist"][-1][k]
                for k in ["obj", "max_res", "rho_res_x", "rho_res_u"]
            ]
            print(
                "| %04d | %6.3e | %6.3e | %6.3e | %6.3e | %6.3e |" % tuple(vals)
            )

        if data["hist"][-1]["max_res"] < res_tol:
            break
        if (it + 2) / (it + 1) * (time.time() - t_start) > time_limit:
            break
    if verbose:
        print(SCP_TABLE_FOOTER)
    return X, U, data


##$#############################################################################
