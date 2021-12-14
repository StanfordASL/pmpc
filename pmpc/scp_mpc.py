##^# library imports ###########################################################
import math, os, pdb, time
from copy import copy

import matplotlib.pyplot as plt, numpy as np
from tqdm import tqdm

from . import utils as utl
from . import julia_utils as ju

jl = None

print_fn = lambda *args, **kwargs: print(*args, **kwargs)


def minimize_AA(Fs, norm=2):
    import cvxpy as cp

    F = np.stack([f.reshape(-1) for f in Fs], -1)
    alf = cp.Variable(F.shape[-1])
    #obj = cp.sum(cp.norm(F @ cp.diag(alf), norm, -2))
    obj = cp.sum_squares(F @ cp.diag(alf))
    prob = cp.Problem(cp.Minimize(obj), [cp.sum(alf) == 1])
    prob.solve(cp.GUROBI)
    return alf.value


def ensure_julia():
    global jl

    if jl is None:
        # from julia import Julia
        # Julia(sysimage="sys.so")
        # print("using sysimaage")
        # from julia import Main as jl_
        # jl = jl_
        # jl.using("PMPC")

        # from julia.api import LibJulia

        # api = LibJulia.load()
        # api.init_julia(["--trace-compile=out.jl"])

        # from julia import Main as jl_
        # jl = jl_
        ##jl.using("PackageCompiler")
        # jl.using("PMPC")
        jl = ju.load_julia()


def load_sysimage(path):
    from julia import Julia

    global jl
    jl = Julia(sysimage=path)

    from julia import Main as jl_

    jl = jl_
    jl.using("PMPC")


def make_sysimage(path):
    ensure_julia()
    # jl.using("PackageCompiler")
    jl.eval(
        'create_sysimage(:PMPC, sysimage_path="%s", precompile_execution_file="out.jl")'
        % path
    )


##$#############################################################################
##^# affine solve using julia ##################################################
def atleast_nd(x, n):
    if x is None:
        return None
    else:
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
    reg_x,
    reg_u,
    slew_rate,
    u_slew,
    x_l,
    x_u,
    u_l,
    u_u,
    method="lqp",
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

    solver_settings = copy(solver_settings)
    if u_slew is not None:
        solver_settings["slew_um1"] = u_slew
    if slew_rate is not None:
        solver_settings["slew_reg"] = slew_rate

    ret = solve_fn(
        *args,
        reg_x=reg_x,
        reg_u=reg_u,
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
        cx, cu = cost_fn(X_prev, U_prev)

        # augment the state cost #############
        if cx is not None:
            X_ref = X_ref - np.linalg.solve(Q, cx[..., None])[..., 0]

        # augment the control cost ###########
        if cu is not None:
            U_ref = U_ref - np.linalg.solve(R, cu[..., None])[..., 0]
    return X_ref, U_ref


##$#############################################################################
##^# SCP MPC ###################################################################
norm = lambda x, p=None, dim=None: np.linalg.norm(x, p, dim)
bmv = lambda A, x: (A @ x[..., None])[..., 0]

XU2vec = lambda X, U: np.concatenate([X.reshape(-1), U.reshape(-1)])
vec2XU = lambda z, Xshape, Ushape: (
    z[: np.prod(Xshape)].reshape(Xshape),
    z[np.prod(Xshape) :].reshape(Ushape),
)


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
    verbose=False,
    debug=False,
    max_iters=100,
    time_limit=1000.0,
    res_tol=1e-5,
    reg_x=1e0,
    reg_u=1e-2,
    slew_rate=0.0,
    u_slew=None,
    cost_fn=None,
    method="lqp",
    solver_settings={},
    solver_state=None,
    AA=False,
    AA_L=20,
    AA_it=20,
):
    t_elaps = time.time()

    # create variables and reference trajectories ##############################
    Q, R = np.copy(Q), np.copy(R)
    if x0.ndim == 1:  # single particle case
        assert x0.ndim == 1 and R.ndim == 3 and Q.ndim == 3
        args = Q, R, x0, X_ref, U_ref, X_prev, U_prev, x_l, x_u, u_l, u_u
        dims = [4, 4, 2, 3, 3, 3, 3, 3, 3, 3, 3]
        args = [atleast_nd(z, dim) for (z, dim) in zip(args, dims)]
        Q, R, x0, X_ref, U_ref, X_prev, U_prev, x_l, x_u, u_l, u_u = args
        single_particle_problem_flag = True
    else:  # multiple particle cases
        assert x0.ndim == 2 and R.ndim == 4 and Q.ndim == 4
        single_particle_problem_flag = False
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
    data = dict(solver_data=[], hist=[], sol_hist=[])
    AA_Fs = []

    field_names = ["it", "elaps", "obj", "resid", "reg_x", "reg_u"]
    fmts = ["%04d", "%8.3e", "%8.3e", "%8.3e", "%8.3e", "%8.3e"]
    tp = utl.TablePrinter(field_names, fmts=fmts)

    # solve sequentially, linearizing ##########################################
    if verbose:
        print_fn(tp.make_header())
    it = 0
    X, U, solver_data = None, None, None
    while it < max_iters:
        X_ = np.concatenate([x0[..., None, :], X_prev[..., :-1, :]], -2)
        f, fx, fu = f_fx_fu_fn(X_, U_prev)
        f = f.reshape((M, N, xdim))
        fx = fx.reshape((M, N, xdim, xdim))
        fu = fu.reshape((M, N, xdim, udim))

        X_ref_, U_ref_ = augment_cost(
            cost_fn, X_prev, U_prev, Q, R, X_ref, U_ref
        )

        args_dyn = (f, fx, fu, x0, X_prev, U_prev)
        args_cost = (Q, R, X_ref_, U_ref_, reg_x, reg_u, slew_rate, u_slew)
        args_cstr = (x_l, x_u, u_l, u_u)
        solver_settings_ = dict(solver_settings, solver_state=solver_state)
        kw = dict(method=method, solver_settings=solver_settings_)

        t_aff_solve = time.time()
        X, U, solver_data = aff_solve(*args_dyn, *args_cost, *args_cstr, **kw)
        t_aff_solve = time.time() - t_aff_solve

        solver_state = solver_data.get("solver_state", None)
        X, U = X.reshape((M, N + 1, xdim)), U.reshape((M, N, udim))

        if debug:
            data["sol_hist"].append((X, U))

        if AA:
            X_ = np.concatenate([x0[..., None, :], X_prev], -2)
            AA_Fs.append(XU2vec(X - X_, U - U_prev))
            if it >= AA_it:
                alfs = minimize_AA(AA_Fs[-min(AA_L, len(AA_Fs)) :], 2)
                XUs = data["sol_hist"][-min(AA_L, len(AA_Fs)) :]
                X = sum(alf * X for (alf, (X, _)) in zip(alfs, XUs))
                U = sum(alf * U for (alf, (_, U)) in zip(alfs, XUs))

        if np.any(np.isnan(X)) or np.any(np.isnan(U)):
            if verbose:
                print_fn("Solver failed...")
            return None, None, None

        X_ = X[..., 1:, :]
        if AA:
            dX = data["sol_hist"][-1][0][..., 1:, :] - X_prev
            dU = data["sol_hist"][-1][1] - U_prev
        else:
            dX, dU = X_ - X_prev, U - U_prev
        max_res = max(np.max(norm(dX, 2, -1)), np.max(norm(dU, 2, -1)))
        dX, dU = X_ - X_ref, U - U_ref
        obj = (np.sum(dX * bmv(Q, dX)) + np.sum(dU * bmv(R, dU))) / N / M

        X_prev, U_prev = X[..., 1:, :], U

        t_run = time.time() - t_elaps
        vals = (it + 1, t_run, obj, max_res, reg_x, reg_u)
        if verbose:
            print_fn(tp.make_values(vals))
        data["solver_data"].append(solver_data)
        data["hist"].append({k: val for (k, val) in zip(field_names, vals)})
        data.setdefault("t_aff_solve", [])
        data["t_aff_solve"].append(t_aff_solve)

        if max_res < res_tol:
            break
        it += 1
        if (time.time() - t_elaps) * (it + 1) / it > time_limit:
            break

    if verbose:
        print_fn(tp.make_footer())
    if verbose and max_res > 1e-2:
        msg = "Bad solution found, the solution is approximate to a residual:"
        print_fn("#" * 80)
        print_fn(msg, "%9.4e" % max_res)
        print_fn("#" * 80)
    if not single_particle_problem_flag:
        return X.reshape((M, N + 1, xdim)), U.reshape((M, N, udim)), data
    else:
        return X.reshape((N + 1, xdim)), U.reshape((N, udim)), data


solve = scp_solve
##$#############################################################################
##^# tuning hyperparameters ####################################################
def tune_scp(
    *args,
    sample_nb=14,
    reg_rng=(-3, 3),
    solve_fn=scp_solve,
    savefig=None,
    **kwargs
):
    reg_ratio = kwargs.get("reg_ratio", 1e-1)

    reg_list = kwargs.get("reg_rng", np.logspace(*reg_rng, sample_nb))
    res_list = []
    for reg in tqdm(reg_list):
        reg_x, reg_u = reg, reg * reg_ratio
        kwargs["reg_x"], kwargs["reg_u"] = reg_x, reg_u
        kwargs["verbose"] = False
        X, U, data = solve_fn(*args, **kwargs)
        inf = 1e2
        res_list.append(inf if data is None else data["hist"][-1]["resid"])
    plt.figure()
    plt.loglog(reg_list, res_list)
    plt.ylabel("final residual")
    plt.xlabel("reg_x")
    plt.title("reg_u = reg_x * %6.1e" % reg_ratio)
    plt.tight_layout()
    plt.grid(b=True, which="major")
    plt.grid(b=True, which="minor")
    if savefig is not None:
        plt.savefig(savefig, dpi=200)
    plt.draw_all()
    plt.pause(1e-1)

    reg_x = reg_list[np.argmin(res_list)]
    reg_u = reg_ratio * reg_x
    return reg_x, reg_u


##$#############################################################################
##^# accelerated SCP ###########################################################
# momentum_update = lambda zk, zkm1, it: zk + it / (it + 3) * (zk - zkm1)
alf = 1.6
momentum_update = lambda zk, zkm1, it: alf * zk + (1.0 - alf) * zkm1


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
    reg_x=1e0,
    reg_u=1e-2,
    slew_rate=0.0,
    u_slew=None,
    cost_fn=None,
    method="lqp",
    solver_settings={},
    solver_state=None,
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

    field_names = ["it", "elaps", "obj", "resid", "reg_x", "reg_u"]
    fmts = ["%04d", "%8.3e", "%8.3e", "%8.3e", "%8.3e", "%8.3e"]
    tp = utl.TablePrinter(field_names, fmts=fmts)

    t_start = time.time()
    data = {}
    if verbose:
        print_fn(tp.make_header())
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
            reg_x=reg_x,
            reg_u=reg_u,
            slew_rate=slew_rate,
            u_slew=u_slew,
            cost_fn=cost_fn,
            method=method,
            solver_settings=solver_settings,
            solver_state=solver_state,
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
                data_["hist"][-1][k] for k in ["obj", "resid", "reg_x", "reg_u"]
            ]
            print_fn(tp.make_values(vals))
        if data["hist"][-1]["resid"] < res_tol:
            break
        if (it + 2) / (it + 1) * (time.time() - t_start) > time_limit:
            break
    if verbose:
        print_fn(tp.make_footer())
    return X, U, data


##$#############################################################################
