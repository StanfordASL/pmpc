##^# library imports ###########################################################
from __future__ import annotations

import pdb
import math
import time
from copy import copy
from typing import Any, Callable, Dict, List, Optional, Tuple
from multiprocessing import Pool
from warnings import warn

import numpy as np

from . import julia_utils as ju
from .utils import TablePrinter, atleast_nd, to_numpy_f64
from .static_backend import aff_solve as static_aff_solve, is_precompiled_backend_available

jl = None
JULIA_SOLVE_FNS = dict()  # attribute look up is expensive (~ 2 ms every time)


def print_fn(*args, **kwargs):
    return print(*args, **kwargs)


def ensure_julia():
    global jl, JULIA_SOLVE_FNS

    if jl is None:
        jl = ju.load_julia()
        # JULIA_SOLVE_FNS["admm"] = jl.admm_solve
        JULIA_SOLVE_FNS["qp"] = jl.lqp_solve
        JULIA_SOLVE_FNS["cone"] = jl.lcone_solve


# fixed-point convergence methods ##################################################################
def AA_method(Fs: List[np.ndarray]) -> np.ndarray:
    F = np.stack([f.reshape(-1) for f in Fs], -1)
    Ft = F[:, :-1] - F[:, -1:]
    th = np.linalg.solve(Ft.T @ Ft + 1e-10 * np.eye(Ft.shape[-1]), -Ft.T @ F[:, -1:]).reshape(-1)
    alf_ = np.concatenate([th, [1.0 - np.sum(th)]], -1)
    return alf_


def smooth_method(Fs: List[np.ndarray]) -> np.ndarray:
    F = np.stack([f.reshape(-1) for f in Fs], -1)
    return np.ones(F.shape[-1]) / F.shape[-1]


def select_method(Fs: List[np.ndarray]) -> np.ndarray:
    F = np.stack([f.reshape(-1) for f in Fs], -1)
    A = np.diag(np.linalg.norm(F, axis=-2) ** 2)
    A = np.concatenate([A, np.ones((A.shape[-2], 1))], -1)
    temp = np.ones((1, A.shape[-1]))
    temp[:, -1] = 0.0
    A = np.concatenate([A, temp], -2)
    b = np.concatenate([np.zeros(F.shape[-1]), np.ones(1)], -1)
    alf = np.linalg.solve(A, b).reshape(-1)[:-1]
    return alf


FILTER_MAP = dict(smooth=smooth_method, select=select_method, AA=AA_method)


# affine solve using julia #########################################################################
def lqp_generate_problem_matrices(x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref):
    ensure_julia()
    x0 = atleast_nd(x0, 2)
    f, X_prev, U_prev, X_ref, U_ref = [atleast_nd(z, 3) for z in [f, X_prev, U_prev, X_ref, U_ref]]
    fx, fu, Q, R = [atleast_nd(z, 4) for z in [fx, fu, Q, R]]
    x0, f, X_prev, U_prev, X_ref, U_ref = [
        ju.py2jl(to_numpy_f64(z), 1) for z in [x0, f, X_prev, U_prev, X_ref, U_ref]
    ]
    fx, fu, Q, R = [ju.py2jl(to_numpy_f64(z), 2) for z in [fx, fu, Q, R]]
    return jl.lqp_generate_problem_matrices(x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref)


def aff_solve(
    f: np.ndarray,
    fx: np.ndarray,
    fu: np.ndarray,
    x0: np.ndarray,
    X_prev: np.ndarray,
    U_prev: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    X_ref: np.ndarray,
    U_ref: np.ndarray,
    reg_x: np.ndarray,
    reg_u: np.ndarray,
    slew_rate: float,
    u_slew: np.ndarray,
    x_l: np.ndarray,
    x_u: np.ndarray,
    u_l: np.ndarray,
    u_u: np.ndarray,
    solver_settings: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    if is_precompiled_backend_available():
        return static_aff_solve(
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
            solver_settings,
        )
    msg = (
        "You are using a python-only version of this package, system Julia will be used. "
        + "Ensure you have the PMPC.jl package installed in your system Julia."
    )
    warn(msg)

    """Solve a single instance of a linearized MPC problem."""
    ensure_julia()
    f = atleast_nd(f, 3)
    fx, fu = atleast_nd(fx, 4), atleast_nd(fu, 4)
    x0 = atleast_nd(x0, 2)
    X_prev, U_prev = atleast_nd(X_prev, 3), atleast_nd(U_prev, 3)
    Q, R = atleast_nd(Q, 4), atleast_nd(R, 4)
    X_ref, U_ref = atleast_nd(X_ref, 3), atleast_nd(U_ref, 3)
    x_l, x_u, u_l, u_u = [atleast_nd(z, 3) for z in [x_l, x_u, u_l, u_u]]

    x_l, x_u, u_l, u_u = [ju.py2jl(to_numpy_f64(z), 1) for z in [x_l, x_u, u_l, u_u]]
    args = [
        ju.py2jl(to_numpy_f64(z), d)
        for (z, d) in zip(
            [x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref],
            [1, 1, 2, 2, 1, 1, 2, 2, 1, 1],
        )
    ]
    solver_settings.setdefault("solver", "ecos")
    method = "qp" if solver_settings["solver"] == "osqp" else "cone"
    solve_fn = JULIA_SOLVE_FNS[method]

    solver_settings = copy(solver_settings) if solver_settings is not None else dict()
    if u_slew is not None:
        solver_settings["slew_um1"] = to_numpy_f64(u_slew)
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


# cost augmentation ################################################################################
def _augment_cost(lin_cost_fn, X_prev, U_prev, Q, R, X_ref, U_ref, problems):
    """Modify the linear reference trajectory to account for the linearized non-linear cost term."""
    if lin_cost_fn is not None:
        cx, cu = lin_cost_fn(X_prev, U_prev, problems)

        # augment the state cost #############
        if cx is not None:
            cx = np.array(cx)
            X_ref = X_ref - np.linalg.solve(Q, cx[..., None])[..., 0]

        # augment the control cost ###########
        if cu is not None:
            cu = np.array(cu)
            U_ref = U_ref - np.linalg.solve(R, cu[..., None])[..., 0]
    return X_ref, U_ref


# SCP MPC ##########################################################################################
def norm(x, p=None, dim=None):
    return np.linalg.norm(x, p, dim)


def bmv(A, x):
    return (A @ x[..., None])[..., 0]


def XU2vec(X, U):
    return np.concatenate([X.reshape(-1), U.reshape(-1)])


def vec2XU(z, Xshape, Ushape):
    return z[: np.prod(Xshape)].reshape(Xshape), z[np.prod(Xshape) :].reshape(Ushape)


def scp_solve(
    f_fx_fu_fn: Callable,
    Q: np.ndarray,
    R: np.ndarray,
    x0: np.ndarray,
    X_ref: Optional[np.ndarray] = None,
    U_ref: Optional[np.ndarray] = None,
    X_prev: Optional[np.ndarray] = None,
    U_prev: Optional[np.ndarray] = None,
    x_l: Optional[np.ndarray] = None,
    x_u: Optional[np.ndarray] = None,
    u_l: Optional[np.ndarray] = None,
    u_u: Optional[np.ndarray] = None,
    verbose: bool = False,
    debug: bool = False,
    max_it: int = 100,
    time_limit: float = 1000.0,
    res_tol: float = 1e-5,
    reg_x: float = 1e0,
    reg_u: float = 1e-2,
    slew_rate: float = 0.0,
    u0_slew: Optional[np.ndarray] = None,
    lin_cost_fn: Optional[Callable] = None,
    cost_fn: Optional[Callable] = None,  # deprecated
    extra_cstrs_fns: Optional[Callable] = None,
    solver_settings: Optional[Dict[str, Any]] = None,
    solver_state: Optional[Dict[str, Any]] = None,
    filter_method: str = "",
    filter_window: int = 5,
    filter_it0: int = 20,
    return_min_viol: bool = False,
    min_viol_it0: int = -1,
    **extra_kw,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compute the SCP solution to a non-linear dynamics, quadratic cost, control problem with
    optional non-linear cost term.

    Args:
        f_fx_fu_fn (Callable): Dynamics with linearization callable.
        Q (np.ndarray): The quadratic state cost.
        R (np.ndarray): The quadratic control cost.
        x0 (np.ndarray): Initial state.
        X_ref (Optional[np.ndarray], optional): Reference state trajectory. Defaults to zeros.
        U_ref (Optional[np.ndarray], optional): Reference control trajectory. Defaults to zeros.
        X_prev (Optional[np.ndarray], optional): Previous state solution. Defaults to x0.
        U_prev (Optional[np.ndarray], optional): Previous control solution. Defaults to zeros.
        x_l (Optional[np.ndarray], optional): Lower bound state constraint. Defaults no cstrs.
        x_u (Optional[np.ndarray], optional): Upper bound state constraint. Defaults to no cstrs.
        u_l (Optional[np.ndarray], optional): Lower bound control constraint. Defaults to no cstrs.
        u_u (Optional[np.ndarray], optional): Upper bound control constraint. Defaults to no cstrs.
        verbose (bool, optional): Whether to print output. Defaults to False.
        debug (bool, optional): Whether to store debugging information. Defaults to False.
        max_it (int, optional): Max number of SCP iterations. Defaults to 100.
        time_limit (float, optional): Time limit in seconds. Defaults to 1000.0.
        res_tol (float, optional): Residual tolerance. Defaults to 1e-5.
        reg_x (float, optional): State improvement regularization. Defaults to 1e0.
        reg_u (float, optional): Control improvement regularization. Defaults to 1e-2.
        slew_rate (float, optional): Slew rate regularization. Defaults to 0.0.
        u0_slew (Optional[np.ndarray], optional): Slew control 0 to regularize to. Defaults to None.
        lin_cost_fn (Optional[Callable], optional): Linearization of the non-linear cost function.
                                                    Defaults to None.
        solver_settings (Optional[Dict[str, Any]], optional): Solver settings. Defaults to None.
        solver_state (Optional[Dict[str, Any]], optional): Solver state. Defaults to None.
        filter_method (str, optional): Filter method to choose. Defaults to "" == no filter.
        filter_window (int, optional): Filter window to pick. Defaults to 5.
        filter_it0 (int, optional): First iteration to start filtering on. Defaults to 20.
        return_min_viol (bool, optional): Whether to return minimum violation solution as well.
                                          Defaults to False.
        min_viol_it0 (int, optional): First iteration to store minimum violation solutions.
                                      Defaults to -1, which means immediately.
    Returns:
        Tuple[np.ndarray, ]: _description_
    """
    if cost_fn is not None:
        raise ValueError("cost_fn is deprecated, use lin_cost_fn instead.")

    t_elaps = time.time()

    # create variables and reference trajectories ##############################
    # from jax import Array

    # if isinstance(Q, Array):
    #    print(f"Q.device = {Q.device()}")

    # zs_ = [Q, R, x0, X_ref, U_ref, X_prev, U_prev, x_l, x_u, u_l, u_u]
    # for z_ in zs_:
    #    print(f"{type(z_)}", end="")
    #    if isinstance(z_, Array):
    #        print(f" device = {z_.device()}")
    #    else:
    #        print()

    x0, reg_x, reg_u = np.array(x0), float(reg_x), float(reg_u)
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

    X_ref = np.zeros((M, N, xdim)) if X_ref is None else np.array(X_ref)
    U_ref = np.zeros((M, N, udim)) if U_ref is None else np.array(U_ref)
    X_prev = np.array(X_prev) if X_prev is not None else X_ref
    U_prev = np.array(U_prev) if U_prev is not None else U_ref
    X_prev, U_prev = X_prev.reshape((M, N, xdim)), U_prev.reshape((M, N, udim))
    X_ref, U_ref = X_ref.reshape((M, N, xdim)), U_ref.reshape((M, N, udim))
    x_l, x_u, u_l, u_u = [
        np.array(z) if z is not None else np.zeros((0, 0, 0)) for z in [x_l, x_u, u_l, u_u]
    ]
    slew_rate = slew_rate if slew_rate is None else float(slew_rate)
    u0_slew = np.array(u0_slew) if u0_slew is not None else None
    data = dict(solver_data=[], hist=[], sol_hist=[])
    Fs = []

    field_names = ["it", "elaps", "obj", "resid", "reg_x", "reg_u"]
    fmts = ["%04d", "%8.3e", "%8.3e", "%8.3e", "%8.3e", "%8.3e"]
    tp = TablePrinter(field_names, fmts=fmts)
    solver_settings = copy(solver_settings) if solver_settings is not None else dict()

    min_viol = math.inf

    # solve sequentially, linearizing ##############################################################
    if verbose:
        print_fn(tp.make_header())
    it = 0
    X, U, solver_data = None, None, None
    while it < max_it:
        X_ = np.concatenate([x0[..., None, :], X_prev[..., :-1, :]], -2)
        f, fx, fu = f_fx_fu_fn(X_, U_prev)
        f = f.reshape((M, N, xdim))
        fx = fx.reshape((M, N, xdim, xdim))
        fu = fu.reshape((M, N, xdim, udim))

        # augment the cost or add extra constraints ################################################

        problems = dict(f_fx_fu_fn=f_fx_fu_fn)
        problems = dict(problems, f=f, fx=fx, fu=fu, x0=x0, X_prev=X_prev, U_prev=U_prev)
        problems = dict(problems, slew_rate=slew_rate, u0_slew=u0_slew)
        problems = dict(problems, x_l=x_l, x_u=x_u, u_l=u_l, u_u=u_u)
        problems = dict(problems, Q=Q, R=R, X_ref=X_ref, U_ref=U_ref)
        problems = dict(extra_kw, **problems)
        X_ref_, U_ref_ = _augment_cost(lin_cost_fn, X_prev, U_prev, Q, R, X_ref, U_ref, problems)
        if extra_cstrs_fns is not None:
            solver_settings["extra_cstrs"] = tuple(extra_cstrs_fns(X_prev, U_prev, problems))
        if "extra_cstrs" in solver_settings:
            solver_settings["extra_cstrs"] = tuple(
                [
                    [(arg.tolist() if hasattr(arg, "tolist") else arg) for arg in extra_cstr]
                    for extra_cstr in solver_settings["extra_cstrs"]
                ]
            )
        args_dyn = (f, fx, fu, x0, X_prev, U_prev)
        args_cost = (Q, R, X_ref_, U_ref_, reg_x, reg_u, slew_rate, u0_slew)
        args_cstr = (x_l, x_u, u_l, u_u)
        solver_settings = solver_settings if solver_settings is not None else dict()
        solver_settings["solver_state"] = solver_state
        kw = dict(solver_settings=solver_settings)

        t_aff_solve = time.time()
        X, U, solver_data = aff_solve(*args_dyn, *args_cost, *args_cstr, **kw)
        t_aff_solve = time.time() - t_aff_solve

        solver_state = solver_data.get("solver_state", None)
        X, U = X.reshape((M, N + 1, xdim)), U.reshape((M, N, udim))

        if debug or filter_method != "":
            data["sol_hist"].append((X, U))

        # filter the result if filter method is requested ##########################################
        if filter_method != "":
            X_ = np.concatenate([x0[..., None, :], X_prev], -2)
            Fs.append(XU2vec(X - X_, U - U_prev))
            if it >= filter_it0:
                alfs = FILTER_MAP[filter_method](Fs[-min(filter_window, len(Fs)) :])
                XUs = data["sol_hist"][-min(filter_window, len(Fs)) :]
                X = sum(alf * X for (alf, (X, _)) in zip(alfs, XUs))
                U = sum(alf * U for (alf, (_, U)) in zip(alfs, XUs))
        # filter the result if filter method is requested ##########################################

        # return if the solver failed ##############################################################
        if np.any(np.isnan(X)) or np.any(np.isnan(U)):
            if verbose:
                print_fn("Solver failed...")
            return None, None, None
        # return if the solver failed ##############################################################

        X_ = X[..., 1:, :]
        if filter_method != "":
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

        # store the minimum violation solution #####################################################
        if return_min_viol and (it >= min_viol_it0 or min_viol_it0 < 0):
            if min_viol > max_res:
                data["min_viol_sol"], min_viol = (X, U), max_res
        # store the minimum violation solution #####################################################

        if max_res < res_tol:
            break
        it += 1
        if (time.time() - t_elaps) * (it + 1) / it > time_limit:
            break

    if verbose:
        print_fn(tp.make_footer())
    if verbose and max_res > 1e-2:
        msg = "Bad solution found, the solution is approximate to a residual:"
        print_fn("#" * 73)
        print_fn(msg, "%9.4e" % max_res)
        print_fn("#" * 73)
    if not debug:
        del data["sol_hist"]
    if not single_particle_problem_flag:
        return X.reshape((M, N + 1, xdim)), U.reshape((M, N, udim)), data
    else:
        return X.reshape((N + 1, xdim)), U.reshape((N, udim)), data


# solve = scp_solve  # set an alias
def solve(*args, **kwargs):
    if kwargs.get("profile", False):
        from line_profiler import LineProfiler

        LP = LineProfiler()
        LP.add_function(scp_solve)
        ret = LP.wrap_function(scp_solve)(*args, **kwargs)
        LP.print_stats(output_unit=1e-3)
    else:
        ret = scp_solve(*args, **kwargs)
    return ret


# tuning hyperparameters ###########################################################################
def tune_scp(
    *args,
    sample_nb: int = 14,
    reg_rng: Tuple[int, int] = (-3, 3),
    solve_fn: Callable = scp_solve,
    savefig: Optional[str] = None,
    **kwargs,
):
    import matplotlib.pyplot as plt
    from tqdm import tqdm

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


def solve_with_a_dict(problem: dict[str, Any]) -> tuple:
    return solve(**problem)


def solve_problems(
    problems: List[Dict[str, Any]],
    verbose: bool = False,
    **kw,  # kwargs to ignore
) -> List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    # with Pool(8) as pool:
    #    return pool.map(solve_with_a_dict, [dict(p, verbose=verbose) for p in problems])
    return [solve(**dict(p, verbose=verbose)) for p in problems]
