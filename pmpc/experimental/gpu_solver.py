import time
from typing import Optional, List, Tuple, Dict, Callable, Any, NamedTuple
from copy import copy
from jfi import jaxm
import numpy as np
from pmpc.utils import TablePrinter  # noqa: E402
from .solver_definitions import Solver, get_pinit_state, get_prun_with_state

tree_util = jaxm.jax.tree_util
Array = jaxm.jax.Array

SOLVE_KWS = {
    "X_ref",
    "U_ref",
    "X_prev",
    "U_prev",
    "x_l",
    "x_u",
    "u_l",
    "u_u",
    "verbose",
    "debug",
    "max_it",
    "time_limit",
    "res_tol",
    "reg_x",
    "reg_u",
    "slew_rate",
    "u_slew",
    "cost_fn",
    "extra_cvx_cost_fn",
    "solver_settings",
    "solver_state",
}


####################################################################################################

print_fn = print


def bmv(A, x):
    return (A @ x[..., None])[..., 0]


def vec(x, n=2):
    return x.reshape(x.shape[:-n] + (-1,))


def atleast_nd(x: Optional[Array], n: int):
    if x is None:
        return None
    else:
        return x.reshape((1,) * max(n - x.ndim, 0) + x.shape)


####################################################################################################


@jaxm.jit
def default_obj_fn(U: Array, args: Dict[str, List[Array]]) -> Array:
    NUMINF = 1e20
    Ft, ft, X_prev, U_prev = args["dyn"]
    Q, R, X_ref, U_ref = args["cost"]
    reg_x, reg_u = args["reg"]
    slew_rate, u0_slew = args["slew"]
    x_l, x_u, u_l, u_u, alpha = args["cstr"]

    X = (bmv(Ft, vec(U - U_prev, 2)) + ft).reshape(U.shape[:-1] + (-1,))
    dX, dU = X - X_ref, U - U_ref
    J = 0.5 * jaxm.mean(jaxm.sum(dX * bmv(Q, dX), axis=-1))
    J = J + 0.5 * jaxm.mean(jaxm.sum(dU * bmv(R, dU), axis=-1))
    J = J + 0.5 * reg_x * jaxm.mean(jaxm.sum((X - X_prev) ** 2, -1))
    J = J + 0.5 * reg_u * jaxm.mean(jaxm.sum((U - U_prev) ** 2, -1))

    # box constraints
    x_l_ = jaxm.where(jaxm.isfinite(x_l), x_l, -NUMINF)
    x_u_ = jaxm.where(jaxm.isfinite(x_u), x_u, NUMINF)
    u_l_ = jaxm.where(jaxm.isfinite(u_l), u_l, -NUMINF)
    u_u_ = jaxm.where(jaxm.isfinite(u_u), u_u, NUMINF)
    J = J + jaxm.mean(jaxm.where(jaxm.isfinite(x_l), -jaxm.log(-alpha * (-X + x_l_)) / alpha, 0.0))
    J = J + jaxm.mean(jaxm.where(jaxm.isfinite(x_u), -jaxm.log(-alpha * (X - x_u_)) / alpha, 0.0))
    J = J + jaxm.mean(jaxm.where(jaxm.isfinite(u_l), -jaxm.log(-alpha * (-U + u_l_)) / alpha, 0.0))
    J = J + jaxm.mean(jaxm.where(jaxm.isfinite(u_u), -jaxm.log(-alpha * (U - u_u_)) / alpha, 0.0))

    # slew rate
    J_slew = slew_rate * jaxm.mean(jaxm.sum((U[..., :-1, :] - U[..., 1:, :]) ** 2, -1))
    slew_rate = jaxm.where(jaxm.all(jaxm.isfinite(u0_slew), -1), slew_rate, 0.0)
    u0_slew = jaxm.where(jaxm.isfinite(u0_slew), u0_slew, 0.0)
    J_slew = J_slew + jaxm.mean(slew_rate * jaxm.sum((U[..., 0, :] - u0_slew) ** 2, -1))
    J = J + jaxm.where(jaxm.isfinite(J_slew), J_slew, 0.0)

    return jaxm.where(jaxm.isfinite(J), J, jaxm.inf)


####################################################################################################
def rollout_step_fx(x, u_f_fx_fu_x_prev_u_prev):
    u, f, fx, fu, x_prev, u_prev = u_f_fx_fu_x_prev_u_prev
    xp = f + bmv(fx, x - x_prev) + bmv(fu, u - u_prev)
    return xp, xp


@jaxm.jit
def rollout_fx(x0, U, f, fx, fu, X_prev, U_prev):
    """Rolls out dynamics into the future based on an initial state x0"""
    xs = [x0[..., None, :]]
    X_prev = jaxm.cat([x0[..., None, :], X_prev[..., :-1, :]], -2)
    U, f, X_prev, U_prev = [jaxm.moveaxis(z, -2, 0) for z in [U, f, X_prev, U_prev]]
    fx, fu = [jaxm.moveaxis(z, -3, 0) for z in [fx, fu]]
    xs = jaxm.moveaxis(jaxm.lax.scan(rollout_step_fx, x0, (U, f, fx, fu, X_prev, U_prev))[1], 0, -2)
    return jaxm.cat([x0[..., None, :], xs], -2)


####################################################################################################
@jaxm.jit
def Ft_ft_fn(x0, U, f, fx, fu, X_prev, U_prev):
    bshape, N, xdim, udim = U.shape[:-2], U.shape[-2], X_prev.shape[-1], U.shape[-1]

    ft_ = rollout_fx(x0, U, f, fx, fu, X_prev, U_prev)[..., 1:, :]
    sum_axes = tuple(range(0, ft_.ndim - 2))
    Ft_ = jaxm.jacobian(
        lambda U: jaxm.sum(rollout_fx(x0, U, f, fx, fu, X_prev, U_prev)[..., 1:, :], sum_axes)
    )(U)
    Ft_ = jaxm.moveaxis(Ft_, -3, 0)
    Ft, ft = Ft_, ft_

    Ft, ft = Ft.reshape(bshape + (N * xdim, N * udim)), ft.reshape(bshape + (N * xdim,))
    return Ft, ft


@jaxm.jit
def U2X(U, U_prev, Ft, ft):
    bshape = U.shape[:-2]
    xdim = ft.shape[-1] // U.shape[-2]
    X = (bmv(Ft, vec(U - U_prev, 2)) + ft).reshape(bshape + (U.shape[-2], xdim))
    return X


####################################################################################################
def aff_solve(
    problem: Dict[str, Array],
    reg_x: Array,
    reg_u: Array,
    solver_settings: Optional[Dict[str, Any]] = None,
    extra_cvx_cost_fn: Optional[Callable] = None,
    problems: Optional[Dict[str, Any]] = None,
) -> Tuple[Array, Array, Any]:
    """Solve a single instance of a linearized MPC problem."""
    solver_settings = copy(solver_settings) if solver_settings is not None else dict()
    alpha = solver_settings["smooth_alpha"]
    x0, f, fx, fu = problem["x0"], problem["f"], problem["fx"], problem["fu"]
    X_prev, U_prev = problem["X_prev"], problem["U_prev"]

    Ft, ft = Ft_ft_fn(x0, U_prev, f, fx, fu, X_prev, U_prev)
    args = dict()
    args["dyn"] = Ft, ft, problem["X_prev"], problem["U_prev"]
    args["cost"] = problem["Q"], problem["R"], problem["X_ref"], problem["U_ref"]
    args["reg"] = reg_x, reg_u
    args["slew"] = problem["slew_rate"], problem["u_slew"]
    args["cstr"] = problem["x_l"], problem["x_u"], problem["u_l"], problem["u_u"], alpha

    default_solver = "CVX"
    if solver_settings.get("solver", default_solver).lower() == "BFGS".lower():
        solver, max_it = Solver.BFGS, 100
    elif solver_settings.get("solver", default_solver).lower() == "LBFGS".lower():
        solver, max_it = Solver.LBFGS, 100
    elif solver_settings.get("solver", default_solver).lower() == "CVX".lower():
        solver, max_it = Solver.CVX, 50
    else:
        msg = f"Solver {solver_settings.get('solver')} not supported."
        raise ValueError(msg)

    if extra_cvx_cost_fn is None:
        obj_fn = default_obj_fn
    else:

        def obj_fn(U, args):
            return default_obj_fn(U, args) + extra_cvx_cost_fn(U, args, problems=problems)

    pinit_state = get_pinit_state(obj_fn)
    prun_with_state = get_prun_with_state(obj_fn)
    state = solver_settings.get("solver_state", None)
    if state is None or solver_settings.get("solver", default_solver).lower() in ["CVX".lower()]:
        state = pinit_state(solver, U_prev, args)

    # solve
    U, state = prun_with_state(solver, U_prev, args, state, max_it=max_it)

    mask = jaxm.tile(
        jaxm.isfinite(state.value)[..., None, None], (1,) * state.value.ndim + U.shape[-2:]
    )
    U = jaxm.where(mask, U, U_prev)

    X = U2X(U, U_prev, Ft, ft)
    return jaxm.cat([x0[..., None, :], X], -2), U, dict(solver_state=state, obj=state.value)


# cost augmentation ################################################################################
_get_new_ref = jaxm.jit(lambda ref, A, c: ref - jaxm.linalg.solve(A, c[..., None])[..., 0])


def _augment_cost(cost_fn, X_prev, U_prev, Q, R, X_ref, U_ref, problems=None):
    """Modify the linear reference trajectory to account for the linearized non-linear cost term."""
    topts = dict(dtype=X_prev.dtype, device=X_prev.device())
    if cost_fn is not None:
        cx, cu = cost_fn(X_prev, U_prev, problems=problems)

        # augment the state cost #############
        if cx is not None:
            X_ref = _get_new_ref(X_ref, Q, jaxm.to(jaxm.array(cx), **topts))

        # augment the control cost ###########
        if cu is not None:
            U_ref = _get_new_ref(U_ref, R, jaxm.to(jaxm.array(cu), **topts))
    return X_ref, U_ref


####################################################################################################


def _is_numeric(x):
    try:
        jaxm.array(x)
        return True
    except TypeError:
        return False


def stack_problems(problems: List[Dict[str, Any]]) -> Dict[str, Any]:
    assert len(problems) > 0
    problems = list(problems)
    prob_structure = tree_util.tree_structure(problems[0])
    problems = [tree_util.tree_flatten(problem)[0] for problem in problems]
    numeric_mask = [_is_numeric(val) for val in problems[0]]
    prob_vals = [
        np.stack([np.array(problem[i]) for problem in problems], 0)
        if is_numeric
        else problems[0][i]
        for (i, is_numeric) in enumerate(numeric_mask)
    ]
    problems = tree_util.tree_unflatten(prob_structure, prob_vals)
    return problems


def sanitize_problem(problems: Dict[str, Any]) -> Dict[str, Any]:
    return tree_util.tree_map(lambda x: x if _is_numeric(x) else None, problems)


def solve_problems(problems: List[Dict[str, Any]]):
    t = time.time()
    problems = stack_problems(problems)
    print(f"Stacking problems took {time.time() - t:.4e} s")
    f_fx_fu_fn = problems["f_fx_fu_fn"]
    Q, R, x0 = problems["Q"], problems["R"], problems["x0"]
    scp_solve_kws = {k: problems[k] for k in SOLVE_KWS if k in problems}
    for k in ["verbose", "max_it", "res_tol", "time_limit"]:
        if k in scp_solve_kws:
            scp_solve_kws[k] = float(scp_solve_kws[k][0])
    problems = sanitize_problem(problems)
    X, U, data = scp_solve(f_fx_fu_fn, Q, R, x0, **scp_solve_kws, problems=problems)
    data_struct = tree_util.tree_structure(data)
    data = tree_util.tree_flatten(data)[0]
    data_list = [
        tree_util.tree_unflatten(
            data_struct,
            [
                x[i] if hasattr(x, "shape") and x.ndim > 0 and x.shape[0] == X.shape[0] else x
                for x in data
            ],
        )
        for i in range(X.shape[0])
    ]
    sols = [(X[i, ...], U[i, ...], data_list[i]) for i in range(X.shape[0])]
    return sols


# SCP MPC ##########################################################################################


def scp_solve(
    f_fx_fu_fn: Callable,
    Q: Array,
    R: Array,
    x0: Array,
    X_ref: Optional[Array] = None,
    U_ref: Optional[Array] = None,
    X_prev: Optional[Array] = None,
    U_prev: Optional[Array] = None,
    x_l: Optional[Array] = None,
    x_u: Optional[Array] = None,
    u_l: Optional[Array] = None,
    u_u: Optional[Array] = None,
    verbose: bool = False,
    max_it: int = 100,
    time_limit: float = 1000.0,
    res_tol: float = 1e-5,
    reg_x: float = 1e0,
    reg_u: float = 1e-2,
    slew_rate: Optional[float] = None,
    u_slew: Optional[Array] = None,
    cost_fn: Optional[Callable] = None,
    extra_cvx_cost_fn: Optional[Callable] = None,
    solver_settings: Optional[Dict[str, Any]] = None,
    solver_state: Optional[Any] = None,
    return_min_viol: bool = False,
    min_viol_it0: int = -1,
    dtype: Any = jaxm.float32,
    device: Any = "cuda",
    problems: Optional[Dict[str, Any]] = None,
) -> Tuple[Array, Array, Dict[str, Any]]:
    """Compute the SCP solution to a non-linear dynamics, quadratic cost, control problem with
    optional non-linear cost term.

    Args:
        f_fx_fu_fn (Callable): Dynamics with linearization callable.
        Q (Array): The quadratic state cost.
        R (Array): The quadratic control cost.
        x0 (Array): Initial state.
        X_ref (Optional[Array], optional): Reference state trajectory. Defaults to zeros.
        U_ref (Optional[Array], optional): Reference control trajectory. Defaults to zeros.
        X_prev (Optional[Array], optional): Previous state solution. Defaults to x0.
        U_prev (Optional[Array], optional): Previous control solution. Defaults to zeros.
        x_l (Optional[Array], optional): Lower bound state constraint. Defaults to no cstrs.
        x_u (Optional[Array], optional): Upper bound state constraint. Defaults to no cstrs.
        u_l (Optional[Array], optional): Lower bound control constraint.. Defaults to no cstrs.
        u_u (Optional[Array], optional): Upper bound control constraint.. Defaults to no cstrs.
        verbose (bool, optional): Whether to print output. Defaults to False.
        max_it (int, optional): Max number of SCP iterations. Defaults to 100.
        time_limit (float, optional): Time limit in seconds. Defaults to 1000.0.
        res_tol (float, optional): Residual tolerance. Defaults to 1e-5.
        reg_x (float, optional): State improvement regularization. Defaults to 1e0.
        reg_u (float, optional): Control improvement regularization. Defaults to 1e-2.
        slew_rate (float, optional): Slew rate regularization. Defaults to 0.0.
        u_slew (Optional[Array], optional): Slew control to regularize to. Defaults to None.
        cost_fn (Optional[Callable], optional): Linearization of the non-linear cost function.
                                                Defaults to None.
        extra_cvx_cost_fn (Optional[Callable], optional): Extra convex cost function.
                                                          Defaults to None.
        solver_settings (Optional[Dict[str, Any]], optional): Solver settings. Defaults to None.
        return_min_viol (bool, optional): Whether to return minimum violation solution as well.
                                          Defaults to False.
        min_viol_it0 (int, optional): First iteration to store minimum violation solutions.
                                      Defaults to -1, which means immediately.
    Returns:
        Tuple[Array, Array, Dict[str, Any]]: X, U, data
    """
    t_elaps = time.time()
    topts = dict(device=device, dtype=dtype)

    # create variables and reference trajectories ##############################
    x0 = jaxm.to(jaxm.array(x0), **topts)
    reg_x, reg_u = jaxm.to(jaxm.array(reg_x), **topts), jaxm.to(jaxm.array(reg_u), **topts)
    Q, R = jaxm.to(jaxm.copy(Q), **topts), jaxm.to(jaxm.copy(R), **topts)
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

    X_ref = (
        jaxm.zeros((M, N, xdim), **topts) if X_ref is None else jaxm.to(jaxm.array(X_ref), **topts)
    )
    U_ref = (
        jaxm.zeros((M, N, udim), **topts) if U_ref is None else jaxm.to(jaxm.array(U_ref), **topts)
    )
    X_prev = jaxm.to(jaxm.array(X_prev), **topts) if X_prev is not None else X_ref
    U_prev = jaxm.to(jaxm.array(U_prev), **topts) if U_prev is not None else U_ref
    X_prev, U_prev = X_prev.reshape((M, N, xdim)), U_prev.reshape((M, N, udim))
    X_ref, U_ref = X_ref.reshape((M, N, xdim)), U_ref.reshape((M, N, udim))
    x_l = (
        jaxm.to(jaxm.array(x_l), **topts)
        if x_l is not None
        else jaxm.nan * jaxm.ones(X_prev.shape, **topts)
    )
    x_u = (
        jaxm.to(jaxm.array(x_u), **topts)
        if x_u is not None
        else jaxm.nan * jaxm.ones(X_prev.shape, **topts)
    )
    u_l = (
        jaxm.to(jaxm.array(u_l), **topts)
        if u_l is not None
        else jaxm.nan * jaxm.ones(U_prev.shape, **topts)
    )
    u_u = (
        jaxm.to(jaxm.array(u_u), **topts)
        if u_u is not None
        else jaxm.nan * jaxm.ones(U_prev.shape, **topts)
    )
    u_slew = (
        jaxm.to(jaxm.array(u_slew), **topts)
        if u_slew is not None
        else jaxm.nan * jaxm.ones(x0.shape[:-1] + (U_prev.shape[-1],), **topts)
    )
    slew_rate = slew_rate if slew_rate is not None else 0.0
    data = dict(solver_data=[], hist=[], sol_hist=[])

    field_names = ["it", "elaps", "obj", "resid", "reg_x", "reg_u", "alpha"]
    fmts = ["%04d", "%8.3e", "%8.3e", "%8.3e", "%.1e", "%.1e", "%.1e"]
    tp = TablePrinter(field_names, fmts=fmts)
    solver_settings = solver_settings if solver_settings is not None else dict()

    min_viol = jaxm.inf

    # solve sequentially, linearizing ##############################################################
    if verbose:
        print_fn(tp.make_header())
    it = 0
    X, U, solver_data = None, None, None
    while it < max_it:
        X_ = jaxm.cat([x0[..., None, :], X_prev[..., :-1, :]], -2)
        f, fx, fu = f_fx_fu_fn(X_, U_prev)
        f = jaxm.to(jaxm.array(f), **topts).reshape((M, N, xdim))
        fx = jaxm.to(jaxm.array(fx), **topts).reshape((M, N, xdim, xdim))
        fu = jaxm.to(jaxm.array(fu), **topts).reshape((M, N, xdim, udim))

        # augment the cost or add extra constraints ################################################
        X_ref_, U_ref_ = _augment_cost(cost_fn, X_prev, U_prev, Q, R, X_ref, U_ref)
        # if extra_cstrs_fns is not None:
        #    solver_settings["extra_cstrs"] = tuple(extra_cstrs_fns(X_prev, U_prev))
        # if "extra_cstrs" in solver_settings:
        #    solver_settings["extra_cstrs"] = tuple(
        #        [
        #            [(arg.tolist() if hasattr(arg, "tolist") else arg) for arg in extra_cstr]
        #            for extra_cstr in solver_settings["extra_cstrs"]
        #        ]
        #    )
        problem = dict(f=f, fx=fx, fu=fu, x0=x0, X_prev=X_prev, U_prev=U_prev)
        problem = dict(problem, Q=Q, R=R, X_ref=X_ref_, U_ref=U_ref_)
        problem = dict(problem, slew_rate=slew_rate, u_slew=u_slew)
        problem = dict(problem, x_l=x_l, x_u=x_u, u_l=u_l, u_u=u_u)
        solver_settings = solver_settings if solver_settings is not None else dict()
        solver_settings["solver_state"] = solver_state
        kw = dict(solver_settings=solver_settings)
        smooth_alpha = kw["solver_settings"].get("smooth_alpha", 1e4)
        new_smooth_alpha = jaxm.minimum(10 ** (-1 + min(it, 10)), smooth_alpha)
        smooth_alpha = new_smooth_alpha
        kw["solver_settings"] = dict(kw["solver_settings"], smooth_alpha=smooth_alpha)

        t_aff_solve = time.time()
        X, U, solver_data = aff_solve(
            problem, reg_x, reg_u, **kw, extra_cvx_cost_fn=extra_cvx_cost_fn, problems=problems
        )
        t_aff_solve = time.time() - t_aff_solve

        solver_state = solver_data.get("solver_state", None)
        X, U = X.reshape((M, N + 1, xdim)), U.reshape((M, N, udim))

        # return if the solver failed ##############################################################
        if jaxm.any(jaxm.isnan(X)) or jaxm.any(jaxm.isnan(U)):
            if verbose:
                print_fn("Solver failed...")
            return None, None, None
        # return if the solver failed ##############################################################

        X_ = X[..., 1:, :]
        dX, dU = X_ - X_prev, U - U_prev
        max_res = max(jaxm.max(jaxm.linalg.norm(dX, 2, -1)), jaxm.max(jaxm.linalg.norm(dU, 2, -1)))
        dX, dU = X_ - X_ref, U - U_ref
        obj = np.mean(solver_data.get("obj", 0.0))
        X_prev, U_prev = X[..., 1:, :], U

        t_run = time.time() - t_elaps
        vals = (it + 1, t_run, obj, max_res, np.mean(reg_x), np.mean(reg_u), np.mean(smooth_alpha))
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
        print_fn("#" * 80)
        print_fn(msg, "%9.4e" % max_res)
        print_fn("#" * 80)
    if not single_particle_problem_flag:
        return X.reshape((M, N + 1, xdim)), U.reshape((M, N, udim)), data
    else:
        return X.reshape((N + 1, xdim)), U.reshape((N, udim)), data
