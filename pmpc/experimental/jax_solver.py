import time
from typing import Optional, List, Tuple, Dict, Callable, Any
from copy import copy
from functools import partial
from enum import Enum

# import jax
from jfi import init

jaxm = init()

from pmpc.utils import TablePrinter # noqa: E402
from .convex_solver import ConvexSolver # noqa: E402
import jaxopt # noqa: E402

Array = jaxm.jax.Array


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
def obj_fn(U: Array, args: Dict[str, List[Array]]) -> Array:
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


class Solver(Enum):
    BFGS = 0
    LBFGS = 1
    CVX = 2



opts = dict(
    maxiter=300,
    verbose=False,
    jit=True,
    linesearch="backtracking",
)
SOLVERS = {
    Solver.BFGS: jaxopt.BFGS(obj_fn, **opts),
    Solver.LBFGS: jaxopt.LBFGS(obj_fn, **opts),
    #Solver.CVX: ConvexSolver(obj_fn, **opts),
    Solver.CVX: ConvexSolver(obj_fn, **dict(opts, maxls=20, linesearch="binary_search")),
}

RUN_METHODS = {k: jaxm.jit(solver.run) for k, solver in SOLVERS.items()}
UPDATE_METHODS = {k: jaxm.jit(solver.update) for k, solver in SOLVERS.items()}


@partial(jaxm.jit, static_argnums=(0,))
def run_with_state(solver: int, z: Array, args: Dict[str, List[Array]], state, max_it: int = 100):
    def body_fn(i, z_state):
        return UPDATE_METHODS[solver](*z_state, args)
    z_state = body_fn(0, (z, state))
    return jaxm.jax.lax.fori_loop(1, max_it, body_fn, z_state)


@partial(jaxm.jit, static_argnums=(0,))
def init_state(solver: int, U_prev: Array, args: Dict[str, List[Array]]):
    return SOLVERS[solver].init_state(U_prev, args)


@partial(jaxm.jit, static_argnums=(0,))
def prun_with_state(solver: int, z: Array, args: Dict[str, List[Array]], state, max_it: int = 100):
    in_axes = jaxm.jax.tree_util.tree_map(
        lambda x: 0 if (hasattr(x, "shape") and x.ndim > 0 and x.shape[0] == z.shape[0]) else None,
        (solver, z, args, state),
    )
    return jaxm.jax.vmap(run_with_state, in_axes=in_axes)(solver, z, args, state)


@partial(jaxm.jit, static_argnums=(0,))
def pinit_state(solver: int, U_prev: Array, args: Dict[str, List[Array]]):
    in_axes = jaxm.jax.tree_util.tree_map(
        lambda x: 0
        if (hasattr(x, "shape") and x.ndim > 0 and x.shape[0] == U_prev.shape[0])
        else None,
        (U_prev, args),
    )
    return jaxm.jax.vmap(SOLVERS[solver].init_state, in_axes=in_axes)(U_prev, args)


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
    f: Array,
    fx: Array,
    fu: Array,
    x0: Array,
    X_prev: Array,
    U_prev: Array,
    Q: Array,
    R: Array,
    X_ref: Array,
    U_ref: Array,
    reg_x: Array,
    reg_u: Array,
    slew_rate: float,
    u_slew: Array,
    x_l: Array,
    x_u: Array,
    u_l: Array,
    u_u: Array,
    solver_settings: Optional[Dict[str, Any]] = None,
) -> Tuple[Array, Array, Any]:
    """Solve a single instance of a linearized MPC problem."""
    solver_settings = copy(solver_settings) if solver_settings is not None else dict()
    alpha = solver_settings["smooth_alpha"]

    Ft, ft = Ft_ft_fn(x0, U_prev, f, fx, fu, X_prev, U_prev)
    args = dict()
    args["dyn"] = Ft, ft, X_prev, U_prev
    args["cost"] = Q, R, X_ref, U_ref
    args["reg"] = reg_x, reg_u
    args["slew"] = slew_rate, u_slew
    args["cstr"] = x_l, x_u, u_l, u_u, alpha

    default_solver = "CVX"
    if solver_settings.get("solver", default_solver).lower() == "BFGS".lower():
        solver, max_it = Solver.BFGS, 100
    elif solver_settings.get("solver", default_solver).lower() == "LBFGS".lower():
        solver, max_it = Solver.LBFGS, 100
    elif solver_settings.get("solver", default_solver).lower() == "CVX".lower():
        solver, max_it = Solver.CVX, 30
    else:
        msg = f"Solver {solver_settings.get('solver')} not supported."
        raise ValueError(msg)
    state = solver_settings.get("solver_state", None)
    if state is None or solver_settings.get("solver", default_solver).lower() in ["CVX".lower()]:
        state = pinit_state(solver, U_prev, args)
    U, state = prun_with_state(solver, U_prev, args, state, max_it=max_it)

    mask = jaxm.tile(
        jaxm.isfinite(state.value)[..., None, None], (1,) * state.value.ndim + U.shape[-2:]
    )
    U = jaxm.where(mask, U, U_prev)

    X = U2X(U, U_prev, Ft, ft)
    return jaxm.cat([x0[..., None, :], X], -2), U, dict(solver_state=state, obj=state.value)


# cost augmentation ################################################################################
_get_new_ref = jaxm.jit(lambda ref, A, c: ref - jaxm.linalg.solve(A, c[..., None])[..., 0])


def _augment_cost(cost_fn, X_prev, U_prev, Q, R, X_ref, U_ref):
    """Modify the linear reference trajectory to account for the linearized non-linear cost term."""
    if cost_fn is not None:
        cx, cu = cost_fn(X_prev, U_prev)

        # augment the state cost #############
        if cx is not None:
            X_ref = _get_new_ref(X_ref, Q, jaxm.array(cx))

        # augment the control cost ###########
        if cu is not None:
            U_ref = _get_new_ref(U_ref, R, jaxm.array(cu))
    return X_ref, U_ref


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
    extra_cstrs_fns: Optional[Callable] = None,
    solver_settings: Optional[Dict[str, Any]] = None,
    solver_state: Optional[Any] = None,
    return_min_viol: bool = False,
    min_viol_it0: int = -1,
) -> Tuple[Array, Array, Dict[str, Any]]:
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
        x_l (Optional[np.ndarray], optional): Lower bound state constraint. Defaults to no cstrs.
        x_u (Optional[np.ndarray], optional): Upper bound state constraint. Defaults to no cstrs.
        u_l (Optional[np.ndarray], optional): Lower bound control constraint.. Defaults to no cstrs.
        u_u (Optional[np.ndarray], optional): Upper bound control constraint.. Defaults to no cstrs.
        verbose (bool, optional): Whether to print output. Defaults to False.
        max_it (int, optional): Max number of SCP iterations. Defaults to 100.
        time_limit (float, optional): Time limit in seconds. Defaults to 1000.0.
        res_tol (float, optional): Residual tolerance. Defaults to 1e-5.
        reg_x (float, optional): State improvement regularization. Defaults to 1e0.
        reg_u (float, optional): Control improvement regularization. Defaults to 1e-2.
        slew_rate (float, optional): Slew rate regularization. Defaults to 0.0.
        u_slew (Optional[np.ndarray], optional): Slew control to regularize to. Defaults to None.
        cost_fn (Optional[Callable], optional): Linearization of the non-linear cost function. 
                                                Defaults to None.
        solver_settings (Optional[Dict[str, Any]], optional): Solver settings. Defaults to None.
        return_min_viol (bool, optional): Whether to return minimum violation solution as well. 
                                          Defaults to False.
        min_viol_it0 (int, optional): First iteration to store minimum violation solutions. 
                                      Defaults to -1, which means immediately.
    Returns:
        Tuple[np.ndarray, ]: _description_
    """
    t_elaps = time.time()

    # create variables and reference trajectories ##############################
    x0, reg_x, reg_u = jaxm.array(x0), jaxm.array(reg_x), jaxm.array(reg_u)
    Q, R = jaxm.copy(Q), jaxm.copy(R)
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

    X_ref = jaxm.zeros((M, N, xdim)) if X_ref is None else jaxm.array(X_ref)
    U_ref = jaxm.zeros((M, N, udim)) if U_ref is None else jaxm.array(U_ref)
    X_prev = jaxm.array(X_prev) if X_prev is not None else X_ref
    U_prev = jaxm.array(U_prev) if U_prev is not None else U_ref
    X_prev, U_prev = X_prev.reshape((M, N, xdim)), U_prev.reshape((M, N, udim))
    X_ref, U_ref = X_ref.reshape((M, N, xdim)), U_ref.reshape((M, N, udim))
    x_l = jaxm.array(x_l) if x_l is not None else jaxm.nan * jaxm.ones(X_prev.shape)
    x_u = jaxm.array(x_u) if x_u is not None else jaxm.nan * jaxm.ones(X_prev.shape)
    u_l = jaxm.array(u_l) if u_l is not None else jaxm.nan * jaxm.ones(U_prev.shape)
    u_u = jaxm.array(u_u) if u_u is not None else jaxm.nan * jaxm.ones(U_prev.shape)
    u_slew = (
        u_slew if u_slew is not None else jaxm.nan * jaxm.ones(x0.shape[:-1] + (U_prev.shape[-1],))
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
        f = jaxm.array(f).reshape((M, N, xdim))
        fx = jaxm.array(fx).reshape((M, N, xdim, xdim))
        fu = jaxm.array(fu).reshape((M, N, xdim, udim))

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
        args_dyn = (f, fx, fu, x0, X_prev, U_prev)
        args_cost = (Q, R, X_ref_, U_ref_, reg_x, reg_u, slew_rate, u_slew)
        args_cstr = (x_l, x_u, u_l, u_u)
        solver_settings = solver_settings if solver_settings is not None else dict()
        solver_settings["solver_state"] = solver_state
        kw = dict(solver_settings=solver_settings)
        smooth_alpha = kw["solver_settings"].get("smooth_alpha", 1e4)
        new_smooth_alpha = jaxm.minimum(10 ** (-1 + min(it, 10)), smooth_alpha)
        #if new_smooth_alpha != smooth_alpha:
        #    print("Deleting")
        #    del kw["solver_settings"]["solver_state"]
        smooth_alpha = new_smooth_alpha
        kw["solver_settings"] = dict(kw["solver_settings"], smooth_alpha=smooth_alpha)

        t_aff_solve = time.time()
        X, U, solver_data = aff_solve(*args_dyn, *args_cost, *args_cstr, **kw)
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
        obj = jaxm.mean(solver_data.get("obj", 0.0))
        X_prev, U_prev = X[..., 1:, :], U

        t_run = time.time() - t_elaps
        vals = (it + 1, t_run, obj, max_res, reg_x, reg_u, smooth_alpha)
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
