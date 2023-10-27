from typing import Callable
from functools import partial

from jfi import jaxm
from jfi.experimental.jit import nestedautojit

from jax import Array, numpy as jnp

from .dynamics import dynamics_linear_matrix, bmv, rollout, masked_rollout

####################################################################################################


def _slew_optimality(U):
    L_mid = 2 * U[..., 1:-1, :] - U[..., :-2, :] - U[..., 2:, :]
    L_1st = U[..., 0:1, :] - U[..., 1:2, :]
    L_lst = U[..., -1:, :] - U[..., -2:-1, :]
    return jaxm.cat([L_1st, L_mid, L_lst], -2)


def log_penalty(A, b, x, s):
    return -jaxm.sum(jaxm.log(-s * (bmv(A, x) - b)), -1) / s


def dlog_penalty(A, b, x, s):
    return bmv(-A.swapaxes(-2, -1), (1 / (bmv(A, x) - b) / s))


####################################################################################################


def linearly_approximate(f_fx_fu_fn: Callable, fn: Callable, stop_grad: bool = True):
    """A function that wraps a dynamics dependent function adding new first two
    arguments: Ft, ft -- the dynamics matrix linear approximation."""

    def fn_out(X, U, problem, *args, **kw):
        x0_, U_, P_, X_ = problem["x0"], U, problem.get("P"), X[..., :-1, :]
        if stop_grad:
            x0_ = jaxm.stop_gradient(x0_)
            U_ = jaxm.stop_gradient(U_)
            P_ = jaxm.stop_gradient(P_) if P_ is not None else None
            X_ = jaxm.stop_gradient(X_)
        f, fx, fu = f_fx_fu_fn(X_, U_, P_)
        Ft, ft = dynamics_linear_matrix(x0_, f, fx, fu, X_, U_)

        # X defined as an affine approximation wrt U
        X = bmv(Ft, U.reshape(U.shape[:-2] + (-1,))) + ft
        X = jaxm.cat([x0_[..., None, :], X.reshape(X_.shape)], -2)

        return fn(Ft, ft, X, U, problem, *args, **kw)

    return fn_out


def rollout_approximate(f_fx_fu_fn: Callable, fn: Callable):
    """A function that wraps a dynamics dependent function adding new first two
    arguments: Ft, ft -- the dynamics matrix linear approximation."""
    dyn_fn = lambda x, u, p: f_fx_fu_fn(x, u, p)[0]

    def fn_out(X, U, problem, *args, **kw):
        x0_, U_, P_, X_ = problem["x0"], U, problem["P"], X[..., :-1, :]
        f, fx, fu = f_fx_fu_fn(X_, U_, P_)
        Ft, ft = dynamics_linear_matrix(x0_, f, fx, fu, X_, U_)
        X = rollout(dyn_fn, x0_, U, P_)
        return fn(Ft, ft, X, U, problem, *args, **kw)

    return fn_out


def masked_rollout_approximate(f_fx_fu_fn: Callable, fn: Callable, mask: Array):
    """A function that wraps a dynamics dependent function adding new first two
    arguments: Ft, ft -- the dynamics matrix linear approximation."""
    dyn_fn = lambda x, u, p: f_fx_fu_fn(x, u, p)[0]

    def fn_out(X, U, problem, *args, **kw):
        x0, U_, P_, X_ = X[..., 0, :], U, problem.get("P"), X[..., :-1, :]
        f, fx, fu = f_fx_fu_fn(X_, U_, P_)
        Ft, ft = dynamics_linear_matrix(x0, f, fx, fu, X_, U_)
        X = masked_rollout(dyn_fn, X, U, P_, mask)
        return fn(Ft, ft, X, U, problem, *args, **kw)

    return fn_out


####################################################################################################


@nestedautojit
def linear_optimality(Ft, ft, X, U, problem):
    bshape = U.shape[:-2]
    vec = lambda x: x.reshape(bshape + (-1,))
    Q, R = problem["Q"], problem["R"]
    U_ref = problem.get("U_ref", jaxm.zeros(R.shape[:-1], dtype=R.dtype))
    X_ref = problem.get("X_ref", jaxm.zeros(Q.shape[:-1], dtype=Q.dtype))

    L_u = bmv(R, U - U_ref).reshape(bshape + (-1,))
    L_x = bmv(Q, X[..., 1:, :] - X_ref).reshape(bshape + (-1,))
    L_x = bmv(jaxm.t(Ft), L_x)

    # handle smoothed constraints ##################################################################
    solver_settings = problem.get("solver_settings", dict())
    if solver_settings.get("smooth_cstr") or "smooth_alpha" in solver_settings:
        alf = solver_settings["smooth_alpha"]
        N, udim = U.shape[-2:]
        G = jaxm.tile(
            jaxm.cat([jnp.eye(N * udim, dtype=U.dtype), -jnp.eye(N * udim, dtype=U.dtype)], -2),
            bshape + (1, 1),
        )
        h = jaxm.cat([vec(problem["u_u"]), -vec(problem["u_l"])], -1)
        L_cstr = dlog_penalty(G, h, U.reshape(bshape + (N * udim,)), alf)
    else:
        L_cstr = 0

    # include extra constraints influence (smoothly only) ##########################################
    L_extra_cstrs = []
    for extra_cstr in problem.get("solver_settings", dict()).get("extra_cstrs", []):
        # we only support a very small subset of extra constraints - log smooth linear inequalities
        l, q, e, G_left, G_right, h, c_left, c_right = extra_cstr
        assert l >= 0 and sum(q) == 0 and e == 0, "We support only linear inequalities"
        assert jaxm.array(G_right).shape[1] == 0, "The right array (G_right) must be empty"
        G_left, G_right, h = [jaxm.array(z) for z in [G_left, G_right, h]]
        L_extra_cstrs.append(
            jaxm.grad(
                lambda U: log_penalty(G_left, h, jaxm.cat([vec(U), bmv(Ft, vec(U)) + ft], -1), alf)
            )(U)
        )

    # compute slew rate optimality conditions ######################################################
    L_slew = problem.get("slew_rate", 0.0) * _slew_optimality(U)
    L_slew = L_slew.reshape(bshape + (-1,))

    # cost function optimality
    L_cost = 0
    if problem.get("lin_cost_fn", None) is not None:
        cx, cu = problem["lin_cost_fn"](X[..., 1:, :], U, problem)
        if cx is not None:
            L_cost = L_cost + bmv(jaxm.t(Ft), cx.reshape(bshape + (-1,)))
        if cu is not None:
            L_cost = L_cost + cu.reshape(bshape + (-1,))

    L = L_u + L_x + L_cstr + L_slew + L_cost + sum(vec(z) for z in L_extra_cstrs)
    return L.reshape(U.shape)


####################################################################################################


@partial(nestedautojit, jit_ints=True)
def compute_sensitivity_L(X, U, idx, problem):
    assert X.ndim == 2 and U.ndim == 2
    assert X.shape[-2] - 1 == U.shape[-2]
    N = U.shape[-2]
    mask = jaxm.ones((N + 1,)).at[idx].set(0.0)[1:]
    fn = masked_rollout_approximate(problem["f_fx_fu_fn"], linear_optimality, mask)
    dX, dU = jaxm.jacobian(lambda X, U: fn(X, U, problem), argnums=(0, 1))(X, U)
    dX = jaxm.lax.dynamic_slice(dX, (0, 0, idx, 0), U.shape + (1, X.shape[-1]))
    K, g = dU.reshape((U.size,) * 2), dX.reshape((U.size, X.shape[-1]))
    L = -jaxm.linalg.solve(K, g)
    L = jaxm.lax.dynamic_slice(L, (2 * idx, 0), (U.shape[-1], X.shape[-1]))
    return L


@partial(nestedautojit, jit_ints=True)
def all_sensitivity_L(X, U, problem):
    idxs = jaxm.arange(U.shape[-2])
    return jaxm.scan(
        lambda carry, idx: (None, compute_sensitivity_L(X, U, idx, problem)), None, idxs
    )[1]
    # return jaxm.vmap(lambda idx: compute_sensitivity_L(X, U, idx, problem))(
    # idxs
    # )
