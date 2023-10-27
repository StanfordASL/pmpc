from typing import Callable, Tuple, Optional
from functools import partial

from jfi import jaxm
from jfi.experimental.jit import autojit

from jax import Array, numpy as jnp

####################################################################################################


def _rollout_step(f: Callable, x: Array, u_p: Tuple[Array, Array]) -> Tuple[Array, Array]:
    u, p = u_p
    xp = f(x, u, p)
    return xp, xp


@autojit
def rollout(f: Callable, x0: Array, U: Array, P: Optional[Array] = None):
    """Rolls out dynamics into the future based on an initial state x0"""
    xs = [x0[..., None, :]]

    P = (jaxm.broadcast_to(P, U.shape[:-1] + (P.shape[-1],))) if P is not None else U[..., :1] * 0
    xs = jaxm.scan(partial(_rollout_step, f), x0, (U.swapaxes(-2, 0), P.swapaxes(-2, 0)))[
        1
    ].swapaxes(-2, 0)
    return jaxm.cat([x0[..., None, :], xs], -2)


####################################################################################################


def _masked_rollout_step(
    f: Callable, x: Array, x_u_p_mask: Tuple[Array, Array]
) -> Tuple[Array, Array]:
    xp, u, p, mask = x_u_p_mask
    xp = jaxm.where(mask, f(x, u, p), xp)
    return xp, xp


#@autojit
def masked_rollout(
    f: Callable, X: Array, U: Array, P: Optional[Array] = None, mask: Optional[Array] = None
):
    """Rolls out dynamics into the future based on an initial state x0, hard
    constraints x for step i + 1 to existing history where mask is 0 at position i."""
    x0 = X[..., 0:1, :]
    P = (jaxm.broadcast_to(P, U.shape[:-1] + (P.shape[-1],))) if P is not None else U[..., :1] * 0
    mask = mask if mask is not None else jaxm.jax.stop_gradient(1 + U[..., 0] * 0)

    assert U.shape[:-1] == mask.shape
    assert X.shape[-2] - 1 == U.shape[-2]

    Xp_, U_ = X[..., 1:, :].swapaxes(-2, 0), U.swapaxes(-2, 0)
    P_, mask_ = P.swapaxes(-2, 0), mask.swapaxes(-1, 0)
    xs = jaxm.scan(partial(_masked_rollout_step, f), x0, (Xp_, U_, P_, mask_))[1].swapaxes(-2, 0)
    return jaxm.cat([x0[..., None, :], xs], -2)


####################################################################################################

bmv = lambda A, x: (A @ x[..., None])[..., 0]


def _Ft_step(i_Ft, fx_fu):
    i, Ft = i_Ft
    fx, fu = fx_fu
    N = Ft.shape[-1] // fu.shape[-1]
    fu_comp = jaxm.nn.one_hot(i, N, dtype=fx.dtype)[..., None, :, None] * fu[..., :, None, :]
    fu_comp = fu_comp.reshape(fu_comp.shape[:-2] + (-1,))
    ret = (i + 1, fx @ Ft + fu_comp)
    return ret, ret[1]


def _ft_step(ft, fx_f_):
    fx, f_ = fx_f_
    ret = bmv(fx, ft) + f_
    return ret, ret


@jaxm.jit
def dynamics_linear_matrix(
    x0: Array, f: Array, fx: Array, fu: Array, X: Array, U: Array
) -> Tuple[Array, Array]:
    """
    Construct the matrix and bias vector that gives from a local linearization
    vec(X) = Ft @ vec(U) + ft.
    """
    bshape, (N, xdim), udim = fx.shape[:-3], fx.shape[-3:-1], fu.shape[-1]
    X = X[..., :N, :]

    Ft = (
        jaxm.scan(
            _Ft_step,
            (0, jnp.zeros(bshape + (xdim, N * udim), dtype=x0.dtype)),
            (fx.swapaxes(-3, 0), fu.swapaxes(-3, 0)),
        )[1]
        .swapaxes(-3, 0)
        .reshape(bshape + (N * xdim, N * udim))
    )

    f_ = f - bmv(fx, X) - bmv(fu, U)
    ft_init = bmv(fx[..., 0, :, :], x0) + f_[..., 0, :]
    ft = (
        jaxm.scan(
            _ft_step,
            ft_init,
            (fx.swapaxes(-3, 0)[1:, ...], f_.swapaxes(-2, 0)[1:, ...]),
        )[1]
        .swapaxes(-2, 0)
        .reshape(bshape + ((N - 1) * xdim,))
    )
    ft = jaxm.cat([ft_init, ft], -1)
    return Ft, ft
