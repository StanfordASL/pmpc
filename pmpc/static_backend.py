from typing import Any, Dict, Optional, Tuple, Union
from copy import copy
import ctypes

import numpy as np

from .import_pmpcjl import import_pmpcjl
from .utils import atleast_nd, to_numpy_f64
from . import julia_utils as ju


try:
    pmpcjl = import_pmpcjl()
except:
    pmpcjl = None


def is_precompiled_backend_available():
    global pmpcjl
    return pmpcjl is not None


def lqp_solve(
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
    verbose=False,
):
    assert x0.ndim == 2
    xdim, M = x0.shape
    assert f.ndim == 3
    N = f.shape[1]
    assert fu.ndim == 4
    udim = fu.shape[1]
    assert x0.shape == (xdim, M), x0.shape
    assert f.shape == (xdim, N, M), f.shape
    assert fx.shape == (xdim, xdim, N, M)
    assert fu.shape == (xdim, udim, N, M)
    assert X_prev.shape == (xdim, N, M)
    assert U_prev.shape == (udim, N, M)
    assert Q.shape == (xdim, xdim, N, M)
    assert R.shape == (udim, udim, N, M)
    assert X_ref.shape == (xdim, N, M)
    assert U_ref.shape == (udim, N, M)
    assert lx.shape == (xdim, N, M), lx.shape
    assert ux.shape == (xdim, N, M), ux.shape
    assert lu.shape == (udim, N, M), lu.shape
    assert uu.shape == (udim, N, M), uu.shape
    assert slew_um1.shape == (udim, M)
    assert slew_reg.shape == (M,)
    assert slew_reg0.shape == (M,), slew_reg0.shape

    x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref, slew_reg, slew_reg0, slew_um1 = [
        np.asfortranarray(z)
        for z in [x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref, slew_reg, slew_reg0, slew_um1]
    ]
    args = (
        int(xdim),
        int(udim),
        int(N),
        int(M),
        int(Nc),
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
        float(reg_x),
        float(reg_u),
        slew_reg,
        slew_reg0,
        slew_um1,
        int(verbose),
    )
    X, U = pmpcjl.lqp_solve(*args)
    X, U = np.reshape(X, (M, N, xdim)), np.reshape(U, (M, N, udim))
    return X, U


def lcone_solve(
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
    smooth_alpha=1e1,
    verbose=False,
):
    assert x0.ndim == 2
    xdim, M = x0.shape
    assert f.ndim == 3
    N = f.shape[1]
    assert fu.ndim == 4
    udim = fu.shape[1]
    assert x0.shape == (xdim, M), x0.shape
    assert f.shape == (xdim, N, M), f.shape
    assert fx.shape == (xdim, xdim, N, M)
    assert fu.shape == (xdim, udim, N, M)
    assert X_prev.shape == (xdim, N, M)
    assert U_prev.shape == (udim, N, M)
    assert Q.shape == (xdim, xdim, N, M)
    assert R.shape == (udim, udim, N, M)
    assert X_ref.shape == (xdim, N, M)
    assert U_ref.shape == (udim, N, M)
    assert lx.shape == (xdim, N, M), lx.shape
    assert ux.shape == (xdim, N, M), ux.shape
    assert lu.shape == (udim, N, M), lu.shape
    assert uu.shape == (udim, N, M), uu.shape
    assert slew_um1.shape == (udim, M)
    assert slew_reg.shape == (M,)
    assert slew_reg0.shape == (M,), slew_reg0.shape

    x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref, slew_reg, slew_reg0, slew_um1 = [
        np.asfortranarray(z)
        for z in [x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref, slew_reg, slew_reg0, slew_um1]
    ]
    args = (
        int(xdim),
        int(udim),
        int(N),
        int(M),
        int(Nc),
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
        float(reg_x),
        float(reg_u),
        slew_reg,
        slew_reg0,
        slew_um1,
        int(verbose),
        float(smooth_alpha),
    )
    X, U = pmpcjl.lcone_solve(*args)
    X, U = np.reshape(X, (M, N, xdim)), np.reshape(U, (M, N, udim))
    return X, U


#JULIA_SOLVE_FNS = dict(qp=lqp_solve, cone=lcone_solve)
JULIA_SOLVE_FNS = dict(cone=lcone_solve)


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
    """Solve a single instance of a linearized MPC problem."""
    assert pmpcjl is not None

    f = atleast_nd(f, 3)
    fx, fu = atleast_nd(fx, 4), atleast_nd(fu, 4)
    x0 = atleast_nd(x0, 2)
    X_prev, U_prev = atleast_nd(X_prev, 3), atleast_nd(U_prev, 3)
    Q, R = atleast_nd(Q, 4), atleast_nd(R, 4)
    X_ref, U_ref = atleast_nd(X_ref, 3), atleast_nd(U_ref, 3)
    x_l, x_u, u_l, u_u = [atleast_nd(z, 3) for z in [x_l, x_u, u_l, u_u]]

    M = x0.shape[-1]

    x_l, x_u, u_l, u_u = [ju.py2jl(to_numpy_f64(z), 1) for z in [x_l, x_u, u_l, u_u]]
    args = [
        ju.py2jl(to_numpy_f64(z), d)
        for (z, d) in zip(
            [x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref],
            [1, 1, 2, 2, 1, 1, 2, 2, 1, 1],
        )
    ]
    x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref = args
    solver_settings.setdefault("solver", "ecos")
    if "smooth_cstr" in solver_settings or "smooth_alpha" in solver_settings:
        method = "cone"
        smooth_alpha = solver_settings.get("smooth_alpha", 1e0)
    else:
        method = "qp"
        smooth_alpha = None

    solver_settings = copy(solver_settings) if solver_settings is not None else dict()

    Nc = -1 if "Nc" not in solver_settings else solver_settings["Nc"]
    x_l = np.nan * np.zeros_like(X_prev) if x_l is None or x_l.size == 0 else x_l
    x_u = np.nan * np.zeros_like(X_prev) if x_u is None or x_u.size == 0 else x_u
    u_l = np.nan * np.zeros_like(U_prev) if u_l is None or u_l.size == 0 else u_l
    u_u = np.nan * np.zeros_like(U_prev) if u_u is None or u_u.size == 0 else u_u
    slew_reg = slew_rate * np.ones_like(x0[0, :])
    slew_reg0 = (
        np.nan * np.zeros((x0.shape[-1],))
        if "slew_reg" not in solver_settings
        else solver_settings["slew_reg"] * np.ones_like(x0[0, :])
    )
    slew_um1 = (
        np.nan * np.zeros_like(U_prev[:, 0, :])
        if u_slew is None
        else ju.py2jl(to_numpy_f64(atleast_nd(u_slew, 2)), 1)
    )

    args = (
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
        x_l,
        x_u,
        u_l,
        u_u,
        reg_x,
        reg_u,
        slew_reg,
        slew_reg0,
        slew_um1,
    )

    #X, U = solve_fn(*args, verbose=solver_settings.get("verbose", False))
    if method == "qp":
        X, U = lqp_solve(*args, verbose=solver_settings.get("verbose", False))
    else:
        X, U = lcone_solve(*args, smooth_alpha, verbose=solver_settings.get("verbose", False))
    X_traj, U_traj = np.concatenate([x0.swapaxes(-1, -2)[:, None, :], X], -2), U
    return X_traj, U_traj, dict()
