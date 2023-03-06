import time, math # noqa: E401

import numpy as np

from .scp_mpc import print_fn, scp_solve
from .utils import TablePrinter

##^# accelerated SCP ###########################################################
# momentum_update = lambda zk, zkm1, it: zk + it / (it + 3) * (zk - zkm1)
alf = 1.6
def momentum_update(zk, zkm1, it):
    return alf * zk + (1.0 - alf) * zkm1


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
    max_it=100,
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
    tp = TablePrinter(field_names, fmts=fmts)

    t_start = time.time()
    data = {}
    if verbose:
        print_fn(tp.make_header())
    for it in range(max_it):
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
            max_it=1,
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
