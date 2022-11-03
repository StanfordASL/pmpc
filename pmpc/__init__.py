from .scp_mpc import *

"""Keyword-compatible arguments in the solver scp solve call."""
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
    "method",
    "solver_settings",
    "solver_state",
    "filter_method",
    "filter_window",
    "filter_it0",
}
