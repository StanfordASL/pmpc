from typing import Callable, Any, Dict, Optional, List
from enum import Enum
from functools import partial

import cloudpickle as cp
from jfi import jaxm
import jaxopt

from .second_order_solvers import ConvexSolver

Array = jaxm.jax.Array

class Solver(Enum):
    BFGS = 0
    LBFGS = 1
    CVX = 2

SOLVERS_STORE = dict()

####################################################################################################

def get_pinit_state(obj_fn: Callable, **kw):
    obj_fn_key = cp.dumps(obj_fn)
    if obj_fn_key not in SOLVERS_STORE:
        SOLVERS_STORE[obj_fn_key] = generate_routines_for_obj_fn(obj_fn, **kw)
    return SOLVERS_STORE[obj_fn_key]["pinit_state"]

def get_prun_with_state(obj_fn: Callable, **kw):
    obj_fn_key = cp.dumps(obj_fn)
    if obj_fn_key not in SOLVERS_STORE:
        SOLVERS_STORE[obj_fn_key] = generate_routines_for_obj_fn(obj_fn, **kw)
    return SOLVERS_STORE[obj_fn_key]["prun_with_state"]

####################################################################################################

def generate_routines_for_obj_fn(
    obj_fn: Callable, solver_opts: Optional[Dict[str, Any]] = None, **kw,
):
    opts = dict(
        kw.get("solver_opts", dict()),
        maxiter=100,
        verbose=False,
        jit=True,
        tol=1e-7,
        linesearch="backtracking",
    )
    try:
        jaxm.jax.devices("gpu")
        device = "cuda"
    except RuntimeError:
        device = "cpu"

    solvers = {
        Solver.BFGS: jaxopt.BFGS(obj_fn, **opts),
        Solver.LBFGS: jaxopt.LBFGS(obj_fn, **opts),
        Solver.CVX: ConvexSolver(
            obj_fn, **dict(opts, maxls=25, linesearch="binary_search", device=device)
        ),
    }
    run_methods = {k: jaxm.jit(solver.run) for k, solver in solvers.items()}
    update_methods = {k: jaxm.jit(solver.update) for k, solver in solvers.items()}

    @partial(jaxm.jit, static_argnums=(0,))
    def run_with_state(
        solver: int, z: Array, args: Dict[str, List[Array]], state, max_it: int = 100
    ):
        def body_fn(i, z_state):
            return update_methods[solver](*z_state, args)

        z_state = body_fn(0, (z, state))
        return jaxm.jax.lax.fori_loop(1, max_it, body_fn, z_state)

    @partial(jaxm.jit, static_argnums=(0,))
    def init_state(solver: int, U_prev: Array, args: Dict[str, List[Array]]):
        return solvers[solver].init_state(U_prev, args)

    @partial(jaxm.jit, static_argnums=(0,))
    def prun_with_state(
        solver: int, z: Array, args: Dict[str, List[Array]], state, max_it: int = 100
    ):
        in_axes = jaxm.jax.tree_util.tree_map(
            lambda x: 0
            if (hasattr(x, "shape") and x.ndim > 0 and x.shape[0] == z.shape[0])
            else None,
            (solver, z, args, state, max_it),
        )
        return jaxm.jax.vmap(run_with_state, in_axes=in_axes)(solver, z, args, state, max_it)

    @partial(jaxm.jit, static_argnums=(0,))
    def pinit_state(solver: int, U_prev: Array, args: Dict[str, List[Array]]):
        in_axes = jaxm.jax.tree_util.tree_map(
            lambda x: 0
            if (hasattr(x, "shape") and x.ndim > 0 and x.shape[0] == U_prev.shape[0])
            else None,
            (U_prev, args),
        )
        return jaxm.jax.vmap(solvers[solver].init_state, in_axes=in_axes)(U_prev, args)

    return dict(
        solvers=solvers,
        run_methods=run_methods,
        update_methods=update_methods,
        init_state=init_state,
        run_with_state=run_with_state,
        prun_with_state=prun_with_state,
        pinit_state=pinit_state,
    )
