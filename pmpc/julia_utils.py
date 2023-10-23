#### library imports ###########################################################
import sys
import re
import time
import shutil
from pathlib import Path
from subprocess import check_output

import numpy as np


def get_julia_version(julia_runtime=None):
    """Read the current (major = x.x) Julia version."""
    if julia_runtime is None:
        julia_runtime = shutil.which("julia")
    assert julia_runtime is not None
    output = check_output([julia_runtime, "--version"]).decode("utf-8")
    m = re.search(r"([0-9]+\.[0-9]+)\.[0-9]+", output)
    assert m is not None
    return m.group(1)

JULIA_VERSION = None
PYTHON_VERSION = None
SYSIMAGE_PATH = None


################################################################################
#### loading julia and including the library source files in julia #############
def load_julia(verbose=False, **kw):
    global JULIA_VERSION, PYTHON_VERSION, SYSIMAGE_PATH
    if JULIA_VERSION is None:
        JULIA_VERSION = get_julia_version()
        PYTHON_VERSION = ".".join(map(str, sys.version_info[:3]))
        SYSIMAGE_PATH = Path("~").expanduser() / ".cache" / "pmpc" / f"pmpc_sysimage_j{JULIA_VERSION}_p{PYTHON_VERSION}.so"

    t = time.time()
    try:
        import julia

        #SYSIMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if SYSIMAGE_PATH.exists():
            kw["sysimage"] = str(SYSIMAGE_PATH)

        julia.Julia(**kw)
        recompiled = False
    except Exception as e:
        print(e)
        if verbose:
            print("Using cache failed, recompiling...")
        import julia

        julia.Julia(**dict(kw, compiled_modules=False))
        recompiled = True
    from julia import Main as jl

    jl.using("PMPC")

    if verbose:
        print("Loading Julia took %e s" % (time.time() - t))
        if recompiled:
            print("Consider getting a dynamically linked Python installation.")
            print("Without it, loading Julia will take a long time each time.")
        else:
            print("Using the currently fastest possible way of loading julia.")
    return jl


#### memory layout conversion routines #########################################
def py2jl(x, keep: int = 1):
    # n = len(np.shape(x))
    n = x.ndim
    # keep, batch = keep_ndims, n - keep_ndims
    return np.transpose(
        # x, tuple(range(-keep, 0)) + tuple(range(batch - 1, -1, -1))
        x,
        tuple(range(n - keep, n)) + tuple(range(n - keep - 1, -1, -1)),
    )


def jl2py(x, keep: int = 1):
    # n = len(np.shape(x))
    n = x.ndim
    # keep, batch = keep_ndims, n - keep_ndims
    return np.transpose(
        # x, tuple(range(-1, -batch - 1, -1)) + tuple(range(0, keep))
        x,
        tuple(range(n - 1, keep - 1, -1)) + tuple(range(0, keep)),
    )


#### optimized memory layout conversion routines ###############################
try:
    import jax
    from jax import numpy as jnp

    @jax.jit
    def py2jl_jit(x, order):
        # return jnp.transpose(x, tuple(range(n - keep, n)) + tuple(range(n - keep - 1, -1, -1)))
        return jnp.transpose(x, order)

    @jax.jit
    def jl2py_jit(x, keep_ndims=1):
        n = len(jnp.shape(x))
        keep, batch = keep_ndims, n - keep_ndims
        return jnp.transpose(x, tuple(range(-1, -batch - 1, -1)) + tuple(range(0, keep)))

except ModuleNotFoundError:
    jax, jnp, py2jl_jit, jl2py_jit = None, None, None, None


################################################################################
#### broadcasting and wrappers for batched julia function ######################
def broadcast_args(*args):
    args = [np.atleast_2d(z).reshape((-1, z.shape[-1])) if z is not None else None for z in args]
    return args


def wrap_fn(fn, out_dims=1):
    def closure(*args, **kwargs):
        xshape = args[0].shape
        bargs = [arg if isinstance(arg, np.ndarray) else None for arg in args]
        bargs = broadcast_args(*bargs)
        args = [
            py2jl(bargs[i], 1) if isinstance(args[i], np.ndarray) else args[i]
            for i in range(len(args))
        ]
        ret = jl2py(fn(*args, **kwargs), out_dims)
        return ret.reshape(xshape[:-1] + ret.shape[1:])

    return closure


################################################################################
