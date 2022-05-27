##^# library imports ###########################################################
import os, sys, time
import numpy as np


##$#############################################################################
##^# loading julia and including the library source files in julia #############
def load_julia(verbose=False):
    t = time.time()
    try:
        import julia

        # path = os.path.abspath(os.path.expanduser("~/.local/lib/sys_pmpc.so"))
        # if os.path.isfile(path):
        #    julia.Julia(sysimage=path)
        # else:
        #    julia.Julia()
        julia.Julia()
        recompiled = False
    except Exception as e:
        print(e)
        if verbose:
            print("Using cache failed, recompiling...")
        import julia

        julia.Julia(compiled_modules=False)
        recompiled = True
    from julia import Main as jl

    # jl.using("PMPC")
    jl.include(
        os.path.expanduser(
            os.path.join(
                "~/Dropbox/stanford",
                "sensitivity_analysis/pmpc/PMPC.jl/src/PMPC.jl",
            )
        )
    )

    if verbose:
        print("Loading Julia took %e s" % (time.time() - t))
        if recompiled:
            print("Consider getting a dynamically linked Python installation.")
            print("Without it, loading Julia will take a long time each time.")
        else:
            print("Using the currently fastest possible way of loading julia.")
    return jl


##^# memory layout conversion routines #########################################
def py2jl(x, keep_ndims=1):
    n = len(np.shape(x))
    keep, batch = keep_ndims, n - keep_ndims
    return np.transpose(
        x, tuple(range(-keep, 0)) + tuple(range(batch - 1, -1, -1))
    )


def jl2py(x, keep_ndims=1):
    n = len(np.shape(x))
    keep, batch = keep_ndims, n - keep_ndims
    return np.transpose(
        x, tuple(range(-1, -batch - 1, -1)) + tuple(range(0, keep))
    )


##$#############################################################################
##^# broadcasting and wrappers for batched julia function ######################
def broadcast_args(*args):
    args = [
        np.atleast_2d(z).reshape((-1, z.shape[-1])) if z is not None else None
        for z in args
    ]
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


##$#############################################################################
