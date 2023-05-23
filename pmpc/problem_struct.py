from functools import partial
from warnings import warn
from copy import copy
from typing import Dict, Tuple
from collections.abc import Mapping

import numpy as np


class Problem(Mapping):
    """A way of initializing an optimal control problem with a majority of
    arguments initialized to defaults."""

    dim_map: Dict[str, Tuple] = {
        "Q": ("N", "xdim", "xdim"),
        "R": ("N", "udim", "udim"),
        "X_ref": ("N", "xdim"),
        "U_ref": ("N", "udim"),
        "X_prev": ("N", "xdim"),
        "U_prev": ("N", "udim"),
        "u_l": ("N", "udim"),
        "u_u": ("N", "udim"),
        "x_l": ("N", "xdim"),
        "x_u": ("N", "xdim"),
        "x0": ("xdim",),
    }

    def _figure_out_dims(self, **kw):
        """Go through keword arguments to figure out dimensions of the problem."""
        dims = dict({k: v for k, v in kw.items() if k in ["N", "xdim", "udim"]})
        for k, v in Problem.dim_map.items():
            if k in kw:
                for i in range(0, -len(v) - 1, -1):
                    dims[Problem.dim_map[k][i]] = kw[k].shape[i]
        for k in ["N", "xdim", "udim"]:
            if k not in dims:
                raise ValueError(f"Missing dimension {k}")
        return dims

    def __init__(self, **kw):
        self._dims = self._figure_out_dims(**kw)
        self._set_defaults()
        self.M = kw.get("M", None)
        for k in Problem.dim_map.keys():
            self._generate_property(k)
        for k in self._dims.keys():
            setattr(Problem, k, property(lambda self, k=k: self._dims[k]))
        for k, v in kw.items():
            if not k.startswith("_"):
                try:
                    setattr(self, k, v)
                except AttributeError:
                    pass
            else:
                warn(f"Cannot set private attribute {k}")
        self._possibly_tile_for_M()
        if not hasattr(self, "Nc"):
            self.Nc = 0

    @property
    def dims(self):
        return copy(self._dims)

    def __repr__(self):
        return f"Problem({self._dims}, id={abs(hash(str(id(self))))})"

    ################################################################################################

    def _generate_property(self, k):
        def _check_dims_and_tile_and_set(k, self, v):
            correct_shape = tuple(self._dims[k_] for k_ in Problem.dim_map[k])
            if self.M is not None:
                correct_shape = (self.M,) + correct_shape
            if v is not None:
                msg = f"v does not have the correct shape, v.shape = {v.shape}, "
                msg = msg + f"correct_shape = {correct_shape[-v.ndim:]}"
                assert v.shape == correct_shape[-v.ndim :], msg
                v = np.array(v)
                v = np.tile(v, correct_shape[: -v.ndim] + ((1,) * v.ndim))
            setattr(self, f"_{k}", v)

        getter = lambda self: getattr(self, f"_{k}")
        setter = partial(_check_dims_and_tile_and_set, k)
        setattr(Problem, k, property(getter, setter))

    ################################################################################################

    def _set_defaults(self, **kw):
        self._Q = np.tile(np.diag(np.ones((self._dims["xdim"],))), (self._dims["N"], 1, 1))
        self._R = np.tile(np.diag(1e-1 * np.ones((self._dims["udim"],))), (self._dims["N"], 1, 1))
        self._x0 = np.zeros((self._dims["xdim"],))
        self._X_ref = np.zeros((self._dims["N"], self._dims["xdim"]))
        self._U_ref = np.zeros((self._dims["N"], self._dims["udim"]))
        self._X_prev = np.tile(self._x0, (self._dims["N"], 1))
        self._U_prev = np.zeros((self._dims["N"], self._dims["udim"]))
        self._u_l, self._u_u, self._x_l, self._x_u = None, None, None, None
        self.solver_settings = dict()
        self.reg_x, self.reg_u, self.max_it, self.res_tol, self.verbose = 1e0, 1e0, 30, 1e-6, True
        self.slew_rate = None
        self.P = None
        for k, v in kw.items():
            setattr(self, f"_{k}", v)

    def _possibly_tile_for_M(self):
        if self.M is None:
            return
        keys = ["_Q", "_R", "_X_ref", "_U_ref", "_X_prev", "_U_prev", "_x0"]
        keys += ["_u_l", "_u_u", "_x_l", "_x_u", "P"]
        for key in keys:
            v = getattr(self, key)
            if v is not None:
                if key[1:] in Problem.dim_map:
                    dim_num = len(Problem.dim_map[key[1:]])
                    assert v.ndim in [dim_num + 1, dim_num]
                    if v.ndim == dim_num:
                        v = np.tile(v, (self.M,) + ((1,) * v.ndim))
                        setattr(self, key, v)

    def to_dict(self):
        # most normal keys
        keys = list(Problem.dim_map.keys())
        keys += ["solver_settings", "reg_x", "reg_u", "max_it", "res_tol", "verbose", "slew_rate"]
        keys += ["P"]
        problem = {k: getattr(self, k, None) for k in keys}
        if self.M is not None:
            if "Nc" in problem["solver_settings"] and problem["solver_settings"]["Nc"] != self.Nc:
                msg = "Nc specified in solver_settings, but Problem specifies Nc via a property."
                msg += f" We will use Nc = {self.Nc} from the Problem."
                warn(msg)
            problem["solver_settings"]["Nc"] = self.Nc

        # dynamics
        if hasattr(self, "f_fx_fu_fn"):
            problem["f_fx_fu_fn"] = self.f_fx_fu_fn
        else:
            warn("No dynamics function specified, please set `prob.f_fx_fu_fn`")

        # optional keys
        optional_keys = ["lin_cost_fn", "extra_cstrs_fns"]
        problem = dict(problem, **{k: getattr(self, k) for k in optional_keys if hasattr(self, k)})

        return problem

    ################################################################################################

    def __iter__(self):
        return iter(self.to_dict().keys())

    def __getitem__(self, k):
        return self.to_dict()[k]

    def __len__(self):
        return len(self.to_dict())

    ################################################################################################
