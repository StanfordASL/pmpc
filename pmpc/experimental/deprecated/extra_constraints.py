from typing import NamedTuple, List, Dict
from jfi import jaxm

Array = jaxm.jax.Array

class ConeCostraint(NamedTuple):
    l: int
    q: List[int]
    e: int
    G_left: Array
    G_right: Array
    h: Array
    c_left: Array
    c_right: Array


class LinearConstraint(NamedTuple):
    A: Array
    b: Array


def lin_cstrs_obj(cstr: LinearConstraint, z: Array, args: Dict[str, List[Array]]) -> Array:
    A, b = cstr
    s = A @ z - b
    alpha = args["cstr"][-1]
    J = -jaxm.mean(jaxm.log(-alpha * s)) / alpha
    return jaxm.where(jaxm.isfinite(J), J, jaxm.inf)


class ExpConstraint(NamedTuple):
    G: Array
    h: Array


def exp_cstrs_obj(cstr: LinearConstraint, z: Array, args: Dict[str, List[Array]]) -> Array:
    G, h = cstr
    s = G @ z - h
    s = jaxm.stack([s[1] * jaxm.exp(s[0] / s[1]) - s[2], -s[1]])
    alpha = args["cstr"][-1]
    J = -jaxm.mean(jaxm.log(-alpha * s)) / alpha
    return jaxm.where(jaxm.isfinite(J), J, jaxm.inf)


def exp_cstrs_obj(cstr: LinearConstraint, z: Array, args: Dict[str, List[Array]]) -> Array:
    G, h = cstr
    s = G @ z - h
    s = jaxm.stack([s[1] * jaxm.exp(s[0] / s[1]) - s[2], -s[1]])
    alpha = args["cstr"][-1]
    J = -jaxm.mean(jaxm.log(-alpha * s)) / alpha
    return jaxm.where(jaxm.isfinite(J), J, jaxm.inf)


#@jaxm.jit
#def obj_with_extra_cstrs_fn(U: Array, args: Dict[str, List[Array]]) -> Array:
#    J = obj_fn(U, args)
#    return J
