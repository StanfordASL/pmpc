from typing import Any

from jfi import jaxm

tree_map = jaxm.jax.tree_util.tree_map


def _is_numeric(x):
    """Check whether and object can be represented as a JAX array."""
    try:
        jaxm.array(x)
        return True
    except TypeError:
        return False


def _to_dtype_device(d: Any, device=None, dtype=None):
    """Convert an arbitrary nested python object to specified dtype and device."""
    return tree_map(
        lambda x: jaxm.to(jaxm.array(x), dtype=dtype, device=device) if _is_numeric(x) else None, d
    )


def _jax_sanitize(x: Any) -> Any:
    """Replace all data that cannot be expressed as a JAX array (e.g., str) with None"""
    return tree_map(lambda x: x if _is_numeric(x) else None, x)
