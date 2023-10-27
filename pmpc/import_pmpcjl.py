from __future__ import annotations

import re
import os
from pathlib import Path
import ctypes
from importlib import import_module
from importlib.util import find_spec

def _get_libname(libfile: str):
    return re.match(r"(lib[^\.]*)\..*", libfile).group(1)


def find_libraries(root_path: Path | str):
    lib_path = Path(root_path) / "PMPC.jl" / "pmpcjl" / "lib"
    #libraries = sum(
    #    [
    #        [Path(root).absolute() / fname for fname in fnames]
    #        for (root, _, fnames) in os.walk(lib_path)
    #    ],
    #    [],
    #)
    libraries = [lib_path / f for f in os.listdir(lib_path)]
    libraries = [f for f in libraries if re.match("lib.*", f.name)]
    libraries = {_get_libname(f.name): f for f in libraries}
    libraries = list(libraries.values())
    return libraries


def import_pmpcjl():
    if find_spec("pmpcjl") is None:
        return None
    root_path = Path(__file__).parents[1]
    libraries = find_libraries(root_path)
    for lib in libraries:
        ctypes.cdll.LoadLibrary(str(lib))
    pmpcjl = import_module("pmpcjl")
    return pmpcjl
