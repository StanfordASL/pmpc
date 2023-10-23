from __future__ import annotations

import re
import os
from pathlib import Path
import ctypes
from importlib import import_module

def find_libraries(root_path: Path | str):
    lib_path = Path(root_path)/ "PMPC.jl" / "pmpcjl" / "lib"
    libraries = [lib_path / f for f in os.listdir(lib_path)]
    libraries = [f for f in libraries if re.match("lib.*", f.name)]
    return libraries


def import_pmpcjl():
    root_path = Path(__file__).parents[1]
    libraries = find_libraries(root_path)
    for lib in libraries:
        ctypes.cdll.LoadLibrary(str(lib))
    pmpcjl = import_module("pmpcjl")
    return pmpcjl