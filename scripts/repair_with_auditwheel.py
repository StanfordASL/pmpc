#!/usr/bin/env python3

import sys
import os
import re
from subprocess import check_call
from pathlib import Path

import sys
import platform

version_info = sys.version_info
implementation = platform.python_implementation()


if __name__ == "__main__":
    files = sum(
        [
            [Path(root) / f for f in fnames if re.match(r".*\.so(\..*|)", f)]
            for (root, _, fnames) in os.walk("PMPC.jl/pmpcjl")
        ],
        [],
    )

    if implementation == "CPython":
        tag = f"cp{version_info.major}{version_info.minor}"
        abi = tag  # CPython ABI tag is the same as its version tag
    else:
        assert False, "No support for no CPython"
    exec_version = f"{tag}-{abi}"
    print(f"exec_version: {exec_version}")

    wheel_names = [
        Path("wheelhouse") / f
        for f in os.listdir("wheelhouse")
        if re.match(f"pmpc.*{exec_version}.*\.whl", f)
    ]
    extra_libs_to_exclude = ["libgfortran.so"] + [f"libgfortran.so.{i}" for i in range(30)]
    cmd = (
        [sys.executable, "-m", "auditwheel", "repair"]
        + sum([["--exclude", fname.name] for fname in files], [])
        + sum([["--exclude", extra_lib] for extra_lib in extra_libs_to_exclude], [])
    )
    for wheel in wheel_names:
        check_call(cmd + [str(wheel)])
