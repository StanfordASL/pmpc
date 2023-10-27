#!/usr/bin/env python3
import os
import shutil
import sys
import re
from typing import Union, Optional
from pathlib import Path
from subprocess import check_output, check_call

pmpc_path = Path(__file__).absolute().parents[1]
compilation_utils_path = Path(pmpc_path) / "scripts" / "compilation_utils.jl"

PathT = Union[Path, str]

DISABLE_PRECOMPILATION = True

####################################################################################################


def is_python_statically_linked() -> bool:
    # check if os is linux or mac
    assert sys.platform in ["linux", "darwin"]
    return re.search("libpython", check_output(["ldd", sys.executable]).decode("utf-8")) is None


def get_julia_version(julia_runtime: Optional[PathT] = None) -> str:
    """Read the current (major = x.x) Julia version."""
    if julia_runtime is None:
        julia_runtime = shutil.which("julia")
    assert julia_runtime is not None
    output = check_output([julia_runtime, "--version"]).decode("utf-8")
    m = re.search(r"([0-9]+\.[0-9]+)\.[0-9]+", output)
    assert m is not None
    return m.group(1)


def install_package_julia_version(julia_runtime: Optional[PathT] = None) -> None:
    """Install the version of the PMPC package for the current (major = x.x) Julia version."""
    if julia_runtime is None:
        julia_runtime = shutil.which("julia")
    assert julia_runtime is not None
    version = get_julia_version(julia_runtime)

    # copy the trace file ######################################################

    # run Julia within Python to fix the precompile file #######################
    if not DISABLE_PRECOMPILATION:
        trace_path = pmpc_path / "src" / "traces" / f"trace_{version}.jl"
        shutil.copy(Path(__file__).absolute().parent / f"trace_{version}.jl", trace_path)
        precompile_file = Path(__file__).absolute().parents[1] / "src" / "precompile.jl"
        if precompile_file.exists():
            os.remove(precompile_file)
        python_prog = f"""
import julia
julia.install()
from julia import Julia
try:
    Julia(runtime="{julia_runtime}")
except:
    Julia(runtime="{julia_runtime}", compiled_modules=False)

from julia import Main as jl
jl.using("Pkg")
jl.eval('Pkg.develop(PackageSpec(path="{str(pmpc_path)}"))')
jl.eval('Pkg.add("PackageCompiler")')
jl.using("PackageCompiler")
jl.eval('Pkg.activate("{str(pmpc_path)}")')
jl.eval('Pkg.instantiate()')
jl.eval('Pkg.resolve()')
jl.include("{str(compilation_utils_path)}")
jl.fix_tracefile("{str(trace_path)}")
    """
        check_call([sys.executable, "-c", python_prog])

        julia_prog = f"""
include("{str(compilation_utils_path)}")
fix_tracefile("{str(trace_path)}")
    """
        check_call([julia_runtime, "-e", julia_prog])
        # copy the now fixed precompile file #######################################
        shutil.copy(
            pmpc_path / "src" / "traces" / f"trace_{version}.jl", pmpc_path / "src" / "precompile.jl"
        )
        shutil.copy(pmpc_path / "versions" / f"Manifest.toml_{version}", pmpc_path / "Manifest.toml")
        shutil.copy(pmpc_path / "versions" / f"Project.toml_{version}", pmpc_path / "Project.toml")

    # install the final package ################################################
    julia_prog = f"""
using Pkg
Pkg.develop(PackageSpec(path="{str(pmpc_path)}"))
    """
    check_call([julia_runtime, "-e", julia_prog])


####################################################################################################


def generate_for_julia_version(julia_runtime: Optional[PathT] = None) -> None:
    """Generate a version of the PMPC package for the current (major = x.x) Julia version."""
    if julia_runtime is None:
        julia_runtime = shutil.which("julia")
    version = get_julia_version(julia_runtime)

    if (pmpc_path / "src" / "precompile.jl").exists():
        os.remove(pmpc_path / "src" / "precompile.jl")
    (pmpc_path / "src" / "precompile.jl").touch()
    # update the Manifest and Project files and store their version in `versions`
    julia_prog = f"""
using Pkg
Pkg.activate("{str(pmpc_path)}")
Pkg.update()
Pkg.activate()
Pkg.develop(PackageSpec(path="{str(pmpc_path)}"))
    """
    check_call([julia_runtime, "-e", julia_prog])
    shutil.copy(pmpc_path / "Manifest.toml", pmpc_path / "versions" / f"Manifest.toml_{version}")
    shutil.copy(pmpc_path / "Project.toml", pmpc_path / "versions" / f"Project.toml_{version}")
    # trace compilation
    python_prog = f"""
from julia import Julia
Julia(runtime="{julia_runtime}", trace_compile="trace_{version}.jl")
from pmpc.remote import precompilation_call
precompilation_call()
    """
    check_call([sys.executable, "-c", python_prog])
    # fix the trace file
    julia_prog = f"""
using Pkg
Pkg.activate("{str(pmpc_path)}")
Pkg.add("PackageCompiler")
include("{str(compilation_utils_path)}")
fix_tracefile("trace_{version}.jl")
    """
    check_call([julia_runtime, "-e", julia_prog])
    shutil.copy(Path(f"trace_{version}.jl"), pmpc_path / "src" / "traces" / f"trace_{version}.jl")


####################################################################################################


def make_sysimage(julia_runtime:Optional[PathT]=None):
    """Generate a sysimage for a particular Julia version."""
    if julia_runtime is None:
        julia_runtime = shutil.which("julia")
    assert julia_runtime is not None
    version = get_julia_version(julia_runtime)

    python_version = ".".join([str(x) for x in sys.version_info[:3]])
    sysimage_path = (
        Path("~").expanduser()
        / ".cache"
        / "pmpc"
        / f"pmpc_sysimage_j{version}_p{python_version}.so"
    )
    if sysimage_path.exists():
        return

    sysimage_path.parent.mkdir(parents=True, exist_ok=True)
    if DISABLE_PRECOMPILATION or is_python_statically_linked():
        check_call([sys.executable, "-m", "julia.sysimage", "pmpc_sysimage.so"])
    else:
        trace_path = pmpc_path / "src" / "traces" / f"trace_{version}.jl"
        python_prog = f"""
import julia
from julia import Julia
try:
    Julia(runtime="{julia_runtime}")
except:
    Julia(runtime="{julia_runtime}", compiled_modules=False)

from julia import Main as jl
jl.using("Pkg")
jl.eval('Pkg.add("PackageCompiler")')
jl.using("PackageCompiler")
jl.include("{str(compilation_utils_path)}")
jl.eval('Pkg.activate("{str(pmpc_path)}")')
jl.eval('Pkg.instantiate()')
jl.eval('Pkg.resolve()')
jl.tracefile2sysimage("{str(trace_path)}")
    """
        check_call([sys.executable, "-c", python_prog])
    os.rename("pmpc_sysimage.so", sysimage_path)


####################################################################################################

if __name__ == "__main__":
    juliaup_cmd = str(Path("~").expanduser() / ".juliaup" / "bin" / "juliaup")
    julia_runtime = str(Path("~").expanduser() / ".juliaup" / "bin" / "julia")
    versions = ["1.6", "1.7", "1.8", "1.9"]

    # for version in versions:
    #    check_call([juliaup_cmd, "default", version])
    #    generate_for_julia_version(julia_runtime)

    for version in ["1.8"]:
        check_call([juliaup_cmd, "default", version])
        install_package_julia_version(julia_runtime)
        make_sysimage(julia_runtime)
