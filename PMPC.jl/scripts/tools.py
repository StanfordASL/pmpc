#!/usr/bin/env python3
import os
import shutil
import sys
import re
from pathlib import Path
from subprocess import check_output, check_call

pmpc_path = str(Path(__file__).absolute().parents[1])
compilation_utils_path = str(Path(pmpc_path) / "scripts" / "compilation_utils.jl")

####################################################################################################

def get_julia_version(julia_runtime=None):
    """Read the current (major = x.x) Julia version."""
    if julia_runtime is None:
        julia_runtime = shutil.which("julia")
    assert julia_runtime is not None
    output = check_output([julia_runtime, "--version"]).decode("utf-8")
    m = re.search(r"([0-9]+\.[0-9]+)\.[0-9]+", output)
    assert m is not None
    return m.group(1)


def install_package_julia_version(julia_runtime=None):
    """Install the version of the PMPC package for the current (major = x.x) Julia version."""
    if julia_runtime is None:
        julia_runtime = shutil.which("julia")
    assert julia_runtime is not None
    version = get_julia_version(julia_runtime)
    trace_path = str(Path(pmpc_path) / "src" / "traces" / f"trace_{version}.jl")
    julia_prog = f"""
using Pkg
Pkg.activate("{pmpc_path}")
Pkg.add("PackageCompiler")
include("{compilation_utils_path}")
fix_tracefile("{trace_path}")
    """
    check_call([julia_runtime, "-e", julia_prog])
    shutil.copy(
        Path(pmpc_path) / "src" / "traces" / f"trace_{version}.jl",
        Path(pmpc_path) / "src" / "precompile.jl",
    )
    shutil.copy(
        Path(pmpc_path) / "versions" / f"Manifest.toml_{version}", Path(pmpc_path) / "Manifest.toml"
    )
    shutil.copy(
        Path(pmpc_path) / "versions" / f"Project.toml_{version}", Path(pmpc_path) / "Project.toml"
    )
    julia_prog = f"""
using Pkg
Pkg.develop(PackageSpec(path="{pmpc_path}"))
    """
    check_call([julia_runtime, "-e", julia_prog])


####################################################################################################


def generate_for_julia_version(julia_runtime=None):
    """Generate a version of the PMPC package for the current (major = x.x) Julia version."""
    if julia_runtime is None:
        julia_runtime = shutil.which("julia")
    version = get_julia_version(julia_runtime)

    if (Path(pmpc_path) / "src" / "precompile.jl").exists():
        os.remove(Path(pmpc_path) / "src" / "precompile.jl")
    (Path(pmpc_path) / "src" / "precompile.jl").touch()
    # update the Manifest and Project files and store their version in `versions`
    julia_prog = f"""
using Pkg
Pkg.activate("{pmpc_path}")
Pkg.update()
Pkg.activate()
Pkg.develop(PackageSpec(path="{pmpc_path}"))
    """
    check_call([julia_runtime, "-e", julia_prog])
    shutil.copy(
        str(Path(pmpc_path) / "Manifest.toml"),
        str(Path(pmpc_path) / "versions" / f"Manifest.toml_{version}"),
    )
    shutil.copy(
        str(Path(pmpc_path) / "Project.toml"),
        str(Path(pmpc_path) / "versions" / f"Project.toml_{version}"),
    )
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
Pkg.activate("{pmpc_path}")
Pkg.add("PackageCompiler")
include("{compilation_utils_path}")
fix_tracefile("trace_{version}.jl")
    """
    check_call([julia_runtime, "-e", julia_prog])
    shutil.copy(
        Path(f"trace_{version}.jl"), Path(pmpc_path) / "src" / "traces" / f"trace_{version}.jl"
    )


####################################################################################################


def make_sysimage(julia_runtime=None):
    """Generate a sysimage for a particular Julia version."""
    if julia_runtime is None:
        julia_runtime = shutil.which("julia")
    assert julia_runtime is not None
    version = get_julia_version(julia_runtime)

    sysimage_path = Path("~").expanduser() / ".cache" / "pmpc" / f"pmpc_sysimage_{version}.so"
    sysimage_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = Path(pmpc_path).absolute() / "src" / "traces" / f"trace_{version}.jl"

    julia_prog = f"""
using Pkg
Pkg.add("PackageCompiler")
using PackageCompiler
include("{compilation_utils_path}")
Pkg.develop(PackageSpec(path="{pmpc_path}"))
Pkg.activate("{pmpc_path}")
Pkg.instantiate()
Pkg.resolve()
#Pkg.activate("{pmpc_path}")
tracefile2sysimage("{str(trace_path)}")
    """
    check_call([julia_runtime, "-e", julia_prog])
    os.rename("pmpc_sysimage.so", sysimage_path)


####################################################################################################

if __name__ == "__main__":
    juliaup_cmd = str(Path("~").expanduser() / ".juliaup" / "bin" / "juliaup")
    julia_runtime = str(Path("~").expanduser() / ".juliaup" / "bin" / "julia")
    versions = ["1.6", "1.7", "1.8", "1.9"]

    #for version in versions:
    #   check_call([juliaup_cmd, "default", version])
    #   generate_for_julia_version(julia_runtime)

    for version in versions:
        check_call([juliaup_cmd, "default", version])
        install_package_julia_version(julia_runtime)
        make_sysimage(julia_runtime)
