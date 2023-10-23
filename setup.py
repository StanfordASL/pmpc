#!/usr/bin/env python3

import os
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from subprocess import check_call

import pybind11
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install

####################################################################################################
####################################################################################################
####################################################################################################


def _install_common(kw):
    setup(
        name="pmpc",
        version="0.6.0",
        packages=find_packages(),
        install_requires=[
            "numpy",
            "julia",
            "zstandard",
            "pyzmq",
            "cloudpickle",
            "redis",
            "tqdm",
            "psutil",
        ],
        dependency_links=[],
        include_package_data=True,
        **kw,
    )


####################################################################################################
####################################################################################################
####################################################################################################


def install_dynamic():
    sys.path.insert(0, str(Path(__file__).absolute().parent / "PMPC.jl" / "scripts"))
    from tools import get_julia_version, install_package_julia_version, make_sysimage

    # custom julia installation script for the PMPC module #######
    def install_julia_package():
        print("Installing the julia python support...")
        import julia

        jl = None

        try:
            from julia import Main as jl

            success = True
        except julia.core.UnsupportedPythonError:
            success = False
        if not success:
            try:
                julia.install()
                from julia import Main as jl  # noqa: F811

                success = True
            except julia.core.UnsupportedPythonError:
                success = False
        if not success:
            try:
                julia.Julia(compiled_modules=False)
                from julia import Main as jl  # noqa: F811
            except julia.core.UnsupportedPythonError:
                pass

        print("Installing the PMPC package...")
        try:
            print(f"Julia version = {get_julia_version()}")
            install_package_julia_version()
            # make_sysimage()
        except Exception as e:
            print(e)
            pass

    # taken from https://stackoverflow.com/questions/20288711/post-install-script-with-python-setuptools
    class PostDevelopCommand(develop):
        """Post-installation for development mode."""

        def run(self):
            install_julia_package()
            develop.run(self)

    class PostInstallCommand(install):
        """Post-installation for installation mode."""

        def run(self):
            install_julia_package()
            install.run(self)

    _install_common(
        dict(
            cmdclass=dict(develop=PostDevelopCommand, install=PostInstallCommand),
        )
    )


####################################################################################################
####################################################################################################
####################################################################################################


def install_static():
    lib_paths = sum(
        [
            [str(Path(root) / f) for f in fnames]
            for (root, _, fnames) in os.walk(
                str(Path(__file__).parent / "PMPC.jl" / "pmpcjl" / "lib")
            )
        ],
        [],
    )
    share_paths = sum(
        [
            [str(Path(root) / f) for f in fnames]
            for (root, _, fnames) in os.walk(
                str(Path(__file__).parent / "PMPC.jl" / "pmpcjl" / "share")
            )
        ],
        [],
    )
    pmpcjl_ext = Extension(
        "pmpcjl",
        sources=[str(Path("PMPC.jl") / "pmpcjl" / "module.cpp")],
        include_dirs=[
            pybind11.get_include(),
            str(Path("PMPC.jl") / "pmpcjl" / "include"),
        ],
        library_dirs=[str(Path("PMPC.jl") / "pmpcjl" / "lib")],
        libraries=["julia", "PMPC"],
        language="c++",
    )

    class BuildPMPCjl(build_ext):
        def run(self):
            check_call(["julia", "PMPC.jl/scripts/build_pmpc_lib.jl"])
            super().run()

    _install_common(
        dict(
            cmdclass={"build_ext": BuildPMPCjl},
            ext_modules=[pmpcjl_ext],
            package_dir={"pmpcjl": "PMPC.jl/pmpcjl"},
            package_data={"": lib_paths + share_paths},
        )
    )


if __name__ == "__main__":
    # try:
    #    install_dynamic()
    #    dynamic_success = True
    # except Exception as e:
    #    traceback.print_exc()
    #    dynamic_success = False
    dynamic_success = False
    try:
       install_static()
       static_success = True
    except Exception as e:
       traceback.print_exc()
       static_success = False
    assert dynamic_success or static_success, "Neither dynamic nor static install succeeded"
