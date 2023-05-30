#!/usr/bin/env python3

import sys
import os
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop


sys.path.insert(0, str(Path(__file__).absolute().parent / "PMPC.jl" / "scripts"))
from tools import install_package_julia_version, get_julia_version, make_sysimage


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
        make_sysimage()
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


# perform setup ##############################################
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
    cmdclass=dict(develop=PostDevelopCommand, install=PostInstallCommand),
)
