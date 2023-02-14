#!/usr/bin/env python3

import os, sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop


# custom julia installation script for the PMPC module #######
def install_julia_package():
    print("Installing the julia python support...")
    import julia

    try:
        from julia import Main as jl
        success = True
    except julia.core.UnsupportedPythonError as e:
        success = False
    if not success:
        try:
            julia.install()
            from julia import Main as jl
            success = True
        except julia.core.UnsupportedPythonError as e:
            success = False
    if not success:
        try:
            julia.Julia(compiled_modules=False)
            from julia import Main as jl
        except julia.core.UnsupportedPythonError as e:
            pass

    print("Installing the PMPC package...")
    try:
        install_PMPC = """
        using Pkg
        Pkg.develop(PackageSpec(path="%s"))
        """ % os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "PMPC.jl"
        )
        jl.eval(install_PMPC)
    except Exception as e:
        print(e)
        pass


# taken from https://stackoverflow.com/questions/20288711/post-install-script-with-python-setuptools
class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        develop.run(self)
        install_julia_package()


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        install.run(self)
        install_julia_package()


# perform setup ##############################################
setup(
    name="pmpc",
    version="0.5",
    packages=find_packages(),
    install_requires=["numpy", "julia", "zstandard", "pyzmq", "cloudpickle", "redis"],
    dependency_links=[],
    include_package_data=True,
    cmdclass=dict(develop=PostDevelopCommand, install=PostInstallCommand),
)
