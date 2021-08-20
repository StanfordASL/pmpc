import os, sys
from setuptools import setup, find_packages
from subprocess import Popen

# perform setup ##############################################
setup(
    name="pmpc",
    version="0.1",
    packages=find_packages(),
    install_requires=[],
    dependency_links=[],
    include_package_data=True,
)

# custom julia installation script for the PMPC module #######
if False and __name__ == "__main__" and len(sys.argv) >= 2 and sys.argv[1] == "install":
    import julia

    try:
        from julia import Main as jl
    except julia.core.UnsupportedPythonError as e:
        julia.install()
        from julia import Main as jl

    try:
        jl.using("PMPC")
        first_installation = False
    except julia.core.JuliaError as e:
        first_installation = True

    try:
        install_PMPC = """
        using Pkg
        Pkg.develop(PackageSpec(path="%s"))
        """ % os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "resources/PMPC.jl"
        )
        jl.eval(install_PMPC)
    except Exception as e:
        print(e)
        pass

    if first_installation:
        path = os.path.join(
            os.path.dirname(__file__), "resources/ecos_repair/update_libecos.py"
        )
        p = Popen(["python3", path])
        p.wait()
