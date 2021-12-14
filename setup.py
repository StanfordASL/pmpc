import os, sys
from setuptools import setup, find_packages
from subprocess import Popen
from shutil import rmtree

# perform setup ##############################################
setup(
    name="pmpc",
    version="0.5",
    packages=find_packages(),
    install_requires=[],
    dependency_links=[],
    include_package_data=True,
)

# custom rules below # #########################################################
action = None if len(sys.argv) < 2 else sys.argv[1]

# custom julia installation script for the PMPC module #######
if __name__ == "__main__" and action == "install":
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
            os.path.abspath(os.path.dirname(__file__)), "PMPC.jl"
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

if __name__ == "__main__" and action == "clean":
    dirname = os.path.abspath(os.path.dirname(__file__))
    flist = ["build", "dist", "pmpc.egg-info"]
    for fname in flist:
        if os.path.isdir(fname):
            rmtree(os.path.join(dirname, fname))
