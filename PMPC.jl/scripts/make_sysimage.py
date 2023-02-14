import sys, os, pdb

from julia import sysimage, Julia


if __name__ == "__main__":
    dirname = os.path.abspath(os.path.dirname(__file__))
    base_path = os.path.join(dirname, "sys_base.so")
    #path = os.path.join(dirname, "sys_pmpc.so")
    path = os.path.abspath(os.path.expanduser("~/.local/lib/sys_pmpc.so"))

    sysimage.main([base_path])
    Julia(sysimage=base_path)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    from julia import Main as jl

    jl.using("PackageCompiler")
    jl.using("PMPC")
    jl.eval('create_sysimage(:PMPC; sysimage_path="%s")' % path)

    os.remove(base_path)
