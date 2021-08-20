import os, shutil, subprocess, sys, re, pdb
from subprocess import Popen

###############################################################
# taken from https://www.pythoncentral.io/hashing-files-with-python/
# author: Andres Torres

import hashlib

BLOCKSIZE = 65536


def hash_file(path):
    hasher = hashlib.sha1()
    with open(path, "rb") as fp:
        buf = fp.read(BLOCKSIZE)
        while len(buf) > 0:
            hasher.update(buf)
            buf = fp.read(BLOCKSIZE)
    return hasher.hexdigest()


###############################################################


def remove_existing_ecos_git():
    if os.path.isdir("ecos"):
        print("Removing existing ecos directory...")
        shutil.rmtree("ecos")

    if os.path.isfile("ecos"):
        print("Removing existing ecos file...")
        os.remove("ecos")


if __name__ == "__main__":
    try:
        ecos = (
            subprocess.check_output(
                ["julia", "-e", "using ECOS; print(ECOS.ecos)"]
            )
            .decode("utf-8")
            .strip("\n")
        )
        if re.match(r".*\.julia/packages/ECOS.*", ecos) is None:
            bad_ecos = True
        else:
            bad_ecos = False
    except subprocess.CalledProcessError:
        bad_ecos = True

    if bad_ecos:
        print("Bad ECOS, fixing...")
        p = Popen(
            [
                "julia",
                "-e",
                """
        using Pkg
        Pkg.rm(PackageSpec(name="ECOS"))
        Pkg.add(PackageSpec(name="ECOS", version="0.11.0"))
        """,
            ]
        )
        p.wait()
        pdb.set_trace()
    else:
        print("Good ECOS")

    remove_existing_ecos_git()
    p = Popen(["git", "clone", "https://github.com/embotech/ecos.git"])
    p.wait()
    p = Popen(["make"], cwd="ecos")
    p.wait()
    p = Popen(["make", "shared"], cwd="ecos")
    p.wait()

    if sys.platform == "linux":
        LIB_EXT = "so"
    elif sys.platform == "darwin":
        LIB_EXT = "dylib"
    LIB_NAME = "libecos." + LIB_EXT

    try:
        root = os.path.join(os.environ["HOME"], ".julia")
        path_list = (
            subprocess.check_output(
                [
                    "find",
                    root,
                    "-regex",
                    os.path.join(root, "packages/.*/" + LIB_NAME + "$"),
                ]
            )
            .decode("utf-8")
            .strip("\n")
            .split("\n")
        )
    except subprocess.CalledProcessError as e:
        print("Looks like ECOS is not installed, aborting.")
        sys.exit()

    print()
    print(path_list)
    print()

    for path in path_list:
        if not path:
            continue
        sha1_before = hash_file(path)
        os.remove(path)
        shutil.copyfile("ecos/" + LIB_NAME, path)
        sha1_after = hash_file(path)
        sha1_correct = hash_file("ecos/" + LIB_NAME)

        print("Replacing:", path)
        print("SHA1 correct =", sha1_correct)
        print("SHA1 before  =", sha1_before)
        print("SHA1 after   =", sha1_after)
        print()

    p = Popen(
        [
            "julia",
            "-e",
            """
        using ECOS, SHA
        fp = open(ECOS.ecos, "r")
        sha_code = bytes2hex(sha1(fp))
        close(fp)
        @assert sha_code == "%s"
        println("Julia confirms that the lib SHA is correct")
        """
            % hash_file("ecos/" + LIB_NAME).lower(),
        ]
    )
    p.wait()

    remove_existing_ecos_git()
