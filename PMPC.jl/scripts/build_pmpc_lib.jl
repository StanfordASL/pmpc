# install the libraries required for this build script #############################################
using Pkg, InteractiveUtils
try
  using PackageCompiler
catch
  Pkg.add("PackageCompiler")
end
using PackageCompiler

# potentially install the library ##################################################################
PMPC_path = joinpath(@__DIR__, "..")
PMPC_pkg = Pkg.PackageSpec(path=PMPC_path)
pmpc_lib_path = joinpath(@__DIR__, "..", "build", "pmpc_lib_$(VERSION)")
pmpcjl_path = joinpath(@__DIR__, "..", "pmpcjl")

if !isdir(pmpc_lib_path)
  # install the PMPC package #########################################################################
  cp(joinpath(PMPC_path, "Manifest_1.6.7.toml"), joinpath(PMPC_path, "Manifest.toml"); force=true)
  if parse(Bool, get(ENV, "PMPC_STATIC_INSTALL", "false")) # we're building for a potentially different Julia version
    rm(joinpath(PMPC_path, "Manifest.toml"), force=true)
    Pkg.activate(PMPC_path)
    Pkg.develop(Pkg.PackageSpec(path=joinpath(PMPC_path, "..", "third_party", "MosekTools.jl")))
    Pkg.resolve()
    Pkg.instantiate()
    Pkg.activate()
  end
  Pkg.develop(PMPC_pkg)
  Pkg.precompile()

  # potentially updatedb to find extra dynamic libraries #############################################
  try
    run(`updatedb`)
  catch
  end

  if !isdir(joinpath(PMPC_path, "build"))
    mkdir(joinpath(PMPC_path, "build"))
  end
  println(repeat("#", 80))
  println("We need to build the PMPC library, this will take about 10 mins!")
  println(repeat("#", 80))
  create_library(
    joinpath(@__DIR__, ".."),
    pmpc_lib_path,
    force=true,
    precompile_execution_file=joinpath(@__DIR__, "..", "src", "c_precompile.jl"),
    #filter_stdlibs=true,
    incremental=false,
  )
end

# vendor in extra libraries ########################################################################
extra_libs = ["libsuitesparse_wrapper"]
for extra_lib in extra_libs
  locations = [x for x in split(read(`locate $extra_lib`, String), "\n") if length(x) > 0]
  locations = [x for x in locations if match(Regex("$extra_lib\\..*"), x) != nothing]
  if length(locations) == 0
    error("Could not find $extra_lib")
  end
  for location in locations
    println("Copying $location")
    try
      cp(location, joinpath(pmpc_lib_path, "lib", "julia", splitpath(location)[end]), force=true)
    catch
    end
  end
end

# copy the library to the PMPC.jl package ##########################################################
libfiles = [f for f in readdir(joinpath(pmpc_lib_path, "lib"), join=true)]
include_path = joinpath(pmpc_lib_path, "include")
lib_path = joinpath(pmpc_lib_path, "lib")
share_path = joinpath(pmpc_lib_path, "share")
cp(include_path, joinpath(pmpcjl_path, "include"), force=true)
cp(lib_path, joinpath(pmpcjl_path, "lib"), force=true)
cp(share_path, joinpath(pmpcjl_path, "share"), force=true)
