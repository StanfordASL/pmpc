# install the libraries required for this build script #############################################
using Pkg, InteractiveUtils
Pkg.add("PackageCompiler")
using PackageCompiler

# remove existing package name resolution ##########################################################
PMPC_path = joinpath(@__DIR__, "..")
PMPC_pkg = Pkg.PackageSpec(path=PMPC_path)
manifest_path = joinpath(PMPC_path, "Manifest.toml") # handle any Julia version
try
  rm(manifest_path)
catch e
end

# fix and install the PMPC development dependencies ################################################
third_party_path = joinpath(@__DIR__, "..", "..", "third_party")
#println("Running fixes for Mosek_jll")
#include(joinpath(@__DIR__, "..", "..", "third_party", "Mosek_jll", "deps", "build.jl"))
#Pkg.develop(PackageSpec(path=joinpath(third_party_path, "Mosek_jll")))
#Pkg.develop(PackageSpec(path=joinpath(third_party_path, "Mosek.jl")))

# install the PMPC package #########################################################################
#Pkg.activate(joinpath(third_party_path, "Mosek.jl"))
#Pkg.develop(Pkg.PackageSpec(url="https://github.com/rdyro/Mosek_jll.jl"))
#Pkg.resolve()
Pkg.activate(PMPC_path)
#Pkg.develop(PackageSpec(path=joinpath(third_party_path, "Mosek_jll")))
Pkg.develop(Pkg.PackageSpec(path=joinpath(third_party_path, "Mosek.jl")))
Pkg.resolve()
Pkg.instantiate()
Pkg.precompile()
Pkg.activate()
Pkg.develop(PMPC_pkg)

# potentially updatedb to find extra dynamic libraries #############################################
try
  run(`updatedb`)
catch e
end

# potentially build the library ####################################################################
pmpc_lib_path = joinpath(@__DIR__, "..", "build", "pmpc_lib_$(VERSION)")
pmpcjl_path = joinpath(@__DIR__, "..", "pmpcjl")
if !isdir(pmpc_lib_path)
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
    catch e
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
