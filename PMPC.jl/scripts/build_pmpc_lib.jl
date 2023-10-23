using Pkg, InteractiveUtils
try
  using PackageCompiler
catch
  Pkg.add("PackageCompiler")
end
using PackageCompiler

# install and resolve the PMPC package to the current Julia version ############
PMPC_path = joinpath(@__DIR__, "..")
PMPC_pkg = Pkg.PackageSpec(path=PMPC_path)
manifest_path = joinpath(PMPC_path, "Manifest.toml") # handle any Julia version
try
  rm(manifest_path)
catch e
end
Pkg.activate(PMPC_path)
Pkg.resolve()
Pkg.activate()
Pkg.develop(PMPC_pkg)


# potentially build the library ################################################
pmpc_lib_path = joinpath(@__DIR__, "..", "build", "pmpc_lib_$(VERSION)")
pmpcjl_path = joinpath(@__DIR__, "..", "pmpcjl")


if !isdir(pmpc_lib_path)
  if !isdir(joinpath(PMPC_path, "build"))
    mkdir(joinpath(PMPC_path, "build"))
  end
  println(repeat("#", 80))
  println(repeat("#", 80))
  println(repeat("#", 80))
  println("We need to build the PMPC library, this will take about 10 mins!")
  println(repeat("#", 80))
  println(repeat("#", 80))
  println(repeat("#", 80))
  create_library(
    joinpath(@__DIR__, ".."),
    pmpc_lib_path,
    force=true,
    precompile_execution_file=joinpath(@__DIR__, "..", "src", "c_precompile.jl"),
    #filter_stdlibs=true,
    incremental=false,
  )
  for extra_lib in ["libsuitesparse_wrapper"]
    locations = [x for x in split(read(`locate $extra_lib`, String), "\n") if length(x) > 0]
    if length(locations) == 0
      error("Could not find $extra_lib")
    end
    cp(locations[1], joinpath(pmpc_lib_path, "lib", "julia", splitpath(locations[1])[end]), force=true)
  end
end

libfiles = [f for f in readdir(joinpath(pmpc_lib_path, "lib"), join=true)]
include_path = joinpath(pmpc_lib_path, "include")
lib_path = joinpath(pmpc_lib_path, "lib")
share_path = joinpath(pmpc_lib_path, "share")
cp(include_path, joinpath(pmpcjl_path, "include"), force=true)
cp(lib_path, joinpath(pmpcjl_path, "lib"), force=true)
cp(share_path, joinpath(pmpcjl_path, "share"), force=true)