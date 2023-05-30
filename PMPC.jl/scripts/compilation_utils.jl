#!/usr/bin/env julia

using Pkg, PackageCompiler

function main(trace_file)
  # find modules used in the trace #############################################
  modules = Set{String}()
  open(trace_file, "r") do f # regex find all modules in the trace file
    lines = [line for line in readlines(f) if length(line) > 0]
    for line in lines
      m = match(r"[^.]([_A-Za-z]+)\.[a-zA-z]", line)
      captures = m == nothing ? String[] : m.captures
      union!(modules, captures)
    end
  end
  delete!(modules, "Main")
  delete!(modules, "Base")
  println("using $(join(modules, ", "))")

  #for mod in modules # install modules found in the trace
  #  try
  #    Pkg.add(mod)
  #  catch e
  #  end
  #end

  # write module using directives into the trace file ##########################
  filecontents = read(trace_file, String)
  open(trace_file, "w") do f
    if match(r"^using", filecontents) == nothing
      #write(f, "using $(join(modules, ", "))\n")
      for mod in modules
        write(f, "using $(mod)\n")
      end
    end
    write(f, filecontents)
  end

  return
end

#main("output.jl")

################################################################################

function comment_out_line(fname::String, line_nb::Int)
  lines = split(read(fname, String), '\n')
  open(fname, "w") do f
    for (i, line) in enumerate(lines)
      if i == line_nb
        write(f, "#$(line)\n")
      else
        write(f, "$(line)\n")
      end
    end
  end
  return
end

function fix_tracefile(trace_file)
  # find modules used in the trace #############################################
  modules = Set{String}()
  open(trace_file, "r") do f # regex find all modules in the trace file
    lines = [line for line in readlines(f) if length(line) > 0]
    for line in lines
      m = match(r"[^.]([_A-Za-z]+)\.[a-zA-z]", line)
      captures = m == nothing ? String[] : m.captures
      union!(modules, captures)
    end
  end
  delete!(modules, "Main")
  delete!(modules, "Base")
  println("using $(join(modules, ", "))")

  # write module using directives into the trace file ##########################
  filecontents = read(trace_file, String)
  open(trace_file, "w") do f
    if match(r"^using", filecontents) == nothing
      #write(f, "using $(join(modules, ", "))\n")
      for mod in modules
        write(f, "using $(mod)\n")
      end
    end
    write(f, filecontents)
  end

  # comment out all the lines that throw errors ################################
  while true
    try
      include(trace_file)
      break
    catch e
      println(e.error)
      if typeof(e.error) == ArgumentError
        pkg_name = match(r"Package ([_A-Za-z]+)", e.error.msg).captures[1]
        try
          Pkg.add(pkg_name)
        catch
          comment_out_line(trace_file, e.line)
        end
      elseif typeof(e.error) == UndefVarError
        comment_out_line(trace_file, e.line)
      elseif typeof(e.error) == UndefVarError
      else
        break
      end
    end
  end

  return
end

function tracefile2sysimage(trace_file)
  create_sysimage(["PMPC"]; sysimage_path="pmpc_sysimage.so", precompile_statements_file=trace_file)
  return
end
