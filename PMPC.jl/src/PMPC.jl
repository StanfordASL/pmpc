module PMPC

#using Revise

export PMPC
export OCProb, OSQPSolver
export set_dyn!, set_cost!, set_xbounds!, set_ubounds!
export convert!, catb
export solve!, solve
export prox_setup!, prox_reset!, prox!
export split_fbvars

export lqp_solve, lcone_solve, lqp_generate_problem_matrices
export rollout, dynamics_violation, make_prob, make_probs
export PMPCs_ctrl, MPCs_ctrl

using LinearAlgebra, SparseArrays, Printf, Base.Threads, Statistics
using OSQP
using REPL
using JuMP, MathOptInterface, ECOS, COSMO
const MOI = MathOptInterface
using PrecompileTools
#using Gurobi # not currently supported
using MosekTools 

include("types.jl")

include("qp_utils.jl")
include("lqp_utils.jl")
include("osqp_solver.jl")

include("cone_utils.jl")
include("cone_solver.jl")

#include("jump_solver.jl")
#include("memory_utils.jl")

include("main.jl")

include("c_interface.jl")
@compile_workload begin
  include("c_precompile.jl")
end

export c_lqp_solve, c_lcone_solve

end # module
