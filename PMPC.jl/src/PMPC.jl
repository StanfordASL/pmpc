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

using LinearAlgebra, SparseArrays, Printf, Base.Threads, SuiteSparse, Suppressor
using OSQP
using REPL
using JuMP, MathOptInterface, ECOS
using MosekTools, COSMO
const MOI = MathOptInterface
using Infiltrator

include("types.jl")
include("qp.jl")
include("osqp.jl")
include("jump.jl")
include("lqp.jl")
include("cone.jl")
include("main.jl")
include("memory_utils.jl")

if isfile(joinpath(@__DIR__, "precompile.jl"))
  include("precompile.jl")
end

end # module
