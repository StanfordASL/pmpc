# constants and types ##############################################################################
const NUM_INF = 1e20

mutable struct JuMPSolution{T}
  z::Vector{T}
  lam_A::Union{Vector{T}, Nothing}
  nu_G::Union{Vector{T}, Nothing}
end
mutable struct JuMPSolver{T}
  prob::Union{OCProb{T}, Nothing}
  P::Union{SpMat{T}, Nothing}
  q::Union{AA{T, 1}, Nothing}
  A::Union{SpMat{T}, Nothing}
  b::Union{AA{T, 1}, Nothing}
  G::Union{SpMat{T}, Nothing}
  l::Union{AA{T, 1}, Nothing}
  u::Union{AA{T, 1}, Nothing}
  n::Union{Int, Nothing}
  m::Union{Int, Nothing}
  p::Union{Int, Nothing}
  model::Union{JuMP.Model, Nothing}
  P_::Union{SpMat{T}, Nothing}
  q_::Union{AA{T, 1}, Nothing}
  z_var::Union{Vector{JuMP.VariableRef}, Nothing}
  z_sol::Union{Vector{T}, Nothing}
  tu_sol::Union{Vector{T}, Nothing}
  tl_sol::Union{Vector{T}, Nothing}
  tl::Union{Vector{JuMP.VariableRef}, Nothing}
  tu::Union{Vector{JuMP.VariableRef}, Nothing}
  var_solution::Dict{VariableRef, T}
  cstr_solution::Dict{ConstraintRef, Tuple{T, T}}
end
JuMPSolver() = JuMPSolver{Float64}(
  repeat([nothing], 20)...,
  Dict{VariableRef, Float64}(),
  Dict{ConstraintRef, Tuple{Float64, Float64}}(),
)


# main setup function with logic handling ##########################################################
function setup!(solver::JuMPSolver{T}; solver_settings...)::Bool where {T}
  # set the default options
  solver_settings = Dict{Symbol, Any}(solver_settings)
  get!(solver_settings, :verbose, false)
  get!(solver_settings, :smooth_cstr, "")
  get!(solver_settings, :smooth_alpha, 1e0)
  #get!(solver_settings, :solver_name, "Mosek")
  get!(solver_settings, :solver_name, "ECOS")

  solver.z_sol = get(solver_settings, :z_sol, nothing)
  solver.tu_sol = get(solver_settings, :tu_sol, nothing)
  solver.tl_sol = get(solver_settings, :tl_sol, nothing)

  # select the solver
  if lowercase(solver_settings[:solver_name]) == "mosek"
    @assert false, "Mosek not supported"
    #solver.model =
    #  JuMP.Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => !solver_settings[:verbose]))
  elseif lowercase(solver_settings[:solver_name]) == "ecos"
    solver.model = JuMP.Model(ECOS.Optimizer)
    set_optimizer_attribute(solver.model, "verbose", solver_settings[:verbose])
  elseif lowercase(solver_settings[:solver_name]) == "osqp"
    solver.model =
      JuMP.Model(optimizer_with_attributes(OSQP.Optimizer, "verbose" => !solver_settings[:verbose]))
    set_optimizer_attribute(solver.model, "verbose", solver_settings[:verbose])
  else
    error("Solver $(solver_settings[:solver_name]) not supported")
  end

  # set solver dimensions and main solution variable z = (U, X)
  @assert solver.P != nothing && solver.q != nothing
  solver.n = solver.n != nothing ? solver.n : size(solver.P, 2)
  solver.m = solver.A != nothing ? size(solver.A, 1) : 0
  z = @variable(solver.model, z[1:(solver.n)])
  solver.z_var = z

  # setup equality constraints
  if solver.A != nothing && solver.b != nothing && size(solver.A, 1) > 0
    @constraint(solver.model, solver.A * z .== solver.b)
  end

  # setup inequality constraints and objective
  @assert solver_settings[:smooth_cstr] in ("", "logbarrier")
  if solver_settings[:smooth_cstr] == ""
    setup_linear_cstr_and_objective!(z, solver; solver_settings...)
  elseif solver_settings[:smooth_cstr] == "logbarrier"
    setup_logbarrier_cstr_and_objective!(z, solver; solver_settings...)
  end
  return true
end


# constraints setup ################################################################################
function setup_linear_cstr_and_objective!(
  z::Vector{VariableRef},
  solver::JuMPSolver{T};
  solver_settings...,
) where {T}
  # set constraints
  if solver.G != nothing && solver.u != nothing && solver.l != nothing && size(solver.G, 1) > 0
    (!all(solver.u .>= NUM_INF)) && (@constraint(solver.model, solver.G * z .<= solver.u))
    (!all(solver.l .<= -NUM_INF)) && (@constraint(solver.model, solver.G * z .>= solver.l))
  end
  #@info("Using linear constraints")
  # set the objective
  @objective(solver.model, Min, 0.5 * dot(z, solver.P, z) + dot(solver.q, z))
end

function setup_logbarrier_cstr_and_objective!(
  z::Vector{VariableRef},
  solver::JuMPSolver{T};
  solver_settings...,
) where {T}
  # use the linear case in the absence of constraints
  if !(solver.G != nothing && solver.u != nothing && solver.l != nothing && size(solver.G, 1) > 0)
    setup_linear_cstr_and_objective!(z, solver; solver_settings...)
    return
  end
  #@info("Using logbarrier constraints")

  # set constraints
  tu, tl = nothing, nothing
  alf = solver_settings[:smooth_alpha]
  solver.p = size(solver.G, 1)
  # upper constraint exists
  if !all(solver.u .>= NUM_INF)
    fu = @expression(solver.model, solver.G * z - solver.u)
    tu = @variable(solver.model, tu[1:(solver.p)])
    @constraint(
      solver.model,
      cu[i=1:(solver.p)],
      [-alf * tu[i], 1, -alf * fu[i]] in MOI.ExponentialCone()
    )
  end
  # lower constraint exists
  if !all(solver.l .<= -NUM_INF)
    fl = @expression(solver.model, solver.l - solver.G * z)
    tl = @variable(solver.model, tl[1:(solver.p)])
    @constraint(
      solver.model,
      cl[i=1:(solver.p)],
      [-alf * tl[i], 1, -alf * fl[i]] in MOI.ExponentialCone()
    )
  end

  # set the objective
  if tu != nothing && tl != nothing
    @objective(solver.model, Min, 0.5 * dot(z, solver.P, z) + dot(solver.q, z) + sum(tu) + sum(tl))
  elseif tu != nothing
    @objective(solver.model, Min, 0.5 * dot(z, solver.P, z) + dot(solver.q, z) + sum(tu))
  elseif tl != nothing
    @objective(solver.model, Min, 0.5 * dot(z, solver.P, z) + dot(solver.q, z) + sum(tl))
  end
end

# methods operating on the type (i.e. class methods) ###############################################
function solve_qp!(solver::JuMPSolver{T}; settings...) where {T}
  if setup!(solver; settings...)
    if haskey(settings, :solver_id)
      previous_solver = restore_solver(settings[:solver_id])
      solver.var_solution = previous_solver.var_solution
      solver.cstr_solution = previous_solver.cstr_solution
      restore_warmstart!(solver)
    end
    optimize!(solver.model)
    store_warmstart!(solver)
    if haskey(settings, :solver_id)
      store_solver!(solver, settings[:solver_id])
    end
    z = JuMP.value.(solver.z_var)
    solver.z_sol = z
    return z, fill(T(NaN), solver.m)
  else
    return fill(T(NaN), solver.n), fill(T(NaN), solver.m)
  end
end
function solve!(solver::JuMPSolver{T}; settings...) where {T}
  z, y = solve_qp!(solver; settings...)
  xdim, udim, N = solver.prob.xdim, solver.prob.udim, solver.prob.N
  @views X = reshape(z[(N * udim + 1):end], xdim, N)
  @views U = reshape(z[1:(N * udim)], udim, N)
  obj = get(settings, :ret_obj, false) ? objective(solver.prob, X, U) : T(NaN)
  return X, U, y, obj
end


# proximal operator support ####################################################
function prox_setup!(solver::JuMPSolver{T}, mask::AA{T, 1}; settings...) where {T}
  @assert solver.n != nothing && solver.P != nothing && solver.q != nothing
  @assert length(mask) == solver.n

  solver.P_, solver.q_ = copy(solver.P), copy(solver.q)
  set_P!(solver, solver.P .+= spdiagm(0 => mask))
end
function prox_reset!(solver::JuMPSolver{T}; settings...) where {T}
  @assert solver.n != nothing && solver.P != nothing && solver.q != nothing
  @assert length(mask) == solver.n
  if solver.P_ == nothing || solver.q_ == nothing
    return
  end
  set_P!(solver, solver.P_)
  set_q!(solver, solver.q_)
  solver.P_, solver.q_ = nothing, nothing
end
function prox!(solver::JuMPSolver{T}, bias::AA{T, 1}; settings...) where {T}
  @assert solver.n != nothing && solver.P != nothing && solver.q != nothing
  @assert length(bias) == solver.n
  @assert solver.P_ != nothing && solver.q_ != nothing
  set_q!(solver, solver.q_ + bias)
  return solve!(solver; settings...)
end
function prox_qp!(solver::JuMPSolver{T}, bias::AA{T, 1}; settings...) where {T}
  @assert solver.n != nothing && solver.P != nothing && solver.q != nothing
  @assert length(bias) == solver.n
  @assert solver.P_ != nothing && solver.q_ != nothing
  set_q!(solver, solver.q_ + bias)
  return solve_qp!(solver; settings...)
end

# conversions ######################################################################################
import Base.convert
function convert(::Type{JuMPSolver}, prob::OCProb{T}) where {T}
  P, q, _ = qp_repr_Pq(prob)
  A, b = qp_repr_Ab(prob)
  G, l, u = qp_repr_Glu(prob)
  return JuMPSolver{T}(
    prob,
    P,
    q,
    A,
    b,
    G,
    l,
    u,
    size(P, 1),
    size(A, 1) + size(G, 1),
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
    nothing,
  )
end
JuMPSolver(prob::OCProb{T}) where {T} = convert(JuMPSolver, prob)

# warmstart utility ################################################################################

"""Copied from https://jump.dev/JuMP.jl/stable/tutorials/conic/start_values/ """
function store_warmstart!(solver::JuMPSolver{T})::Nothing where {T}
  solver.var_solution = Dict(x => value(x) for x in all_variables(solver.model))
  for (F, S) in list_of_constraint_types(solver.model)
    try
      for ci in all_constraints(solver.model, F, S)
        solver.cstr_solution[ci] = (value(ci), dual(ci))
      end
    catch
      #@warn("Something went wrong getting $F-in-$S. Skipping", maxlog = 1)
    end
  end
  return
end

"""Copied from https://jump.dev/JuMP.jl/stable/tutorials/conic/start_values/ """
function restore_warmstart!(solver::JuMPSolver{T})::Nothing where {T}
  # we can loop through our cached solutions and set the starting values.
  for (x, primal_start) in solver.var_solution
    set_start_value(x, primal_start)
  end
  for (ci, (primal_start, dual_start)) in solver.cstr_solution
    set_start_value(ci, primal_start)
    set_dual_start_value(ci, dual_start)
  end
  return
end
