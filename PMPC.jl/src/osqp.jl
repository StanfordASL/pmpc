# OSQP solver ##################################################################
mutable struct OSQPSolver{T}
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
  model::Union{OSQP.Model, Nothing}
  P_::Union{SpMat{T}, Nothing}
  q_::Union{AA{T, 1}, Nothing}
end
OSQPSolver() = OSQPSolver{Float64}(repeat([nothing], 13)...)
function augmented_A(solver::OSQPSolver{T}) where {T}
  Aa = spzeros(T, 0, solver.n)
  la, ua = zeros(T, 0), zeros(T, 0)
  if solver.A != nothing && solver.b != nothing
    Aa = vcat(Aa, solver.A)
    la, ua = vcat(la, solver.b), vcat(ua, solver.b)
  end
  if solver.G != nothing && solver.l != nothing && solver.u != nothing
    Aa = vcat(Aa, solver.G)
    la, ua = vcat(la, solver.l), vcat(ua, solver.u)
  end
  return Aa, la, ua
end
function setup!(solver::OSQPSolver{T}; solver_settings...) where {T}
  solver_settings = Dict{Symbol, Any}(solver_settings)
  get!(solver_settings, :verbose, false)
  success = true
  if solver.model == nothing
    solver.model = OSQP.Model()
    @assert solver.P != nothing && solver.q != nothing
    solver.n = solver.n != nothing ? solver.n : size(solver.P, 2)
    Aa, la, ua = augmented_A(solver)
    solver.m = size(Aa, 1)
    success = true
    try
      if size(Aa, 1) > 0
        OSQP.setup!(solver.model; P=solver.P, q=solver.q, A=Aa, l=la, u=ua, solver_settings...)
      else
        OSQP.setup!(solver.model; P=solver.P, q=solver.q, solver_settings...)
      end
      success = true
    catch e
      success = false
    end
  end
  return success
end
function OSQP_update_settings!(solver::OSQPSolver{T}; settings...) where {T}
  (solver.model != nothing) && (OSQP.update_settings!(solver.model; settings...))
  return
end
function solve_qp!(solver::OSQPSolver{T}; settings...) where {T}
  if setup!(solver; settings...)
    ret = OSQP.solve!(solver.model)
    return ret.x, ret.y
  else
    return fill(T(NaN), solver.n), fill(T(NaN), solver.m)
  end
end
function solve!(solver::OSQPSolver{T}; settings...) where {T}
  z, y = solve_qp!(solver; settings...)
  xdim, udim, N = solver.prob.xdim, solver.prob.udim, solver.prob.N
  @views X = reshape(z[(N * udim + 1):end], xdim, N)
  @views U = reshape(z[1:(N * udim)], udim, N)
  obj = get(settings, :ret_obj, false) ? objective(solver.prob, X, U) : T(NaN)
  return X, U, y, obj
end
function set_P!(solver::OSQPSolver{T}, P::SpMat{T}) where {T}
  solver.P == P && return
  if (solver.P == nothing || solver.P.colptr != P.colptr || solver.P.rowval != P.rowval)
    solver.model = nothing
  end
  solver.P = P
  solver.n = size(P, 1)
  if solver.model != nothing
    OSQP.update!(solver.model; Px=solver.Px.nzval)
  end
end
function set_q!(solver::OSQPSolver{T}, q::AA{T, 1}) where {T}
  solver.q == q && return
  solver.q = q
  solver.n = size(q, 1)
  if solver.model != nothing
    OSQP.update!(solver.model; q=solver.q)
  end
end
function set_A!(solver::OSQPSolver{T}, A::SpMat{T}) where {T}
  solver.A == A && return
  if (solver.A == nothing || solver.A.colptr != A.colptr || solver.A.rowval != A.rowval)
    solver.model = nothing
  end
  solver.A = A
  if solver.model != nothing
    Aa, la, ua = augmented_A(solver)
    solver.m = size(Aa, 1)
    if size(Aa, 1) > 0
      OSQP.update!(solver.model; Ax=Aa.nnz)
    end
  end
end
function set_b!(solver::OSQPSolver{T}, l::SpMat{T}) where {T}
  solver.b == b && return
  solver.b = b
  if solver.model != nothing
    Aa, la, ua = augmented_A(solver)
    solver.m = size(Aa, 1)
    if size(Aa, 1) > 0
      OSQP.update!(solver.model; l=la, u=ua)
    end
  end
end
function set_G!(solver::OSQPSolver{T}, G::SpMat{T}) where {T}
  solver.G == G && return
  if (solver.G == nothing || solver.G.colptr != G.colptr || solver.G.rowval != G.rowval)
    solver.model = nothing
  end
  solver.G = G
  if solver.model != nothing
    Aa, la, ua = augmented_A(solver)
    solver.m = size(Aa, 1)
    if size(Aa, 1) > 0
      OSQP.update!(solver.model; Ax=Aa.nnz)
    end
  end
end
function set_l!(solver::OSQPSolver{T}, l::SpMat{T}) where {T}
  solver.l == l && return
  solver.l = l
  if solver.model != nothing
    Aa, la, ua = augmented_A(solver)
    solver.m = size(Aa, 1)
    if size(Aa, 1) > 0
      OSQP.update!(solver.model; l=la)
    end
  end
end
function set_u!(solver::OSQPSolver{T}, u::SpMat{T}) where {T}
  solver.u == u && return
  solver.u = u
  if solver.model != nothing
    Aa, la, ua = augmented_A(solver)
    solver.m = size(Aa, 1)
    if size(Aa, 1) > 0
      OSQP.update!(solver.model; l=la)
    end
  end
end
# OSQP solver ##################################################################


# proximal operator support ####################################################
function prox_setup!(solver::OSQPSolver{T}, mask::AA{T, 1}; settings...) where {T}
  @assert solver.n != nothing && solver.P != nothing && solver.q != nothing
  @assert length(mask) == solver.n

  solver.P_, solver.q_ = copy(solver.P), copy(solver.q)
  set_P!(solver, solver.P .+= spdiagm(0 => mask))
end
function prox_reset!(solver::OSQPSolver{T}; settings...) where {T}
  @assert solver.n != nothing && solver.P != nothing && solver.q != nothing
  @assert length(mask) == solver.n
  if solver.P_ == nothing || solver.q_ == nothing
    return
  end
  set_P!(solver, solver.P_)
  set_q!(solver, solver.q_)
  solver.P_, solver.q_ = nothing, nothing
end
function prox!(solver::OSQPSolver{T}, bias::AA{T, 1}; settings...) where {T}
  @assert solver.n != nothing && solver.P != nothing && solver.q != nothing
  @assert length(bias) == solver.n
  @assert solver.P_ != nothing && solver.q_ != nothing
  set_q!(solver, solver.q_ + bias)
  return solve!(solver; settings...)
end
function prox_qp!(solver::OSQPSolver{T}, bias::AA{T, 1}; settings...) where {T}
  @assert solver.n != nothing && solver.P != nothing && solver.q != nothing
  @assert length(bias) == solver.n
  @assert solver.P_ != nothing && solver.q_ != nothing
  set_q!(solver, solver.q_ + bias)
  return solve_qp!(solver; settings...)
end
# proximal operator support ####################################################


# conversions ##################################################################
import Base.convert
function convert(::Type{OSQPSolver}, prob::OCProb{T}) where {T}
  P, q, _ = qp_repr_Pq(prob)
  A, b = qp_repr_Ab(prob)
  G, l, u = qp_repr_Glu(prob)
  return OSQPSolver{T}(
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
  )
end
OSQPSolver(prob::OCProb{T}) where {T} = convert(OSQPSolver, prob)
################################################################################
