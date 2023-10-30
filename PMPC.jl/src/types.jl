const AA = AbstractArray
const SpMat = SparseMatrixCSC

# utils ########################################################################
function catb(x_list...)
  x_list_flat = view.(x_list, :)
  X = hcat(x_list_flat...)
  ret = reshape(X, (size(x_list[1])..., length(x_list)))
  return ret
end

get_types(::Type{Dict{TK, TV}}) where {TK, TV} = (TK, TV)

import Base.convert

function Base.convert(::Type{Dict{Symbol, Any}}, d::Dict)
  if get_types(typeof(d))[1] <: Symbol
    return d
  else
    return Dict{Symbol, Any}(Symbol(p.first) => p.second for p in d)
  end
end

macro ptime(f, n=1, prepend_text="")
  return quote
    for i_ in 1:($n)
      @suppress $(esc(f))
    end
    print($prepend_text)
    @time local val = $(esc(f))
    val
  end
end

function efficient_vcat(mats::AA{SparseMatrixCSC{T1, T2}, 1}) where {T1, T2}
  n = length(mats)
  @assert length(mats) >= 0
  N, cols = length(mats), size(mats[1], 2)
  @assert all(map(i -> size(mats[i], 2) == cols, 2:n))
  elnum = sum(nnz.(mats))
  Xp = zeros(T2, cols + 1)
  Xp, Xi, Xx = zeros(T2, cols + 1), zeros(T2, elnum), zeros(T1, elnum)
  rows = size.(mats, 1)
  cumrows = [0; cumsum(rows)]
  k = 0
  eidxs = ones(Int, N)
  Xp[1] = 1
  for c in 1:cols
    k_old = k
    for i in 1:N
      ei = eidxs[i]
      while ei < mats[i].colptr[c + 1]
        k += 1
        Xi[k] = mats[i].rowval[ei] + cumrows[i]
        Xx[k] = mats[i].nzval[ei]
        ei += 1
      end
      eidxs[i] = ei
    end
    Xp[c + 1] = Xp[c] + (k - k_old)
  end
  X = SparseMatrixCSC{T1, T2}(cumrows[N + 1], cols, Xp, Xi, Xx)
  return X
end
# utils ########################################################################


# main class ###################################################################
mutable struct OCProb{T}
  x0::Union{AA{T, 1}, Nothing}
  f::Union{AA{T, 2}, Nothing}
  fx::Union{AA{T, 3}, Nothing}
  fu::Union{AA{T, 3}, Nothing}
  X_prev::Union{AA{T, 2}, Nothing}
  U_prev::Union{AA{T, 2}, Nothing}
  Q::Union{AA{T, 3}, Nothing}
  R::Union{AA{T, 3}, Nothing}
  X_ref::Union{AA{T, 2}, Nothing}
  U_ref::Union{AA{T, 2}, Nothing}
  lx::Union{AA{T, 2}, Nothing}
  ux::Union{AA{T, 2}, Nothing}
  lu::Union{AA{T, 2}, Nothing}
  uu::Union{AA{T, 2}, Nothing}
  reg_x::T
  reg_u::T
  slew_reg0::T
  slew_um1::Union{AA{T, 1}, Nothing}
  slew_reg::T
  N::Int
  xdim::Int
  udim::Int
end

OCProb() = OCProb{Float64}(repeat([nothing], 14)..., 0.0, 0.0, 0.0, nothing, 0.0, 0, 0, 0)

OCProb{T}() where {T} = OCProb{T}(repeat([nothing], 14)..., 0.0, 0.0, 0.0, nothing, 0.0, 0, 0, 0)

function set_cost!(
  prob::OCProb{T},
  Q::AA{T, 3},
  R::AA{T, 3},
  X_ref::AA{T, 2},
  U_ref::AA{T, 2},
) where {T}
  prob.Q, prob.R, prob.X_ref, prob.U_ref = Q, R, X_ref, U_ref
  prob.N, prob.xdim, prob.udim = size(Q, 3), size(Q, 1), size(R, 1)
  prob.slew_um1 = prob.slew_um1 != nothing ? prob.slew_um1 : zeros(T, prob.udim)
  return
end

function set_dyn!(
  prob::OCProb{T},
  x0::AA{T, 1},
  f::AA{T, 2},
  fx::AA{T, 3},
  fu::AA{T, 3},
  X_prev::AA{T, 2},
  U_prev::AA{T, 2},
) where {T}
  prob.x0 = x0
  prob.f, prob.fx, prob.fu, prob.X_prev, prob.U_prev = f, fx, fu, X_prev, U_prev
  prob.N, prob.xdim, prob.udim = size(fx, 3), size(fx, 1), size(fu, 2)
  prob.slew_um1 = prob.slew_um1 != nothing ? prob.slew_um1 : zeros(T, prob.udim)
  return
end

function set_xbounds!(prob::OCProb{T}, lx::AA{T, 2}, ux::AA{T, 2}) where {T}
  prob.lx, prob.ux = lx, ux
  @assert prob.xdim == nothing || prob.xdim == size(lx, 1) == size(ux, 1)
  @assert prob.N == nothing || prob.N == size(lx, 2) == size(ux, 2)
  prob.N, prob.xdim = size(lx, 2), size(lx, 1)
  return
end

function set_ubounds!(prob::OCProb{T}, lu::AA{T, 2}, uu::AA{T, 2}) where {T}
  prob.lu, prob.uu = lu, uu
  @assert prob.udim == nothing || prob.udim == size(lu, 1) == size(uu, 1)
  @assert prob.N == nothing || prob.N == size(lu, 2) == size(uu, 2)
  prob.N, prob.udim = size(lu, 2), size(lu, 1)
  prob.slew_um1 = prob.slew_um1 != nothing ? prob.slew_um1 : zeros(T, prob.udim)
  return
end

function set_ctrl_slew!(
  prob::OCProb{T};
  slew_reg=nothing,
  slew_reg0=nothing,
  slew_um1=nothing,
) where {T}
  prob.slew_reg = slew_reg != nothing ? slew_reg : prob.slew_reg
  prob.slew_reg0 = slew_reg0 != nothing ? slew_reg0 : prob.slew_reg0
  prob.slew_um1 = slew_um1 != nothing ? slew_um1 : prob.slew_um1
end

function objective(prob::OCProb{T}, X::AA{T, 2}, U::AA{T, 2}) where {T}
  P, q, _ = qp_repr_Pq(prob)
  z = vcat(reshape(U, :), reshape(X, :))
  return z' * (P * z) + q' * z
end

function rollout!(prob::OCProb{T}, X::AA{T, 2}, U::AA{T, 2}) where {T}
  @views begin
    X[:, 1] = prob.f[:, 1] + prob.fu[:, :, 1] * (U[:, 1] - prob.U_prev[:, 1])
    for i in 2:(prob.N)
      X[:, i] = (
        prob.f[:, i] +
        prob.fu[:, :, i] * (U[:, i] - prob.U_prev[:, i]) +
        prob.fx[:, :, i] * (X[:, i - 1] - prob.X_prev[:, i - 1])
      )
    end
  end
  return
end

function rollout(prob::OCProb{T}, U::AA{T, 2}) where {T}
  X = zeros(T, prob.xdim, prob.N)
  rollout!(prob, X, U)
  return X
end

function rollout!(prob::OCProb{T}, X::AA{T, 2}, U::AA{T, 2}, L::AA{T, 3}, l::AA{T, 2}) where {T}
  @views begin
    U[:, 1] = l[:, 1] + L[:, :, 1] * prob.x0
    X[:, 1] = prob.f[:, 1] + prob.fu[:, :, 1] * (U[:, 1] - prob.U_prev[:, 1])
    for i in 2:(prob.N)
      U[:, i] = l[:, i] + L[:, :, i] * X[:, i - 1]
      X[:, i] = (
        prob.f[:, i] +
        prob.fu[:, :, i] * (U[:, i] - prob.U_prev[:, i]) +
        prob.fx[:, :, i] * (X[:, i - 1] - prob.X_prev[:, i - 1])
      )
    end
  end
  return
end

function rollout(prob::OCProb{T}, L::AA{T, 3}, l::AA{T, 2}) where {T}
  X, U = zeros(T, prob.xdim, prob.N), zeros(T, prob.udim, prob.N)
  rollout!(prob, X, U, L, l)
  return X, U
end

function shorten_horizon(N::Integer, xs::AA{T}...) where {T}
  n = length(xs)
  ret = Array{AA{T}, 1}(undef, n)
  @views for i in 1:n
    ret[i] = view(xs[i], fill(:, ndims(xs[i]) - 2)..., 1:N, :)
  end
  return ret
end

function shorten_horizon(prob::OCProb{T}, N::Integer) where {T}
  N = min(N, prob.N)
  probp = OCProb{T}()
  probp.x0 = prob.x0
  @views for s in [:f, :X_prev, :U_prev, :X_ref, :U_ref, :lx, :ux, :lu, :uu]
    p = getproperty(prob, s)
    p = p != nothing ? p[:, 1:N] : nothing
    setproperty!(probp, s, p)
  end
  @views for s in [:fx, :fu, :Q, :R]
    p = getproperty(prob, s)
    p = p != nothing ? p[:, :, 1:N] : nothing
    setproperty!(probp, s, p)
  end
  probp.reg_x = prob.reg_x
  probp.reg_u = prob.reg_u

  probp.slew_reg0 = prob.slew_reg0
  probp.slew_um1 = prob.slew_um1
  probp.slew_reg = prob.slew_reg

  probp.N = N
  probp.xdim = prob.xdim
  probp.udim = prob.udim
  return probp
end

function make_prob(
  x0::AA{T, 1},
  f::AA{T, 2},
  fx::AA{T, 3},
  fu::AA{T, 3},
  X_prev::AA{T, 2},
  U_prev::AA{T, 2},
  Q::AA{T, 3},
  R::AA{T, 3},
  X_ref::AA{T, 2},
  U_ref::AA{T, 2};
  settings...,
) where {T}
  settings = Dict{Symbol, Any}(settings)
  lx, ux = get(settings, :lx, nothing), get!(settings, :ux, nothing)
  lu, uu = get(settings, :lu, nothing), get!(settings, :uu, nothing)
  reg_x = get(settings, :reg_x, 0.0)
  reg_u = get(settings, :reg_u, 0.0)
  lx, ux, lu, uu = map(z -> (z == nothing || prod(size(z)) != 0) ? z : nothing, (lx, ux, lu, uu))
  prob = OCProb{T}()
  prob.reg_x, prob.reg_u = reg_x, reg_u
  set_dyn!(prob, x0, f, fx, fu, X_prev, U_prev)
  set_cost!(prob, Q, R, X_ref, U_ref)

  # errors catching ###################################
  if (lu != nothing && uu == nothing) || (lu == nothing && uu != nothing)
    @warn("Only one of two control box constraints specified, CONTROL CONSTRAINTS WILL BE IGNORED")
  end
  if (lx != nothing && ux == nothing) || (lx == nothing && ux != nothing)
    @warn("Only one of the two state box constraints specified, STATE CONSTRAINTS WILL BE IGNORED")
  end
  (lx != nothing && ux != nothing) && (set_xbounds!(prob, lx, ux))
  (lu != nothing && uu != nothing) && (set_ubounds!(prob, lu, uu))

  set_ctrl_slew!(
    prob;
    slew_reg0=get(settings, :slew_reg0, nothing),
    slew_um1=get(settings, :slew_um1, nothing),
    slew_reg=get(settings, :slew_reg, nothing),
  )
  return prob
end

#function make_solver(
#  ::Type{SolverT},
#  x0::AA{T, 1},
#  f::AA{T, 2},
#  fx::AA{T, 3},
#  fu::AA{T, 3},
#  X_prev::AA{T, 2},
#  U_prev::AA{T, 2},
#  Q::AA{T, 3},
#  R::AA{T, 3},
#  X_ref::AA{T, 2},
#  U_ref::AA{T, 2};
#  settings...,
#) where {SolverT, T}
#  prob = make_prob(x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref; settings...)
#  return SolverT(prob)
#end

function args2prob(args...; settings...)
  return make_prob(args...; settings...)
end

function prob2args(prob; settings...)
  settings = Dict{Symbol, Any}(settings)
  x0, f, fx, fu, X_prev, U_prev =
    map(s -> getproperty(prob, s), [:x0, :f, :fx, :fu, :X_prev, :U_prev])
  Q, R, X_ref, U_ref = map(s -> getproperty(prob, s), [:Q, :R, :X_ref, :U_ref])
  reg_x, reg_u = prob.reg_x, prob.reg_u
  settings[:reg_x], settings[:reg_u] = reg_x, reg_u
  return x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref, settings
end

#function solve(
#  ::Type{SolverT},
#  x0::AA{T, 1},
#  f::AA{T, 2},
#  fx::AA{T, 3},
#  fu::AA{T, 3},
#  X_prev::AA{T, 2},
#  U_prev::AA{T, 2},
#  Q::AA{T, 3},
#  R::AA{T, 3},
#  X_ref::AA{T, 2},
#  U_ref::AA{T, 2};
#  settings...,
#) where {SolverT, T}
#  solver = make_solver(SolverT, x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref; settings...)
#  return solve!(solver; settings...)
#end
#function solve(
#  tp::String,
#  x0::AA{T, 1},
#  f::AA{T, 2},
#  fx::AA{T, 3},
#  fu::AA{T, 3},
#  X_prev::AA{T, 2},
#  U_prev::AA{T, 2},
#  Q::AA{T, 3},
#  R::AA{T, 3},
#  X_ref::AA{T, 2},
#  U_ref::AA{T, 2};
#  settings...,
#) where {SolverT, T}
#  return solve(eval(Symbol(tp)), x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref; settings...)
#end

function dynamics_violation(prob::OCProb{T}, X::AA{T, 2}, U::AA{T, 2}) where {T}
  viols = zeros(T, prob.N)
  @views viol = norm(X[:, 1] - (prob.f[:, 1] + prob.fu[:, :, 1] * (U[:, 1] - prob.U_prev[:, 1])))
  viols[1] = viol
  @views for j in 2:(prob.N)
    viol_old = viol
    viol += norm(
      X[:, j] - (
        prob.f[:, j] +
        prob.fx[:, :, j] * (X[:, j - 1] - prob.X_prev[:, j - 1]) +
        prob.fu[:, :, j] * (U[:, j] - prob.U_prev[:, j])
      ),
    )
    viols[j] = viol - viol_old
  end
  return viol, viols
end
# main class ###################################################################
