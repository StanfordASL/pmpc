#### library imports ###########################################################
using Statistics, Printf

#### utils #########################################################################################
function make_probs(
  x0::AA{T, 2},
  f::AA{T, 3},
  fx::AA{T, 4},
  fu::AA{T, 4},
  X_prev::AA{T, 3},
  U_prev::AA{T, 3},
  Q::AA{T, 4},
  R::AA{T, 4},
  X_ref::AA{T, 3},
  U_ref::AA{T, 3};
  settings...,
) where {T}
  settings = Dict{Symbol, Any}(Symbol(p.first) => p.second for p in settings)
  xdim, udim, N, M = size(fu)
  lx, ux, lu, uu = map(s -> get(settings, s, nothing), (:lx, :ux, :lu, :uu))
  lx, ux, lu, uu = map(z -> (z == nothing || prod(size(z)) != 0) ? z : nothing, (lx, ux, lu, uu))
  reg_x, reg_u = map(s -> get(settings, s, 0.0), (:reg_x, :reg_u))
  reg_x = size(reg_x) == () ? repeat([reg_x], M) : reg_x
  reg_u = size(reg_u) == () ? repeat([reg_u], M) : reg_u

  slew_reg0, slew_reg = map(s -> get(settings, s, 0.0), (:slew_reg0, :slew_reg))
  slew_reg0 = size(slew_reg0) == () ? repeat([slew_reg0], M) : slew_reg0
  slew_reg = size(slew_reg) == () ? repeat([slew_reg], M) : slew_reg
  slew_um1 = get(settings, :slew_um1, zeros(T, udim, M))
  slew_um1 = ndims(slew_um1) == 1 ? repeat(slew_um1, 1, M) : slew_um1

  probs = Array{OCProb{T}, 1}(undef, M)
  @views for i in 1:M
    settings[:lx] = lx != nothing ? lx[:, :, i] : nothing
    settings[:ux] = ux != nothing ? ux[:, :, i] : nothing
    settings[:lu] = lu != nothing ? lu[:, :, i] : nothing
    settings[:uu] = uu != nothing ? uu[:, :, i] : nothing
    settings[:reg_x] = reg_x[i]
    settings[:reg_u] = reg_u[i]
    settings[:slew_reg0] = slew_reg0 != nothing ? slew_reg0[i] : 0.0
    settings[:slew_um1] = slew_um1 != nothing ? slew_um1[:, i] : nothing
    settings[:slew_reg] = slew_reg != nothing ? slew_reg[i] : 0.0
    probs[i] = make_prob(
      x0[:, i],
      f[:, :, i],
      fx[:, :, :, i],
      fu[:, :, :, i],
      X_prev[:, :, i],
      U_prev[:, :, i],
      Q[:, :, :, i],
      R[:, :, :, i],
      X_ref[:, :, i],
      U_ref[:, :, i];
      settings...,
    )
  end
  return probs
end

#function make_solvers(
#  ::Type{SolverT},
#  x0::AA{T, 2},
#  f::AA{T, 3},
#  fx::AA{T, 4},
#  fu::AA{T, 4},
#  X_prev::AA{T, 3},
#  U_prev::AA{T, 3},
#  Q::AA{T, 4},
#  R::AA{T, 4},
#  X_ref::AA{T, 3},
#  U_ref::AA{T, 3};
#  settings...,
#) where {SolverT, T}
#  probs = make_probs(x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref; settings...)
#  return map(prob -> SolverT(prob), probs)
#end

function args2probs(args...; settings...)
  return make_probs(args...; settings...)
end

function probs2args(probs; settings...)
  settings = Dict{Symbol, Any}(Symbol(p.first) => p.second for p in settings)
  x0, f, fx, fu, X_prev, U_prev =
    map(s -> catb(getproperty.(probs, s)...), [:x0, :f, :fx, :fu, :X_prev, :U_prev])
  Q, R, X_ref, U_ref = map(s -> catb(getproperty.(probs, s)...), [:Q, :R, :X_ref, :U_ref])
  reg_x = getproperty.(probs, :reg_x)
  reg_u = getproperty.(probs, :reg_u)
  settings[:reg_x], settings[:reg_u] = reg_x, reg_u
  slew_reg0 = getproperty.(probs, :slew_reg0)
  slew_um1 = getproperty.(probs, :slew_um1)
  slew_reg = getproperty.(probs, :slew_reg)
  all(slew_reg0 .!= nothing) && (settings[:slew_reg0] = slew_reg0)
  all(slew_um1 .!= nothing) && (settings[:slew_reg0] = slew_um1)
  all(slew_reg .!= nothing) && (settings[:slew_reg] = slew_reg)
  return x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref, settings
end

function scale_probs_cost!(probs::AA{OCProb{T}, 1}, weights::AA{T, 1}) where {T}
  M = length(probs)
  @assert length(weights) == M
  weights .*= (1 / sum(weights))
  if weights != nothing
    for i in 1:M
      probs[i].Q .*= weights[i]
      probs[i].R .*= weights[i]
      (probs[i].reg_x != nothing) && (probs[i].reg_x *= weights[i])
      (probs[i].reg_u != nothing) && (probs[i].reg_u *= weights[i])
      (probs[i].slew_reg0 != nothing) && (probs[i].slew_reg0 *= weights[i])
      (probs[i].slew_um1 != nothing) && (probs[i].slew_um1 .*= weights[i])
      (probs[i].slew_reg != nothing) && (probs[i].slew_reg *= weights[i])
    end
  end
  return
end

#### large QP (default solver for consensus and single particle problems) ##########################
function lqp_solve(
  probs::AA{OCProb{T}, 1};
  settings...,
)::Tuple{Array, Array, Dict{Symbol, Any}} where {T}
  # setup #####################################################
  settings = Dict(Symbol(p.first) => p.second for p in settings)
  M = length(probs)
  xdim, udim, N = probs[1].xdim, probs[1].udim, probs[1].N
  coerce = get(settings, :coerce, false)

  # read in the consensus horizon
  Nc = get(settings, :Nc, N)
  Nc = Nc >= 0 ? Nc : N
  weights = get(settings, :weights, nothing)
  (weights != nothing) && (scale_probs_cost!(probs, weights))


  # problem building ##########################################
  P, q = lqp_repr_Pq(probs, Nc)
  A, b = lqp_repr_Ab(probs, Nc)
  G, l, u = lqp_repr_Gla(probs, Nc)
  # set default solver
  haskey(settings, :smooth_alpha) && !get(settings, :smooth_cstr, "logbarrier")
  if !haskey(settings, :solver)
    settings[:solver] = length(get(settings, :smooth_cstr, "")) > 0 ? "ecos" : "osqp"
  end
  if lowercase(settings[:solver]) == "osqp"
    solver = OSQPSolver()
  else
    solver = JuMPSolver()
  end
  solver.P, solver.q, solver.A, solver.b = P, q, A, b
  solver.G, solver.l, solver.u = G, l, u

  # problem solving solving ###################################
  z, _ = solve_qp!(solver; settings...)
  X, U = split_lqp_vars(probs, Nc, z)
  @views if coerce
    println("Coercing")
    ubar = mean(U[:, 1:Nc, :]; dims=3)[:, :, 1]
    for i in 1:M
      U[:, 1:Nc, i] .= ubar
      rollout!(probs[i], X[:, :, i], U[:, :, i])
    end
  end

  # constructing the return objects ###########################
  data = Dict{Symbol, Any}(
    :obj => get(settings, :return_objective, false) ? objective(probs, X, U) : T(NaN),
  )
  (get(settings, :return_solver, false)) && (data[:solver] = solver)
  if get(settings, :return_solver_id, false)
    solver_id = haskey(settings, :solver_id) ? settings[:solver_id] : generate_unique_solver_id()
    store_solver!(solver, solver_id)
    data[:solver_id] = solver_id
  end
  return X, U, data
end

function lqp_solve(
  x0::AA{T, 2},
  f::AA{T, 3},
  fx::AA{T, 4},
  fu::AA{T, 4},
  X_prev::AA{T, 3},
  U_prev::AA{T, 3},
  Q::AA{T, 4},
  R::AA{T, 4},
  X_ref::AA{T, 3},
  U_ref::AA{T, 3};
  settings...,
) where {T}
  return lqp_solve(
    make_probs(x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref; settings...);
    settings...,
  )
end

##^# large cone ################################################################

function lcone_solve(probs::AA{OCProb{T}, 1}; settings...) where {T}
  settings = Dict{Symbol, Any}(Symbol(p.first) => p.second for p in settings)
  M = length(probs)
  xdim, udim, N = probs[1].xdim, probs[1].udim, probs[1].N
  ret_obj = get(settings, :ret_obj, false)
  Nc, coerce = get(settings, :Nc, N), get(settings, :coerce, false)
  Nc = Nc >= 0 ? Nc : N
  Nf = N - Nc
  weights = get(settings, :weights, nothing)
  (weights != nothing) && (scale_probs_cost!(probs, weights))
  k = get(settings, :k, M)
  k = k >= 0 ? k : M
  solver_settings =
    Dict{Symbol, Any}(Symbol(p.first) => p.second for p in get(settings, :solver_settings, Dict()))
  get!(solver_settings, :verbose, false)
  verbose = get(settings, :verbose, false)
  verbose && println("SOCP_k = ", k)

  get!(settings, :solver, "ecos")

  begin
    # generate the matrices
    Pq_G_left, Pq_G_right, Pq_h = lcone_repr_Pq(probs, Nc)
    Pq_G = hcat(Pq_G_left, Pq_G_right)
    A, b = lcone_repr_Ab(probs, Nc)
    F, l, u = lcone_repr_Gla(probs, Nc)
    F = hcat(F, spzeros(size(F, 1), size(Pq_G_right, 2)))
    A = hcat(A, spzeros(size(A, 1), size(Pq_G_right, 2)))
    # small value to incentivize anchor y and t otherwise t + y has a degree of freedom
    COST_ANCHOR_EPS = 1e-3
    c = [
      zeros(Nc * udim + M * (N * xdim + Nf * udim))
      (1 + COST_ANCHOR_EPS) * ones(M)
      (1 - COST_ANCHOR_EPS) * k
    ]

    # construct the combined inequality matrix
    y_nonneg = hcat(spzeros(M, size(Pq_G_left, 2)), -I, spzeros(M, 1)) # y vars non-negative
    G = vcat(y_nonneg, Pq_G)
    h = [zeros(M); Pq_h]

    # wrap into a cone problem representation
    @assert length(h) == size(G, 1)
    @assert length(b) == size(A, 1)
    @assert length(c) == size(G, 2) == size(A, 2)
    cone_problem = ConeProblem(M, fill(2 + N * (xdim + udim), M), 0, G, A, c, h, b)
  end

  # handle constraints
  get!(settings, :smooth_cstr, "")
  get!(settings, :smooth_alpha, 1e0)
  get!(settings, :smooth_beta, 1e0)
  if size(F, 1) > 0
    @assert settings[:smooth_cstr] in ["", "logbarrier", "squareplus"]
    if settings[:smooth_cstr] == "logbarrier"
      G_left, G_right, h = smoothen_linear_inequlities(
        vcat(-F, F),
        vcat(-l, u),
        settings[:smooth_alpha];
        method="logbarrier",
        solver=lowercase(settings[:solver]),
      )
      c_left, c_right = zeros(size(G_left, 2)), ones(size(G_right, 2))
      @assert size(G_left, 1) == 3 * (length(l) + length(u))
      augment_cone_problem!(
        cone_problem;
        extra_cstr=(0, Int[], length(l) + length(u), G_left, G_right, h, c_left, c_right),
      )
    elseif settings[:smooth_cstr] == "squareplus"
      G_left, G_right, h = smoothen_linear_inequlities(
        vcat(-F, F),
        vcat(-l, u),
        settings[:smooth_alpha],
        settings[:smooth_beta];
        method="squareplus",
        solver=lowercase(settings[:solver]),
      )
      c_left, c_right = zeros(size(G_left, 2)), ones(size(G_right, 2))
      @assert size(G_left, 1) == 3 * (length(l) + length(u))
      augment_cone_problem!(
        cone_problem;
        extra_cstr=(0, fill(3, length(l) + length(u)), 0, G_left, G_right, h, c_left, c_right),
      )
    elseif settings[:smooth_cstr] == ""
      G_left, h = vcat(-F, F), vcat(-l, u)
      G_right = spzeros(size(G_left, 1), 0)
      c_left, c_right = zeros(length(cone_problem.c)), zeros(0)
      augment_cone_problem!(
        cone_problem;
        extra_cstr=(size(G_left, 1), Int[], 0, G_left, G_right, h, c_left, c_right),
      )
    end

    # incorporate extra, ad hoc constraints
    for extra_cstr in get(settings, :extra_cstrs, Tuple[])
      l, q, e, G_left, G_right, h, c_left, c_right = extra_cstr
      q = convert(Vector{Int}, q)
      G_left, G_right = sparse(G_left), sparse(G_right)
      @assert l + sum(q) + 3 * e == size(G_left, 1) == size(G_right, 1)
      if get(settings, :smooth_cstr, "") == "logbarrier"
        @assert size(G_right, 2) == 0 "We only support left matrix reformulation"
        G_left_new, G_right_new, h_new = smoothen_linear_inequlities(
          G_left[1:l, :],
          h[1:l],
          settings[:smooth_alpha];
          method="logbarrier",
          solver=lowercase(settings[:solver]),
        )
        G_left, G_right = vcat(G_left[l+1:end, :], G_left_new), G_right_new
        h = vcat(h[l+1:end], h_new)
        l, e = 0, e + div(size(G_left_new, 1), 3)
        c_right = ones(size(G_right, 2))
      end
      augment_cone_problem!(
        cone_problem;
        extra_cstr=(l, q, e, G_left, G_right, h, c_left, c_right),
      )
    end
  end

  # solve
  @assert lowercase(settings[:solver]) in ["ecos", "cosmo", "jump", "mosek"]
  if lowercase(settings[:solver]) == "ecos"
    sol = ECOS_solve(cone_problem; solver_settings...)
  elseif lowercase(settings[:solver]) == "cosmo"
    sol = COSMO_solve(cone_problem; solver_settings...)
  elseif lowercase(settings[:solver]) in ["jump", "mosek"]
    sol = JuMP_solve(cone_problem; solver_settings...)
  end

  ts = sol.x[(((Nc + M * (N - Nc)) * udim + M * N * xdim) + 1):end]

  z = sol.x[1:((Nc + M * (N - Nc)) * udim + M * N * xdim)]
  P, q, resid = qp_repr_Pq(probs[1])

  X, U = split_lqp_vars(probs, Nc, sol.x[1:((Nc + M * (N - Nc)) * udim + M * N * xdim)])
  @views if coerce
    println("Coercing")
    ubar = mean(U[:, 1:Nc, :]; dims=3)[:, :, 1]
    for i in 1:M
      U[:, 1:Nc, i] .= ubar
      rollout!(probs[i], X[:, :, i], U[:, :, i])
    end
  end
  obj = ret_obj ? bobjective(probs, X, U) : T(NaN)
  data = Dict{Symbol, Any}(
    #:probs => probs,
    :obj => obj,
    :settings => settings,
    :ts => ts,
    :residual => 0.5 * dot(z, P, z) + dot(z, q) + resid,
  )
  return X, U, data
end
function lcone_solve(
  x0::AA{T, 2},
  f::AA{T, 3},
  fx::AA{T, 4},
  fu::AA{T, 4},
  X_prev::AA{T, 3},
  U_prev::AA{T, 3},
  Q::AA{T, 4},
  R::AA{T, 4},
  X_ref::AA{T, 3},
  U_ref::AA{T, 3};
  settings...,
) where {T}
  probs = make_probs(x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref; settings...)
  return lcone_solve(probs; settings...)
end
##$#############################################################################

#### utility functions for canonical problem representation ########################################
function lqp_generate_problem_matrices(probs::AA{OCProb{T}, 1}; settings...)::Tuple where {T}
  settings = Dict{Symbol, Any}(Symbol(p.first) => p.second for p in settings)
  # read in the consensus horizon
  Nc = get(settings, :Nc, -1)
  Nc = Nc >= 0 ? Nc : probs[1].N

  # scale the problem objectives
  weights = get(settings, :weights, nothing)
  (weights != nothing) && (scale_probs_cost!(probs, weights))

  # problem building
  P, q = lqp_repr_Pq(probs, Nc)
  A, b = lqp_repr_Ab(probs, Nc)
  G, l, u = lqp_repr_Gla(probs, Nc)

  return P, q, A, b, G, l, u
end

function lqp_generate_problem_matrices(
  x0::AA{T, 2},
  f::AA{T, 3},
  fx::AA{T, 4},
  fu::AA{T, 4},
  X_prev::AA{T, 3},
  U_prev::AA{T, 3},
  Q::AA{T, 4},
  R::AA{T, 4},
  X_ref::AA{T, 3},
  U_ref::AA{T, 3};
  settings...,
) where {T}
  return lqp_generate_problem_matrices(
    make_probs(x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref; settings...);
    settings...,
  )
end
