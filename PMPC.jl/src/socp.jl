# aliases ##########################################################################################
const AA, SpMat, F64 = AbstractArray, SparseMatrixCSC, Float64

mutable struct ConeProblem
  l::Int                             # number of non-negative 
  q::Vector{<:Int}                   # array of second-order cones sizes
  e::Int                             # number of exponential cones (each one is of size 3 always)
  G::Union{SpMat{F64, Int}, Nothing} # inequality matrix G * x <=_K h
  A::Union{SpMat{F64, Int}, Nothing} # equality matrix
  c::AA{F64, 1}                      # cost vector: min_x c^T x
  h::Union{AA{F64, 1}, Nothing}      # right hand side of the inequlity constraints G * x <=_K h
  b::Union{AA{F64, 1}, Nothing}      # right hand side of the equality constraint
end

# solver routines ##################################################################################
struct ECOS_result
  x::Array{Float64, 1}
  y::Array{Float64, 1}
  z::Array{Float64, 1}
  s::Array{Float64, 1}
  info::ECOS.stats
end

function ECOS_setup(problem::ConeProblem; settings...)
  l, q, e, G, A, c, h, b = [getproperty(problem, k) for k in [:l, :q, :e, :G, :A, :c, :h, :b]]
  G_ = G != nothing ? copy(G) : spzeros(0, length(c))
  A_ = A != nothing ? copy(A) : spzeros(0, length(c))
  c_ = copy(c)
  h_ = h != nothing ? copy(h) : zeros(0)
  b_ = b != nothing ? copy(b) : zeros(0)
  m, n = size(G_)
  p = size(A_, 1)

  # constraints must add to the size of G
  @assert (l + sum(q) + 3 * e == m) "m = $(m), but (l, sum(q), e) = $((l, sum(q), e))"
  @assert length(c_) == size(G_, 2) == size(A_, 2)
  @assert size(G_, 1) == length(h_)
  @assert size(A_, 1) == length(b_)

  # we're using G * x - h <=_K 0 but ECOS is using h - G * x <=_K 0
  G_[(l + 1):end, :] .*= -1
  h_[(l + 1):end, :] .*= -1

  Gpr, Gjc, Gir = G_.nzval, G_.colptr .- 1, G_.rowval .- 1
  Apr, Ajc, Air = A_.nzval, A_.colptr .- 1, A_.rowval .- 1

  probp = ECOS.ECOS_setup(n, m, p, l, length(q), q, e, Gpr, Gjc, Gir, Apr, Ajc, Air, c_, h_, b_)

  # set settings of the problem, loaded memory is non-modifiable
  prob = unsafe_load(probp)
  old_settings = unsafe_load(prob.stgs)
  new_settings = ECOS.settings(
    [
      (
        haskey(settings, property) ?
        typeof(getproperty(old_settings, property))(settings[property]) :
        getproperty(old_settings, property)
      ) for property in propertynames(old_settings)
    ]...,
  )
  unsafe_store!(prob.stgs, new_settings)
  var_ref_for_gc = (Gpr, Gjc, Gir, Apr, Ajc, Air, c_, h_, b_, new_settings)
  return probp, var_ref_for_gc
end

function ECOS_solve(problem::ConeProblem; settings...)
  l, q, e, G, A, c, h, b = [getproperty(problem, k) for k in [:l, :q, :e, :G, :A, :c, :h, :b]]
  (G != nothing && !(typeof(G) <: SpMat{F64, Int})) && (problem.G = sparse(G))
  (A != nothing && !(typeof(A) <: SpMat{F64, Int})) && (problem.A = sparse(A))
  probp, var_ref_for_gc = ECOS_setup(problem; settings...)
  @assert probp != Ptr{ECOS.pwork}(0)
  status = ECOS.ECOS_solve(probp)
  #(status != 0) && (@warn("ECOS status is not optimal, status = $(status)"))
  prob = unsafe_load(probp)
  n, m, p = prob.n, prob.m, prob.p

  info = deepcopy(unsafe_load(prob.info))
  x = copy(unsafe_wrap(Array, prob.x, n))
  y = copy(unsafe_wrap(Array, prob.y, prob.p))
  z = copy(unsafe_wrap(Array, prob.z, m))
  s = copy(unsafe_wrap(Array, prob.s, m))
  ECOS.ECOS_cleanup(probp, 0)
  return ECOS_result(x, y, z, s, info)
end

function COSMO_solve(problem::ConeProblem; settings...)
  l, q, e, G, A, c, h, b = [getproperty(problem, k) for k in [:l, :q, :e, :G, :A, :c, :h, :b]]
  G = G != nothing ? G : spzeros(0, length(problem.c))
  h = h != nothing ? h : zeros(0)
  A = A != nothing ? A : spzeros(0, length(problem.c))
  b = b != nothing ? b : zeros(0)
  !(typeof(G) <: SpMat{F64, Int}) && (G = sparse(G))
  !(typeof(A) <: SpMat{F64, Int}) && (A = sparse(A))
  m, n = size(G)
  @assert (l + sum(q) + 3 * e == m) "m = $(m), but (l, sum(q), e) = $((l, sum(q), e))"
  @assert length(c) == size(G, 2) == size(A, 2)
  @assert size(G, 1) == length(h)
  @assert size(A, 1) == length(b)

  model = COSMO.Model()
  cstr = COSMO.Constraint{F64}[]

  if A != nothing
    push!(cstr, COSMO.Constraint(A, -b, COSMO.ZeroSet))
  end

  # COSMO uses the convention G x + h, but we're using G * x - h 
  k = 0
  if l > 0
    push!(cstr, COSMO.Constraint(-G[(k + 1):(k + l), :], h[(k + 1):(k + l)], COSMO.Nonnegatives))
    k += l
  end
  for q_size in q
    push!(
      cstr,
      COSMO.Constraint(G[(k + 1):(k + q_size), :], -h[(k + 1):(k + q_size)], COSMO.SecondOrderCone),
    )
    k += q_size
  end
  if e > 0
    for _ in 1:e
      push!(
        cstr,
        COSMO.Constraint(G[(k + 1):(k + 3), :], -h[(k + 1):(k + 3)], COSMO.ExponentialCone),
      )
      k += 3
    end
  end
  stgs = COSMO.Settings()
  for pair in settings
    if pair.first in propertynames(stgs)
      setproperty!(stgs, pair.first, pair.second)
    end
  end
  COSMO.assemble!(model, spzeros(n, n), c, cstr; settings=stgs)

  result = COSMO.optimize!(model)
  return result
end

function JuMP_solve(problem::ConeProblem; settings...)
  l, q, e, G, A, c, h, b = [getproperty(problem, k) for k in [:l, :q, :e, :G, :A, :c, :h, :b]]
  G = G != nothing ? G : spzeros(0, length(problem.c))
  h = h != nothing ? h : zeros(0)
  A = A != nothing ? A : spzeros(0, length(problem.c))
  b = b != nothing ? b : zeros(0)
  (G != nothing && !(typeof(G) <: SpMat{F64, Int})) && (problem.G = sparse(G))
  (A != nothing && !(typeof(A) <: SpMat{F64, Int})) && (problem.A = sparse(A))

  if false
    model = JuMP.Model(
      optimizer_with_attributes(
        () -> ECOS.Optimizer(),
        Dict([string(p.first) => p.second for p in settings])...,
      ),
    )
  else
    model = JuMP.Model(optimizer_with_attributes(Mosek.Optimizer, "QUIET" => !get(settings, :verbose, false)))
  end
  z = @variable(model, z[1:length(problem.c)])
  cstr = []
  if size(A, 1) > 0
    push!(cstr, @constraint(model, A * z - b .== 0))
  end
  k = 0
  if l > 0
    push!(cstr, @constraint(model, -G[1:(problem.l), :] * z + h[1:l] in MOI.Nonnegatives(l)))
    k += l
  end
  if length(q) > 0
    for q_size in q
      push!(
        cstr,
        @constraint(
          model,
          G[(k + 1):(k + q_size), :] * z - h[(k + 1):(k + q_size)] in MOI.SecondOrderCone(q_size)
        )
      )
      k += q_size
    end
  end
  if e > 0
    for _ in 1:e
      push!(
        cstr,
        @constraint(model, G[(k + 1):(k + 3), :] * z - h[(k + 1):(k + 3)] in MOI.ExponentialCone())
      )
      k += 3
    end
  end
  @objective(model, Min, c' * z)
  optimize!(model)

  return (x=value.(z),)
end

# utilities ########################################################################################
function Pqr2Gh(P::AA{F64, 2}, q::AA{F64, 1}, r::F64=0.0)
  # ||  t - alf  ||
  # || L * x - b ||_2 <= t - bet
  # is equivalent to 
  # 0.5 * x' * P * x + q' * x + r <= t
  # L = 1 / sqrt(2) * P^(1/2)
  # b = L' \ (-q / 2)
  # alf = -(b' * b - r) - 0.25
  # bet = 0.25 - (b' * b - r)

  is_sparse = typeof(P) <: SpMat{F64, Int}
  F = cholesky(P)
  L = try # try if this is permuted factorization
    p = F.p
    sparse(sparse(F.L)[sortperm(p), :]') / sqrt(2.0)
  catch
    (is_sparse ? sparse(F.L)' : F.L') / sqrt(2.0)
  end

  # check error
  err = norm(L' * L - 0.5 * P)
  if err > 1e-9
    @warn("Error in P reconstruction is $(err)")
  end

  b = L' \ (-q / 2)
  bTb = dot(b, b)
  alf, bet = -(bTb - r) - 0.25, 0.25 - (bTb - r)

  G_left = is_sparse ? vcat(spzeros(2, size(L, 2)), L) : vcat(zeros(2, size(L, 2)), L)
  h = vec(vcat(alf, bet, b))
  G_right = vcat(sparse([1.0, 1.0]), spzeros(size(L, 1)))
  # we've produced the convention G * x - h <=_K
  return G_left, G_right, h
end


function lsocp_repr_Pq(probs::AA{OCProb{F64}, 1}, Nc::Integer; settings...)
  M = length(probs)
  @assert M >= 1
  N, xdim, udim = probs[1].N, probs[1].xdim, probs[1].udim
  Nc = Nc >= 0 ? Nc : N
  Nf = N - Nc
  G_left_s = Vector{SpMat{F64, Int}}(undef, M)
  G_right_s = Vector{SpMat{F64, Int}}(undef, M)
  h_s = Vector{Vector{F64}}(undef, M)
  #@threads for i in 1:M
  for i in 1:M
    G_left_s[i], G_right_s[i], h_s[i] = Pqr2Gh(qp_repr_Pq(probs[i])...)
  end
  G_right = blockdiag(G_right_s...)
  G_right = hcat(G_right, efficient_vcat(G_right_s))
  h = reduce(vcat, h_s)

  Gs = Vector{SpMat{F64, Int}}(undef, M)
  #@threads for i in 1:M
  for i in 1:M
    G_ucons = G_left_s[i][:, 1:(Nc * udim)]
    G_rest = G_left_s[i][:, (Nc * udim + 1):end]
    row_nb = size(G_ucons, 1)
    mf = N * xdim + Nf * udim
    Gs[i] = hcat(G_ucons, spzeros(row_nb, mf * (i - 1)), G_rest, spzeros(row_nb, mf * (M - i)))
  end
  G_left = efficient_vcat(Gs)
  return G_left, G_right, h
end


# problem augmentation #############################################################################
function augment_cone_problem!(problem::ConeProblem; settings...)
  # extra linear equality
  if haskey(settings, :extra_cstr)
    l, q, e, G_left, G_right, h, c_left, c_right = settings[:extra_cstr]
    G_left, G_right = sparse(G_left), sparse(G_right)
    @assert l + sum(q) + 3 * e == size(G_left, 1)
    @assert size(G_left, 1) == size(G_right, 1)
    @assert length(c_left) == size(G_left, 2)
    @assert length(h) == size(G_left, 1)

    n_fill_cols = size(problem.G, 2) - size(G_left, 2)
    n_new_vars = size(G_right, 2)

    prob_st, prob_en = 1, problem.l
    st, en = 1, l

    G_lin = vcat(
      hcat(problem.G[prob_st:prob_en, :], spzeros(prob_en - prob_st + 1, n_new_vars)),
      hcat(G_left[st:en, :], spzeros(l, n_fill_cols), G_right[st:en, :]),
    )
    h_lin = vcat(problem.h[prob_st:prob_en], h[st:en])


    prob_q_sum, q_sum = sum(problem.q), sum(q)
    prob_st, prob_en = prob_en + 1, problem.l + prob_q_sum
    st, en = en + 1, en + q_sum

    G_soc = vcat(
      hcat(problem.G[prob_st:prob_en, :], spzeros(prob_en - prob_st + 1, n_new_vars)),
      hcat(G_left[st:en, :], spzeros(en - st + 1, n_fill_cols), G_right[st:en, :]),
    )
    h_soc = vcat(problem.h[prob_st:prob_en], h[st:en])

    prob_st, prob_en = prob_en + 1, prob_en + 3 * problem.e
    st, en = en + 1, en + 3 * e
    G_exp = vcat(
      hcat(problem.G[prob_st:prob_en, :], spzeros(prob_en - prob_st + 1, n_new_vars)),
      hcat(G_left[st:en, :], spzeros(en - st + 1, n_fill_cols), G_right[st:en, :]),
    )
    h_exp = vcat(problem.h[prob_st:prob_en], h[st:en])

    problem.G = vcat(G_lin, G_soc, G_exp)
    problem.h = vcat(h_lin, h_soc, h_exp)
    problem.l = problem.l + l
    problem.q = vcat(problem.q, q)
    problem.e = problem.e + e
    @assert problem.l + sum(problem.q) + 3 * problem.e == size(problem.G, 1)

    problem.c = copy(problem.c)
    problem.c[1:length(c_left)] .+= c_left
    problem.c = vcat(problem.c, c_right)

    problem.A = hcat(problem.A, spzeros(size(problem.A, 1), size(G_right, 2)))
  end

  if haskey(settings, :extra_eq)
    A_left, A_right, b, c_left, c_right = settings[:extra_cstr]

    n_fill_cols = size(problem.A, 2) - size(A_left, 2)
    n_new_vars = size(A_right, 2)

    problem.A = vcat(
      hcat(problem.A, spzeros(size(problem.A, 1), n_new_vars)),
      hcat(A_left, spzeros(size(A_left, 1), n_fill_cols), A_right),
    )
    problem.b = vcat(problem.b, b)
    problem.c = copy(problem.c)
    problem.c[1:length(c_left)] .+= c_left
    problem.c = vcat(problem.c, zeros(n_fill_cols), c_right)
  end
  return
end

# constraints translations #########################################################################
function make_logbarrier_constraint(
  g::AbstractVector{F64},
  hi::F64,
  alpha::F64;
  solver::String="ecos",
)
  # convention G * x - h <=_K 0
  @assert lowercase(solver) in ["ecos", "cosmo", "jump"]
  if lowercase(solver) == "ecos"
    # exp(x / z) <= y / z
    # x / z <= log(y) - log(z)
    # -log(y) + log(z) <= -x / z
    # for z = 1, -log(y) <= -x
    # for min t, we have (x = -alpha * t, y = -alpha * (g' * u), z = 1)
    G_left = vcat(spzeros(1, length(g)), -alpha * g', spzeros(1, length(g)))
    h = [0; -alpha * hi; -1]
    #elseif lowercase(solver) in ["cosmo", "jump"]
  elseif lowercase(solver) in ["cosmo", "jump"]
    # y * exp(x / y) <= z
    # log(y) + x / y <= log(z)
    # -log(z) <= -x - log(y)
    # for y = 1, -log(z) <= -x
    # for min t, we have (x = -t * alpha, y = 1, z = -alpha * (g' * u))
    G_left = vcat(spzeros(1, length(g)), spzeros(1, length(g)), -alpha * g')
    h = [0; -1; -alpha * hi]
  end
  G_right = sparse([-alpha; 0; 0])

  return G_left, G_right, h
end

function smoothen_linear_inequlities(
  A::SpMat{F64, Int},
  b::Vector{F64},
  alpha::F64,
  beta::F64=1.0;
  method::String="logbarrier",
  solver::String="ecos",
)
  @assert method in ["logbarrier", "squareplus"]
  m = size(A, 1)
  @assert m == length(b)
  G_left_s, G_right_s = Vector{SpMat{F64, Int}}(undef, m), Vector{SpMat{F64, Int}}(undef, m)
  h_s = Vector{Vector{F64}}(undef, m)
  if method == "logbarrier"
    for i in 1:size(A, 1)
      a, bi = A[i, :], b[i]
      G_left_s[i], G_right_s[i], h_s[i] = make_logbarrier_constraint(a, bi, alpha; solver=solver)
    end
  elseif method == "squareplus"
    for i in 1:size(A, 1)
      a, bi = A[i, :], b[i]
      G_left_s[i] = sparse(vcat(-a', a', spzeros(size(a')...)))
      G_right_s[i] = sparse([2 / beta; 0.0; 0.0])
      h_s[i] = [-bi; bi; 1.0 / alpha]
    end
  end
  G_left, G_right, h = efficient_vcat(G_left_s), blockdiag(G_right_s...), reduce(vcat, h_s)
  return G_left, G_right, h
end


function lsocp_repr_Ab(probs::AA{OCProb{F64}, 1}, Nc::Integer; settings...)
  return lqp_repr_Ab(probs, Nc; settings...)
end
function lsocp_repr_Gla(probs::AA{OCProb{F64}, 1}, Nc::Integer; settings...)
  return lqp_repr_Gla(probs, Nc; settings...)
end
