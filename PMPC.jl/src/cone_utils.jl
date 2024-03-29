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
  #b = F' \ (-q / 2)
  #println("error is $(norm(b - b_prime))")
  bTb = dot(b, b)
  alf, bet = -(bTb - r) - 0.25, 0.25 - (bTb - r)

  G_left = is_sparse ? vcat(spzeros(2, size(L, 2)), L) : vcat(zeros(2, size(L, 2)), L)
  h = vec(vcat(alf, bet, b))
  G_right = vcat(sparse([1.0, 1.0]), spzeros(size(L, 1)))
  # we've produced the convention G * x - h <=_K
  return G_left, G_right, h
end


function lcone_repr_Pq(probs::AA{OCProb{F64}, 1}, Nc::Integer; settings...)
  M = length(probs)
  @assert M >= 1
  N, xdim, udim = probs[1].N, probs[1].xdim, probs[1].udim
  Nc = Nc >= 0 ? Nc : N
  Nf = N - Nc
  G_left_s = Vector{SpMat{F64, Int}}(undef, M)
  G_right_s = Vector{SpMat{F64, Int}}(undef, M)
  h_s = Vector{Vector{F64}}(undef, M)
  @threads for i in 1:M
    G_left_s[i], G_right_s[i], h_s[i] = Pqr2Gh(qp_repr_Pq(probs[i])...)
  end
  G_right = blockdiag(G_right_s...)
  G_right = hcat(G_right, efficient_vcat(G_right_s))
  h = reduce(vcat, h_s)

  Gs = Vector{SpMat{F64, Int}}(undef, M)
  @threads for i in 1:M
    G_ucons = G_left_s[i][:, 1:(Nc * udim)]
    G_rest_u = G_left_s[i][:, (Nc * udim + 1):N * udim]
    G_x = G_left_s[i][:, (N * udim + 1):end]
    #@assert size(G_ucons, 2) == Nc * udim
    #@assert size(G_rest_u, 2) == Nf * udim
    #@assert size(G_x, 2) == N * xdim
    m = size(G_ucons, 1)
    Gs[i] = hcat(G_ucons, spzeros(m, Nf * udim * (i - 1)), G_rest_u, spzeros(m, Nf * udim * (M - i)), spzeros(m, N * xdim * (i - 1)), G_x, spzeros(m, (M - i) * N * xdim))
    #@assert size(Gs[i], 2) == udim * (Nc + M * Nf) + xdim * N * M
  end
  #G_left = efficient_vcat(Gs)
  G_left = reduce(vcat, Gs)
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
  @assert lowercase(solver) in ["ecos", "cosmo", "jump", "gurobi", "mosek"]
  if lowercase(solver) == "ecos"
    # exp(x / z) <= y / z
    # x / z <= log(y) - log(z)
    # -log(y) + log(z) <= -x / z
    # for z = 1, -log(y) <= -x
    # for min t, we have (x = -alpha * t, y = -alpha * (g' * u), z = 1)
    G_left = vcat(spzeros(1, length(g)), -alpha * g', spzeros(1, length(g)))
    h = [0; -alpha * hi; -1]
    #elseif lowercase(solver) in ["cosmo", "jump"]
  elseif lowercase(solver) in ["cosmo", "jump", "gurobi", "mosek"]
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


function lcone_repr_Ab(probs::AA{OCProb{F64}, 1}, Nc::Integer; settings...)
  return lqp_repr_Ab(probs, Nc; settings...)
end
function lcone_repr_Gla(probs::AA{OCProb{F64}, 1}, Nc::Integer; settings...)
  return lqp_repr_Gla(probs, Nc; settings...)
end
