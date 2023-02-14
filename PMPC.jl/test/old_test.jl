using LinearAlgebra, SparseArrays, Printf, Random
using BenchmarkTools, PyPlot
include(
  joinpath(
    ENV["HOME"],
    "Dropbox/stanford/meta",
    "camelid/camelid/controllers/pmpc/PMPC/src",
    "PMPC.jl",
  ),
)
Random.seed!(2020)

N, xdim, udim = 100, 2, 1
x0 = [5.0; 5.0]
fx = PMPC.catb(map(_ -> [1.0 0.1; 0.0 1.0], 1:N)...)
fu = PMPC.catb(map(_ -> [0.0; 1.0][:, :], 1:N)...)
f = PMPC.catb(map(i -> i == 1 ? fx[i] * x0 : zeros(2), 1:N)...)
X_prev, U_prev = zeros(xdim, N), zeros(udim, N)
X_ref, U_ref = zeros(xdim, N), zeros(udim, N)

Q = PMPC.catb(map(i -> diagm(0 => (i == N ? [1e2, 1e2] : [1e2, 1.0])), 1:N)...)
q = PMPC.catb(map(_ -> zeros(xdim), 1:N)...)
R = PMPC.catb(map(_ -> diagm(0 => [1e-2]), 1:N)...)
r = PMPC.catb(map(_ -> zeros(udim), 1:N)...)

function test_qp_ilqr()
  plot_flag = true

  prob = PMPC.OCProb()
  PMPC.set_dyn!(prob, x0, f, fx, fu, X_prev, U_prev)
  PMPC.set_cost!(prob, Q, R, X_ref, U_ref)
  PMPC.set_ubounds!(prob, -5 * ones(udim, N), 5 * ones(udim, N))
  PMPC.set_xbounds!(prob, -5.0 * ones(xdim, N), 5.0 * ones(xdim, N))

  for i in 1:1
    @time x_qp, u_qp = let

      print("    ")
      @time solver = PMPC.OSQPSolver(prob)
      print("    ")
      @time resx, _ = PMPC.solve!(solver)

      x = [x0 reshape(resx[(N * udim + 1):end], (xdim, :))]
      u = reshape(resx[1:(N * udim)], (udim, :))

      if plot_flag || i == 1
        figure(2)
        clf()
        for i in 1:xdim
          plot(x[i, :], label="x" * string(i))
        end
        for i in 1:udim
          plot(u[i, :], label="u" * string(i))
        end
        legend()
      end

      x, u
    end
  end
  println("-----------------------------------------------------------")

  x_lqr, u_lqr = let
    #L, l = iLQR_backward(f, fx, fu, Q, q, R, r)
    #x, u = iLQR_forward(f, fu, fx, L, l)
    solver = PMPC.iLQRSolver(prob)
    X, U = PMCP.solve!(solver)

    X = [x0 X]

    if plot_flag
      figure(1)
      clf()
      for i in 1:xdim
        plot(X[i, :], label="x" * string(i))
      end
      for i in 1:udim
        plot(U[i, :], label="u" * string(i))
      end
      legend()
    end

    X, U
  end
end

function test_cons()
  x0 = [5.0; 5.0; 5.0; 5.0]
  M, N, xdim, udim = 100, 30, 4, 2
  f = zeros(xdim, N, M)
  fx, fu = zeros(xdim, xdim, N, M), zeros(xdim, udim, N, M)
  rng = range(0.1, stop=0.9, length=10)
  @views for i in 1:M
    r1, r2 = rand(rng), rand(rng)
    @views for j in 1:N
      fx[:, :, j, i] = Float64[1 r1 0 0; 0 1 0 0; 0 0 1 r2; 0 0 0 1]
      fu[:, :, j, i] = Float64[0 0; 1 0; 0 0; 0 1]
      f[:, j, i] = j == 1 ? fx[:, :, j, i] * x0 : zeros(xdim)
    end
  end
  X_prev, U_prev = zeros(xdim, N, M), zeros(udim, N, M)

  Q, R = zeros(xdim, xdim, N, M), zeros(udim, udim, N, M)
  @views for i in 1:M
    @views for j in 1:N
      Q[:, :, j, i] = diagm(0 => ones(xdim))
      R[:, :, j, i] = diagm(0 => ones(udim))
    end
  end
  X_ref, U_ref = zeros(xdim, N, M), zeros(udim, N, M)
  x0 = repeat(x0, 1, M)

  @time Xs, Us, Ls, ls =
    PMPC.admm_fbsolve(x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref; rho=5e0, max_it=300)
  #@time Xs, Us = PMPC.admm_solve(x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref,
  #                               U_ref; rho=50.0, max_it=100)

  #=
  probs = map(_ -> PMPC.OCProb(), 1:M)
  @views for i in 1:M
  probs[i].rho_res_x, probs[i].rho_res_u = 0.0, 0.0
  PMPC.set_dyn!(probs[i], x0[:, i], f[:, :, i], fx[:, :, :, i], 
  fu[:, :, :, i], X_prev[:, :, i], U_prev[:, :, i])
  PMPC.set_cost!(probs[i], Q[:, :, :, i], R[:, :, :, i], X_ref[:, :, i],
  U_ref[:, :, i])
  end
  probs = map(p -> PMPC.shorten_horizon(5, p), probs)
  f, fx, fu, X_prev, U_prev = PMPC.shorten_horizon(5, f, fx, fu, X_prev, U_prev)
  Q, R, X_ref, U_ref = PMPC.shorten_horizon(5, Q, R, X_ref, U_ref)
  println(join(string.([size(x0), size(f), size(fx), size(fu), size(X_prev),
  size(U_prev)]), ", "))
  println(join(string.([size(Q), size(R), size(X_ref), size(U_ref)]), ", "))
  @time Xs, Us, Ls, ls = PMPC.bilqr_solve(x0, f, fx, fu, X_prev, U_prev, Q, R,
  X_ref, U_ref)
  @time Xs, Us, Ls, ls = PMPC.bilqr_solve(x0, f, fx, fu, X_prev, U_prev, Q, R,
  X_ref, U_ref)
  @time Xs, Us, Ls, ls = PMPC.bilqr_solve(x0, f, fx, fu, X_prev, U_prev, Q, R,
  X_ref, U_ref)
  #@time Xs, Us, Ls, ls = PMPC.bilqr_solve(probs)
  =#

  for i in 1:min(M, 10)
    figure(i)
    clf()
    for j in 1:xdim
      plot([x0[j]; Xs[j, :, i]], label="x" * string(j))
    end
    legend()
  end

  figure(M + 1)
  clf()
  for i in 1:min(M)
    for j in 1:udim
      plot(Us[j, :, i])
    end
  end
end

function test_sqp()
  x0 = [5.0; 5.0; 5.0; 5.0]
  M, N, xdim, udim = 1, 30, 4, 2
  f = zeros(xdim, N, M)
  fx, fu = zeros(xdim, xdim, N, M), zeros(xdim, udim, N, M)
  rng = range(0.1, stop=0.9, length=10)
  @views for i in 1:M
    r1, r2 = rand(rng), rand(rng)
    @views for j in 1:N
      fx[:, :, j, i] = Float64[1 r1 0 0; 0 1 0 0; 0 0 1 r2; 0 0 0 1]
      fu[:, :, j, i] = Float64[0 0; 1 0; 0 0; 0 1]
      f[:, j, i] = j == 1 ? fx[:, :, j, i] * x0 : zeros(xdim)
    end
  end
  X_prev, U_prev = zeros(xdim, N, M), zeros(udim, N, M)

  Q, R = zeros(xdim, xdim, N, M), zeros(udim, udim, N, M)
  @views for i in 1:M
    @views for j in 1:N
      Q[:, :, j, i] = diagm(0 => ones(xdim))
      R[:, :, j, i] = diagm(0 => ones(udim))
    end
  end
  X_ref, U_ref = zeros(xdim, N, M), zeros(udim, N, M)

  f, fx, fu = f[:, :, 1], fx[:, :, :, 1], fu[:, :, :, 1]
  X_prev, U_prev = X_prev[:, :, 1], U_prev[:, :, 1]
  Q, R = Q[:, :, :, 1], R[:, :, :, 1]
  X_ref, U_ref = X_ref[:, :, 1], U_ref[:, :, 1]

  prob = PMPC.OCProb{Float64}()
  PMPC.set_dyn!(prob, x0, f, fx, fu, X_prev, U_prev)
  PMPC.set_cost!(prob, Q, R, X_ref, U_ref)

  fbs = PMPC.FBSolver(prob)
  z = randn(N * (xdim * udim + xdim + udim + udim))
  PMPC.prox_setup!(fbs, 5e1 * ones(fbs.n))
  @time z, _ = PMPC.prox!(fbs, zeros(fbs.n))
  @time z, _ = PMPC.prox!(fbs, zeros(fbs.n))
  @time z, _ = PMPC.prox!(fbs, zeros(fbs.n))
  L, x, l, u = PMPC.split_fbvars(fbs, z)

  figure(1)
  clf()
  for i in 1:xdim
    plot([x0[i]; x[i, :]], label="x" * string(i))
  end
  for i in 1:udim
    plot(u[i, :], label="u" * string(i))
  end

  L = PermutedDimsArray(L, (2, 1, 3))
  x, u = PMPC.rollout(prob, L, l)

  figure(2)
  clf()
  for i in 1:xdim
    plot([x0[i]; x[i, :]], label="x" * string(i))
  end
  for i in 1:udim
    plot(u[i, :], label="u" * string(i))
  end
end

function test_large_sqp()
  x0 = [5.0; 5.0; 5.0; 5.0]
  M, N, xdim, udim = 50, 30, 4, 2

  f = zeros(xdim, N, M)
  rho_res_x, rho_res_u = 0.0, 0.0
  fx, fu = zeros(xdim, xdim, N, M), zeros(xdim, udim, N, M)
  rng = range(0.1, stop=0.9, length=10)
  @views for i in 1:M
    r1, r2 = rand(rng), rand(rng)
    @views for j in 1:N
      fx[:, :, j, i] = Float64[1 r1 0 0; 0 1 0 0; 0 0 1 r2; 0 0 0 1]
      fu[:, :, j, i] = Float64[0 0; 1 0; 0 0; 0 1]
      f[:, j, i] = j == 1 ? fx[:, :, j, i] * x0 : zeros(xdim)
    end
  end
  X_prev, U_prev = zeros(xdim, N, M), zeros(udim, N, M)

  Q, R = zeros(xdim, xdim, N, M), zeros(udim, udim, N, M)
  @views for i in 1:M
    @views for j in 1:N
      Q[:, :, j, i] = diagm(0 => ones(xdim))
      R[:, :, j, i] = diagm(0 => ones(udim))
    end
  end
  X_ref, U_ref = zeros(xdim, N, M), zeros(udim, N, M)
  x0 = repeat(x0, 1, M)

  M2 = 10
  probs = map(_ -> PMPC.OCProb{Float64}(), 1:M2)
  @views for i in 1:M2
    probs[i].rho_res_x, probs[i].rho_res_u = rho_res_x, rho_res_u
    PMPC.set_dyn!(
      probs[i],
      x0[:, i],
      f[:, :, i],
      fx[:, :, :, i],
      fu[:, :, :, i],
      X_prev[:, :, i],
      U_prev[:, :, i],
    )
    PMPC.set_cost!(probs[i], Q[:, :, :, i], R[:, :, :, i], X_ref[:, :, i], U_ref[:, :, i])
  end
  global fbs = PMPC.LFBSolver(probs)

  #=
  h = 1e-5
  J = Array{Array{Float64, 1}, 1}(undef, fbs.n)
  for i in 1:fbs.n
  hvec = zeros(fbs.n)
  hvec[i] = h
  fzp, fzm = zeros(fbs.m), zeros(fbs.m)
  PMPC.f_fn!(fzp, fbs, fbs.z + hvec)
  PMPC.f_fn!(fzm, fbs, fbs.z - hvec)
  J[i] = (fzp - fzm) ./ (2.0 * h)
  end
  global J = hcat(J...)
  global fz = PMPC.fz_fn(fbs, fbs.z)
  global fz2 = copy(fz)
  PMPC.fz_fn!(fz2, fbs, fbs.z)
  @assert fz == fz2

  L, X, l, U = PMPC.split_fbvars(fbs, fbs.z)
  global JL = J[:, 1:length(L)]
  global JX = J[:, length(L)+1:length(L)+length(X)]
  global Jl = J[:, length(L)+length(X)+1:length(L)+length(X)+length(l)]
  global JU = J[:, length(L)+length(X)+length(l)+1:end]

  fz_ = fz
  fz = collect(fz)
  global fzL = fz[:, 1:length(L)]
  global fzX = fz[:, length(L)+1:length(L)+length(X)]
  global fzl = fz[:, length(L)+length(X)+1:length(L)+length(X)+length(l)]
  global fzU = fz[:, length(L)+length(X)+length(l)+1:end]
  fz = fz_
  =#


  z, _ = PMPC.solve!(fbs)
  L, X, l, U = PMPC.split_fbvars(fbs, z)
  println(size(X))
  println(size(U))
  L = PermutedDimsArray(L, (2, 1, 3))
  X, U = zeros(xdim, N, M), zeros(udim, N, M)
  PMPC.bilqr_forward!(X, U, x0, f, fx, fu, X_prev, U_prev, L, l)

  Xs, Us = X, U
  for i in 20:min(M, 30)
    figure(i)
    clf()
    for j in 1:xdim
      plot([x0[j]; Xs[j, :, i]], label="x" * string(j))
    end
    legend()
  end

  figure(M + 1)
  clf()
  for i in 1:min(M)
    for j in 1:udim
      plot(Us[j, :, i])
    end
  end

  #=
  fbs2 = PMPC.FBSolver(probs[1])
  PMPC.solve!(fbs2)
  =#
  return
end

function test_large_qp()
  x0 = [5.0; 5.0; 5.0; 5.0]
  M, N, xdim, udim = 100, 30, 4, 2

  f = zeros(xdim, N, M)
  rho_res_x, rho_res_u = 0.0, 0.0
  fx, fu = zeros(xdim, xdim, N, M), zeros(xdim, udim, N, M)
  rng = range(0.1, stop=0.9, length=10)
  @views for i in 1:M
    r1, r2 = rand(rng), rand(rng)
    @views for j in 1:N
      fx[:, :, j, i] = Float64[1 r1 0 0; 0 1 0 0; 0 0 1 r2; 0 0 0 1]
      fu[:, :, j, i] = Float64[0 0; 1 0; 0 0; 0 1]
      f[:, j, i] = j == 1 ? fx[:, :, j, i] * x0 : zeros(xdim)
    end
  end
  X_prev, U_prev = zeros(xdim, N, M), zeros(udim, N, M)

  Q, R = zeros(xdim, xdim, N, M), zeros(udim, udim, N, M)
  @views for i in 1:M
    @views for j in 1:N
      Q[:, :, j, i] = diagm(0 => ones(xdim))
      R[:, :, j, i] = diagm(0 => ones(udim))
    end
  end
  X_ref, U_ref = zeros(xdim, N, M), zeros(udim, N, M)
  x0 = repeat(x0, 1, M)

  probs = map(_ -> PMPC.OCProb{Float64}(), 1:M)
  lu, uu = -ones(udim, N, M), ones(udim, N, M)
  @views for i in 1:M
    probs[i].rho_res_x, probs[i].rho_res_u = rho_res_x, rho_res_u
    PMPC.set_dyn!(
      probs[i],
      x0[:, i],
      f[:, :, i],
      fx[:, :, :, i],
      fu[:, :, :, i],
      X_prev[:, :, i],
      U_prev[:, :, i],
    )
    PMPC.set_cost!(probs[i], Q[:, :, :, i], R[:, :, :, i], X_ref[:, :, i], U_ref[:, :, i])
    PMPC.set_ubounds!(probs[i], lu[:, :, i], uu[:, :, i])
  end


  Nc = -1
  P, q = PMPC.large_qp_repr_Pq(probs, Nc)
  A, b = PMPC.large_qp_repr_Ab(probs, Nc)
  A_, b_ = PMPC.qp_repr_Ab(probs[1])
  G, l, u = PMPC.large_qp_repr_Gla(probs, Nc)

  for i in 1:3
    @time begin
      solver = PMPC.OSQPSolver()
      solver.P, solver.q, solver.A, solver.b = P, q, A, b
      solver.G, solver.l, solver.u = G, l, u
      z, _ = PMPC.solve!(solver)
    end
  end
  solver = PMPC.OSQPSolver()
  solver.P, solver.q, solver.A, solver.b = P, q, A, b
  solver.G, solver.l, solver.u = G, l, u
  z, _ = PMPC.solve!(solver)
  X, U = PMPC.split_large_qp_vars(probs, Nc, z)

  @time X, U = PMPC.admm_solve(
    x0,
    f,
    fx,
    fu,
    X_prev,
    U_prev,
    Q,
    R,
    X_ref,
    U_ref;
    rho=50.0,
    max_it=100,
    lu=lu,
    uu=uu,
  )
  @time X, U = PMPC.admm_solve(
    x0,
    f,
    fx,
    fu,
    X_prev,
    U_prev,
    Q,
    R,
    X_ref,
    U_ref;
    rho=50.0,
    max_it=100,
    lu=lu,
    uu=uu,
  )
  @time X, U = PMPC.admm_solve(
    x0,
    f,
    fx,
    fu,
    X_prev,
    U_prev,
    Q,
    R,
    X_ref,
    U_ref;
    rho=50.0,
    max_it=100,
    lu=lu,
    uu=uu,
  )

  z, _ = PMPC.solve!(PMPC.OSQPSolver(probs[1]))
  X2, U2 = reshape(z[(N * udim + 1):end], xdim, N, 1), reshape(z[1:(N * udim)], udim, N, 1)

  #X, U = X2, U2
  for i in 1:min(M, 10)
    figure(i)
    clf()
    for r in 1:xdim
      plot([x0[r, i]; X[r, :, i]], label="x" * string(r))
    end
    legend()
  end

  figure(M + 1)
  clf()
  for i in 1:min(M, 10)
    for r in 1:udim
      plot(U[r, :, i], label="u" * string(r))
    end
  end
  legend()

  1
end

function test_feedback()
  x0 = [5.0; 5.0; 5.0; 5.0]
  M, N, xdim, udim = 100, 30, 4, 2
  f = zeros(xdim, N, M)
  rho_res_x, rho_res_u = 0.0, 0.0
  fx, fu = zeros(xdim, xdim, N, M), zeros(xdim, udim, N, M)
  rng = range(0.1, stop=0.9, length=10)
  @views for i in 1:M
    r1, r2 = rand(rng), rand(rng)
    @views for j in 1:N
      fx[:, :, j, i] = Float64[1 r1 0 0; 0 1 0 0; 0 0 1 r2; 0 0 0 1]
      fu[:, :, j, i] = Float64[0 0; 1 0; 0 0; 0 1]
      f[:, j, i] = j == 1 ? fx[:, :, j, i] * x0 : zeros(xdim)
    end
  end
  X_prev, U_prev = zeros(xdim, N, M), zeros(udim, N, M)

  Q, R = zeros(xdim, xdim, N, M), zeros(udim, udim, N, M)
  @views for i in 1:M
    @views for j in 1:N
      Q[:, :, j, i] = diagm(0 => ones(xdim))
      R[:, :, j, i] = diagm(0 => ones(udim))
    end
  end
  X_ref, U_ref = zeros(xdim, N, M), zeros(udim, N, M)
  x0 = repeat(x0, 1, M)

  probs = map(_ -> PMPC.OCProb{Float64}(), 1:M)
  lu, uu = -ones(udim, N, M), ones(udim, N, M)
  @views for i in 1:M
    probs[i].rho_res_x, probs[i].rho_res_u = rho_res_x, rho_res_u
    PMPC.set_dyn!(
      probs[i],
      x0[:, i],
      f[:, :, i],
      fx[:, :, :, i],
      fu[:, :, :, i],
      X_prev[:, :, i],
      U_prev[:, :, i],
    )
    PMPC.set_cost!(probs[i], Q[:, :, :, i], R[:, :, :, i], X_ref[:, :, i], U_ref[:, :, i])
    PMPC.set_ubounds!(probs[i], lu[:, :, i], uu[:, :, i])
  end

  prob = probs[1]
  P, q, _ = PMPC.qp_repr_Pq(prob)
  A, b = PMPC.qp_repr_Ab(prob)

  K = [P A'; A 0.0*I] + 1e-9 * I
  Dx0g = vcat(zeros(length(q), xdim), fx[:, :, 1, 1], zeros((N - 1) * xdim, xdim))

  function ldlt_factor(A)
    F = ldlt(A)
    p = SuiteSparse.CHOLMOD.get_perm(F)
    pinv = sortperm(p)
    F = sparse(SuiteSparse.CHOLMOD.Sparse(F))
    L, D = SuiteSparse.CHOLMOD.getLd!(F)
    return L, D, p, pinv
  end
  function ldlt_Lsolve!(L, x)
    n, Lp, Li, Lx = length(x), L.colptr, L.rowval, L.nzval
    for i in 1:n
      for j in (Lp[i] + 1):(Lp[i + 1] - 1) # has a diagonal of all ones
        x[Li[j]] -= Lx[j] * x[i]
      end
    end
    return
  end
  function ldlt_Ltsolve!(L, x)
    n, Lp, Li, Lx = length(x), L.colptr, L.rowval, L.nzval
    for i in n:-1:1
      for j in (Lp[i] + 1):(Lp[i + 1] - 1) # has a diagonal of all ones
        x[i] -= Lx[j] * x[Li[j]]
      end
    end
    return
  end
  function ldlt_solve!(L, D, p, pinv, b)
    b .= b[p]
    ldlt_Lsolve!(L, b)
    b ./= D
    ldlt_Ltsolve!(L, b)
    b .= b[pinv]
  end
  function rankoneupdate!(L, D, p, pinv, alf, s)
    n, Lp, Li, Lx = size(L, 1), L.colptr, L.rowval, L.nzval
    s = s[p]
    w = L * s
    alfj = alf
    for j in 1:n
      wj = w[j]
      pj = wj
      dpj = D[j] + alfj * pj^2
      betj = pj * alfj / dpj
      alfj = D[j] * alfj / dpj
      D[j] = dpj
      for r in (Lp[j] + 1):(Lp[j] - 1)
        wj = wj - pj * Lx[r]
        Lx[r] = Lx[r] + betj * wj
      end
      display(wj)
    end
    return
  end

  @time L, D, p, pinv = ldlt_factor(K)
  @time L, D, p, pinv = ldlt_factor(K)
  @time L, D, p, pinv = ldlt_factor(K)
  @assert norm(L - LowerTriangular(L)) < 1e-7
  y = randn(size(L, 1))
  x = copy(y)
  ldlt_solve!(L, D, p, pinv, x)
  display(norm(K - (L * spdiagm(0 => D) * L')[pinv, pinv]))
  display(norm(K * x - y))

  s = randn(size(L, 1))
  rankoneupdate!(L, D, p, pinv, 1.0, s)
  display(maximum((K + s * s') - (L * spdiagm(0 => D) * L')[pinv, pinv]))

  solver = PMPC.iLQRSolver(prob)
  X, U, L, l, obj = PMPC.solve!(solver)
  #display(L)

  return
end
# testing scripts ##############################################################
