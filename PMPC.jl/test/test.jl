using LinearAlgebra, SparseArrays, Printf, Random, SuiteSparse
using BenchmarkTools
using PMPC
Random.seed!(2020)

PLOT_FLAG = false

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

      if PLOT_FLAG || i == 1
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
    X, U = PMPC.solve!(solver)

    X = [x0 X]

    if PLOT_FLAG
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

  if PLOT_FLAG
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

  if PLOT_FLAG
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
  if PLOT_FLAG
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
  end

  #=
  fbs2 = PMPC.FBSolver(probs[1])
  PMPC.solve!(fbs2)
  =#
  return
end

function test_lqp()
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
  P, q = PMPC.lqp_repr_Pq(probs, Nc)
  A, b = PMPC.lqp_repr_Ab(probs, Nc)
  A_, b_ = PMPC.qp_repr_Ab(probs[1])
  G, l, u = PMPC.lqp_repr_Gla(probs, Nc)

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
  if PLOT_FLAG
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
  end

  return
end

function test_sens()
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

  K = [P A'; A 0.0*I]
  display(P - P')
  sol = -K \ [-q; -b]
  U = reshape(sol[1:(N * udim)], udim, :)
  X = reshape(sol[(N * udim + 1):(N * (udim + xdim))], xdim, :)
  lam = sol[(N * (udim + xdim) + 1):end]

  Dx0g = vcat(zeros(length(q), xdim), fx[:, :, 1, 1], zeros((N - 1) * xdim, xdim))
  K_ = K + 1e-9 * I
  F = ldlt(K + 1e-9 * I)

  j = 1
  #=
  E = hcat(spzeros(xdim, N * udim), spzeros(xdim, xdim * (J - 2)),
           sparse(I, xdim, xdim), spzeros(xdim, xdim * (N - j + 1)),
           spzeros(xdim, xdim * N))
  y2 = zeros(xdim, xdim)
  Dxg = vcat(spzeros(N * udim, xdim), spzeros(N * xdim, xdim),
             spzeros(xdim * (j - 1), xdim),
             fx[:, :, i, 1], spzeros(xdim * (N - j), xdim))
  display(E)
  M = (E * (F \ sparse(E')))
  soly = -(M \ (y2 - E * (F \ -Dxg)))
  solx = F \ (-Dxg - E' * soly)
  L2 = PMPC.catb(map(i -> view(solx, udim*(j-1)+1:udim*j, :), 1:N)...)
  display(L2)
  println("-------------------------------------------------------------------")
  =#

  #=
  @time L, fp = PMPC.qp_ctrl_sens(prob, j)
  @time L, fp = PMPC.qp_ctrl_sens(prob, j)
  @time L, fp = PMPC.qp_ctrl_sens(prob, j)
  @time L, fp = PMPC.qp_ctrl_sens(prob, j)
  println()
  @time L = PMPC.qp_ctrl_sens(fp, j)
  @time L = PMPC.qp_ctrl_sens(fp, j)
  @time L = PMPC.qp_ctrl_sens(fp, j)
  display(L)
  println("-------------------------------------------------------------------")

  #=
  figure(1); clf()
  for r in 1:udim
  plot(U[r, :])
  end

  figure(2); clf()
  for r in 1:xdim
  plot([x0[r]; X[r, :]])
  end
  =#

  solver = PMPC.iLQRSolver(prob) 
  X, U, L, l, obj = PMPC.solve!(solver)
  display(L)
  println("-------------------------------------------------------------------")
  =#
  Nc = -1
  @time L, fp = PMPC.lqp_ctrl_sens(probs, Nc, 1, j)
  @time L, fp = PMPC.lqp_ctrl_sens(probs, Nc, 1, j)
  @time L, fp = PMPC.lqp_ctrl_sens(probs, Nc, 1, j)
  println()
  @time L = PMPC.lqp_ctrl_sens(fp, 1, j)
  @time L = PMPC.lqp_ctrl_sens(fp, 1, j)
  @time L = PMPC.lqp_ctrl_sens(fp, 1, j)

  return
end

function test_constr()
  x0 = [5.0; 5.0; 5.0; 5.0]
  M, N, xdim, udim = 100, 60, 4, 2
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
    PMPC.set_ctrl_slew!(probs[i], slew_rho=100.0, slew_um1=zeros(udim))
  end

  #@btime P, q = PMPC.lqp_repr_Pq($probs, 1)
  #@btime A, b = PMPC.lqp_repr_Ab($probs, 1)
  @time P, q = PMPC.lqp_repr_Pq(probs, 1)
  @time P, q = PMPC.lqp_repr_Pq(probs, 1)
  @time P, q = PMPC.lqp_repr_Pq(probs, 1)
  println()
  @time A, b = PMPC.lqp_repr_Ab(probs, 1)
  @time A, b = PMPC.lqp_repr_Ab(probs, 1)
  @time A, b = PMPC.lqp_repr_Ab(probs, 1)
  #@btime K = [$P $A'; $A 0.0 * I]
  #@btime K = hcat(vcat($P, $A), vcat($A', spzeros(size($A, 1), size($A, 1)))) + 1e-9 * I
  K = hcat(vcat(P, A), vcat(A', spzeros(size(A, 1), size(A, 1)))) + 1e-9 * I
  #@btime F = ldlt($K)
  F = ldlt(K)
  #P.nzval .= 1.0
  #display(collect(P))
  #display(q)
  return
end

function test_lqp_sens()
  x0 = [5.0; 5.0; 5.0; 5.0]
  M, N, xdim, udim = 100, 5, 4, 2
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
    PMPC.set_ctrl_slew!(probs[i], slew_rho=100.0)
  end
  @time ret = PMPC.lqp_solve(probs, Nc=-1)
  @time ret = PMPC.lqp_solve(probs, Nc=-1)
  @time ret = PMPC.lqp_solve(probs, Nc=-1)
  return

  j = 2
  Nc = -1
  @time L, fp = PMPC.lqp_ctrl_sens(probs, Nc, 1, j)
  @time L, fp = PMPC.lqp_ctrl_sens(probs, Nc, 1, j)
  @time L, fp = PMPC.lqp_ctrl_sens(probs, Nc, 1, j)
  @time L, fp = PMPC.lqp_ctrl_sens(probs, Nc, 1, j)
  println()
  Ls = Array{Float64, 2}[]
  for i in 1:M
    push!(Ls, PMPC.lqp_ctrl_sens(fp, i, j))
  end
  Ls = sum(Ls; dims=3)
  L = sum(Ls)
  #@time L = PMPC.lqp_ctrl_sens(fp, 1, j)
  #@time L = PMPC.lqp_ctrl_sens(fp, 1, j)
  #@time L = PMPC.lqp_ctrl_sens(fp, 1, j)
  #@time L = PMPC.lqp_ctrl_sens(fp, 1, j)
  display(L)

  solver = PMPC.iLQRSolver(probs[1])
  X, U, L, l, obj = PMPC.solve!(solver)
  display(L[:, :, j])
  println("-------------------------------------------------------------------")

  return
end

function test_lsocp()
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

  lu, uu = -1 * ones(udim, N, M), 1 * ones(udim, N, M)
  PMPC.@ptime X, U =
    PMPC.lsocp_solve(x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref; lu=lu, uu=uu) 4
  PMPC.@ptime X2, U2 =
    PMPC.lqp_solve(x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref; lu=lu, uu=uu) 4

  if PLOT_FLAG
    for i in 1:min(M, 3)
      figure(i)
      clf()
      for r in 1:xdim
        plot([x0[r, i]; X[r, :, i]], label="x" * string(r))
      end
      legend()
    end

    figure(M + 1)
    clf()
    for i in 1:min(M, 100)
      for r in 1:udim
        plot(U[r, :, i], label="u" * string(r))
      end
    end
    legend()

    for i in 1:min(M, 3)
      figure(1000 + i)
      clf()
      for r in 1:xdim
        plot([x0[r, i]; X2[r, :, i]], label="x" * string(r))
      end
      legend()
    end

    figure(2000 + M + 1)
    clf()
    for i in 1:min(M, 100)
      for r in 1:udim
        plot(U2[r, :, i], label="u" * string(r))
      end
    end
    legend()
  end

  return nothing
end
# testing scripts ##############################################################

tests = [
  test_qp_ilqr,
  test_cons,
  test_sqp,
  test_large_sqp,
  test_lqp,
  test_sens,
  test_constr,
  test_lqp_sens,
  test_lsocp,
]

for test in tests
  try
    test()
  catch
  end
end
