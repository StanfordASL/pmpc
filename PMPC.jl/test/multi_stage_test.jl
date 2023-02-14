#using Revise
#
if !isdefined(Main, :PMPC)
  #using Pkg
  #Pkg.develop(; path=joinpath(@__DIR__, ".."))
  #using PMPC
end
include(joinpath(@__DIR__, "..", "src", "PMPC.jl"))

using LinearAlgebra, SparseArrays, Printf, Random, SuiteSparse
using BenchmarkTools, Debugger, PyPlot
Random.seed!(2020)

function test_multi_stage()
  N, xdim, udim = 20, 4, 2
  Nr = 4
  Mb = 3
  M = Mb^Nr

  rho_res_x, rho_res_u = 0.0, 0.0
  x0 = [5.0; 5.0; 5.0; 5.0]
  X_prev, U_prev = zeros(xdim, N, M), zeros(udim, N, M)
  f = zeros(xdim, N, M)
  fx, fu = zeros(xdim, xdim, N, M), zeros(xdim, udim, N, M)
  rng = range(0.1, stop=0.9, length=10)
  @views for i in 1:M
    r1, r2 = rand(rng), rand(rng)
    @views for j in 1:N
      fx[:, :, j, i] = Float64[1 r1 0 0; 0 1 0 0; 0 0 1 r2; 0 0 0 1]
      #fx[:, :, j, i] = randn(xdim, xdim)
      fu[:, :, j, i] = Float64[0 0; 1 0; 0 0; 0 1]
      #fu[:, :, j, i] = randn(xdim, udim)
      f[:, j, i] = j == 1 ? fx[:, :, j, i] * x0 : zeros(xdim)
    end
  end
  x0 = repeat(x0, 1, M)

  X_ref, U_ref = zeros(xdim, N, M), zeros(udim, N, M)
  Q, R = zeros(xdim, xdim, N, M), zeros(udim, udim, N, M)
  @views for i in 1:M
    @views for j in 1:N
      Q[:, :, j, i] = diagm(0 => ones(xdim))
      R[:, :, j, i] = diagm(0 => ones(udim))
    end
  end

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
    #PMPC.set_ubounds!(probs[i], lu[:, :, i], uu[:, :, i])
    #PMPC.set_ctrl_slew!(probs[i], slew_rho=100.0)
  end
  global P, q = PMPC.multi_stage_repr_Pq(probs, Nr, Mb)
  @assert size(P, 1) == size(P, 2)
  println(udim * ((N - Nr) * M + div(Mb^Nr - 1, Mb - 1)) + N * M * xdim)
  println(size(P, 1))
  @assert size(P, 1) == udim * ((N - Nr) * M + div(Mb^Nr - 1, Mb - 1)) + N * M * xdim
  global A, b = PMPC.multi_stage_repr_Ab(probs, Nr, Mb)
  @assert size(A, 2) == size(P, 1)
  @assert size(A, 1) == size(b, 1)

  close("all")
  figure()
  imshow(collect(abs.(P)) .> 0.0)

  figure()
  #imshow(collect(abs.(A))[:, 1:udim * (Mb^Nr - 1)] .> 0.0)
  imshow(collect(abs.(A)) .> 0.0)

  solver = PMPC.OSQPSolver()
  solver.P, solver.q, solver.A, solver.b = P, q, A, b
  z, _ = PMPC.solve_qp!(solver)
  global X, U = PMPC.split_multi_stage_vars(probs, Nr, Mb, z)

  figure()
  for i in 1:M
    for r in 1:xdim
      plot(X[r, :, i]; color=@sprintf("C%d", r))
    end
  end
  title("X")
  tight_layout()

  figure()
  for i in 1:M
    for r in 1:udim
      plot(U[r, :, i]; color=@sprintf("C%d", r))
    end
  end
  title("U")
  tight_layout()

  ret = nothing
  return ret
end
ret = test_multi_stage()
return
# testing scripts ##############################################################
