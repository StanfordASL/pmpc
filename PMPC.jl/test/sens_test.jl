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

function fit_line(x, y)
  return hcat(0.0 * x .+ 1, x) \ y
end

function test_lqp_sens()
  x0 = [5.0; 5.0; 5.0; 5.0]
  M, N, xdim, udim = 1, 20, 4, 2
  Nc = N
  f = zeros(xdim, N, M)
  rho_res_x, rho_res_u = 0.0, 0.0
  fx, fu = zeros(xdim, xdim, N, M), zeros(xdim, udim, N, M)
  rng = range(0.1, stop=0.9, length=10)
  @views for i in 1:M
    r1, r2 = rand(rng), rand(rng)
    @views for j in 1:N
      #fx[:, :, j, i] = Float64[1 r1 0 0; 0 1 0 0; 0 0 1 r2; 0 0 0 1]
      fx[:, :, j, i] = randn(xdim, xdim)
      #fu[:, :, j, i] = Float64[0 0; 1 0; 0 0; 0 1]
      fu[:, :, j, i] = randn(xdim, udim)
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
    #PMPC.set_ubounds!(probs[i], lu[:, :, i], uu[:, :, i])
    #PMPC.set_ctrl_slew!(probs[i], slew_rho=100.0)
  end

  #@run PMPC.disturbance_sens(probs)
  dsp = PMPC.disturbance_sens(probs)

  close("all")

  global grad = PMPC.disturbance_sensitivity(dsp; deq_dp=Array(1.0 * I, N * M * xdim, N * M * xdim))
  probs_org = deepcopy(probs)
  for i in 1:3
    ds = range(0; stop=1.0, length=100)
    r = randn(xdim)
    Us = []
    for d in ds
      probs = deepcopy(probs_org)
      probs[1].f[:, i] .+= d * r
      X, U, _ = PMPC.lqp_solve(probs)
      push!(Us, U)
    end
    probs = deepcopy(probs_org)
    grad = PMPC.disturbance_sensitivity(dsp; deq_dp=Array(1.0 * I, N * M * xdim, N * M * xdim))[
      1:udim,
      (xdim * (i - 1) + 1):(xdim * i),
    ]
    display(grad)
    display(grad * r)
    println()

    figure()
    plot(ds, [U[1, 1, 1] for U in Us])
    plot(ds, [U[2, 1, 1] for U in Us])
    th1 = fit_line(ds, [U[1, 1, 1] for U in Us])
    th2 = fit_line(ds, [U[2, 1, 1] for U in Us])
    display(th1[2])
    display(th2[2])

    println()
  end


  #figure()
  #imshow(abs.(grad[1:Nc * udim, :]))
  #colorbar()

  return dsp
end
dsp = test_lqp_sens();
return
# testing scripts ##############################################################
