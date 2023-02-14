using PMPC
using Test

@testset "PMPC" begin
  # generating random data
  M, N, xdim, udim = 3, 11, 4, 2
  x0 = randn(xdim, M)
  f = randn(xdim, N, M)
  fx = randn(xdim, xdim, N, M)
  fu = randn(xdim, udim, N, M)
  X_prev = randn(xdim, N, M)
  U_prev = randn(udim, N, M)
  Q = randn(xdim, xdim, N, M)
  R = randn(udim, udim, N, M)
  for i in 1:N
    for j in 1:M
      Q[:, :, i, j] = Q[:, :, i, j]' * Q[:, :, i, j]
      R[:, :, i, j] = R[:, :, i, j]' * R[:, :, i, j]
    end
  end
  X_ref = randn(xdim, N, M)
  U_ref = randn(udim, N, M)

  lu = -0.2 * ones(udim, N, M)
  uu = 0.2 * ones(udim, N, M)
  lx = -100 * ones(xdim, N, M)
  ux = 100 * ones(xdim, N, M)

  args = x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref
  settings = Dict{Symbol, Any}()

  @testset "main solver interface" begin
    X_sol, U_sol, data = lqp_solve(args...; solver_name="osqp")
    @test all(.!isnan.(X_sol))
    @test all(.!isnan.(U_sol))

    # with constraints
    settings = merge(settings, Dict(:lu => lu, :uu => uu, :lx => lx, :ux => ux))
    @test (lqp_solve(args...; solver_name="osqp", settings...); true)
    @test (lqp_solve(args...; solver_name="ecos", settings...); true)
    @test (lqp_solve(args...; solver_name="mosek", settings...); true)
  end

  @testset "types and conversion utilities" begin
    probs = make_probs(args...; settings...)
    @test (PMPC.JuMPSolver(); true)
  end

  @testset "JuMP Solver" begin
    settings = merge(settings, Dict(:lu => lu, :uu => uu, :lx => lx, :ux => ux))
    # smooth constraints
    settings[:smooth_cstr] = ""
    @test (lqp_solve(args...; solver_name="osqp", settings...); true)
    @test (lqp_solve(args...; solver_name="ecos", settings...); true)
    settings[:smooth_cstr] = "logbarrier"
    @test (lqp_solve(args...; solver_name="ecos", settings...); true)
    @test (lqp_solve(args...; solver_name="mosek", settings...); true)
  end

  @testset "saving solvers" begin
    settings = merge(settings, Dict(:lu => lu, :uu => uu, :lx => lx, :ux => ux))
    settings[:save_solver] = true
    settings[:smooth_cstr] = "logbarrier"
    settings[:return_solver] = true
    _, _, data = lqp_solve(args...; solver_name="ecos", settings...)
    @test haskey(data, :solver)

    settings[:return_solver_id] = true
    _, _, data = lqp_solve(args...; solver_name="ecos", settings...)
    @test haskey(data, :solver_id)
    _, _, data2 = lqp_solve(args...; solver_name="ecos", settings...)
    @test data[:solver_id] != data2[:solver_id]

    settings[:solver_id] = data[:solver_id]
    _, _, data3 = lqp_solve(args...; solver_name="ecos", settings...)
    @test data[:solver_id] == data3[:solver_id]
  end

end
