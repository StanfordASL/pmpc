using Revise
using PMPC
using BenchmarkTools

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

lu, uu = -0.2 * ones(udim, N, M), 0.2 * ones(udim, N, M)
lx, ux = -100 * ones(xdim, N, M), 100 * ones(xdim, N, M)

args = x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref
settings = Dict{Symbol, Any}()

settings = merge(settings, Dict(:lu => lu, :uu => uu, :lx => lx, :ux => ux))
settings[:return_solver] = true

solver_name = "osqp"

X_sol, U_sol, data = lqp_solve(args...; solver_name=solver_name, settings...)

settings[:return_solver_id] = true
_, _, data = lqp_solve(args...; solver_name=solver_name, settings...)
@info("showing solver id -> $(data[:solver_id])")

#@btime lqp_solve(args...; solver_name = "ecos", settings...)
delete!(settings, :solver_id)
@btime lqp_solve(args...; solver_name=solver_name, settings...)

return
