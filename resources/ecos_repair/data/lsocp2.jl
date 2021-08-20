using LinearAlgebra, SparseArrays
#using PyPlot
using PMPC

function main()
  println("\nRunning main\n")
  x0 = [5.0; 5.0; 5.0; 5.0]
  #M, N, xdim, udim = 5, 20, 4, 2
  M, N, xdim, udim = 10, 20, 4, 2

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

  println("\nDone with dynamics, running ECOS\n")

  PMPC.@ptime X, U = PMPC.lsocp_solve(x0, f, fx, fu, X_prev, U_prev, Q, R,
                                      X_ref, U_ref; lu=lu, uu=uu) 4
  PMPC.@ptime X2, U2 = PMPC.lqp_solve(x0, f, fx, fu, X_prev, U_prev, Q, R,
                                      X_ref, U_ref; lu=lu, uu=uu) 4

  #for i in 1:min(M, 3)
  #  figure(i); clf()
  #  for r in 1:xdim
  #    plot([x0[r, i]; X[r, :, i]], label="x" * string(r))
  #  end
  #  legend()
  #end

  #figure(M + 1); clf()
  #for i in 1:min(M, 100)
  #  for r in 1:udim
  #    plot(U[r, :, i], label="u" * string(r))
  #  end
  #end
  #legend()

  #for i in 1:min(M, 3)
  #  figure(1000 + i); clf()
  #  for r in 1:xdim
  #    plot([x0[r, i]; X2[r, :, i]], label="x" * string(r))
  #  end
  #  legend()
  #end

  #figure(2000 + M + 1); clf()
  #for i in 1:min(M, 100)
  #  for r in 1:udim
  #    plot(U2[r, :, i], label="u" * string(r))
  #  end
  #end
  #legend()

  return nothing
end
main()
