using SparseArrays, LinearAlgebra
using PMPC

add_dim(x, M) = repeat(x; inner=vcat(fill(1, ndims(x)), [M]))

function precompile_c_interface()
  M = 1
  N, xdim, udim = 30, 2, 1
  Nc = 3
  x0 = [5.0; 5.0]
  fx = PMPC.catb(map(_ -> [1.0 0.1; 0.0 1.0], 1:N)...)
  fu = PMPC.catb(map(_ -> [0.0; 1.0][:, :], 1:N)...)
  f = PMPC.catb(map(i -> i == 1 ? fx[:, :, i] * x0 : zeros(2), 1:N)...)
  #X_prev, U_prev = randn(xdim, N), randn(udim, N)
  X_prev, U_prev = zeros(xdim, N), zeros(udim, N)
  #X_ref, U_ref = randn(xdim, N), randn(udim, N)
  X_ref, U_ref = zeros(xdim, N), zeros(udim, N)
  Q = PMPC.catb(map(i -> diagm(0 => (i == N ? [1e0, 1e0] : [1e0, 1.0])), 1:N)...)
  #q = PMPC.catb(map(_ -> zeros(xdim), 1:N)...)
  R = PMPC.catb(map(_ -> diagm(0 => [1e0]), 1:N)...)
  #r = PMPC.catb(map(_ -> zeros(udim), 1:N)...)
  u_limit = 1.0
  lu, uu = -u_limit * ones(udim, N), u_limit * ones(udim, N)
  x_limit = 20.0
  lx, ux = -x_limit * ones(xdim, N), x_limit * ones(xdim, N)


  x0 = add_dim(x0, M)
  f = add_dim(f, M)
  fx = add_dim(fx, M)
  fu = add_dim(fu, M)
  X_prev = add_dim(X_prev, M)
  U_prev = add_dim(U_prev, M)
  Q = add_dim(Q, M)
  R = add_dim(R, M)
  X_ref = add_dim(X_ref, M)
  U_ref = add_dim(U_ref, M)
  lu, uu = add_dim(lu, M), add_dim(uu, M)
  lx, ux = add_dim(lx, M), add_dim(ux, M)

  f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref =
    [collect(z) for z in [f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref]]
  lx, ux, lu, uu = [collect(z) for z in [lx, ux, lu, uu]]

  reg_x, reg_u = 1e0, 1e-1

  slew_reg = 1.0 * ones(M)
  slew_reg0 = 0.0 * ones(M)
  slew_um1 = zeros(udim, M)

  smooth_alpha = 1e0

  X_out, U_out = zeros(xdim, N, M), zeros(udim, N, M)

  for verbose in [0, 1]
    for u_cstr_present in [false, true]
      for x_cstr_present in [false, true]
        for slew0_present in [false, true]
          for slew_present in [false, true]
            println(
              "\nverbose = $(verbose) u_cstr = $(u_cstr_present) x_cstr = $(x_cstr_present) " *
              "slew0 = $(slew0_present) slew = $(slew_present)\n",
            )
            flush(stdout)

            c_lqp_solve(
              pointer(X_out),
              pointer(U_out),
              Csize_t(xdim),
              Csize_t(udim),
              Csize_t(N),
              Csize_t(M),
              Clonglong(Nc),
              pointer(x0),
              pointer(f),
              pointer(fx),
              pointer(fu),
              pointer(X_prev),
              pointer(X_prev),
              pointer(Q),
              pointer(R),
              pointer(X_ref),
              pointer(U_ref),
              pointer(x_cstr_present ? lx : NaN * lx),
              pointer(x_cstr_present ? ux : NaN * ux),
              pointer(u_cstr_present ? lu : NaN * lu),
              pointer(u_cstr_present ? uu : NaN * uu),
              Cdouble(reg_x),
              Cdouble(reg_u),
              pointer(slew_present ? slew_reg : NaN * slew_reg),
              pointer(slew0_present ? slew_reg0 : NaN * slew_reg0),
              pointer(slew0_present ? slew_um1 : NaN * slew_um1),
              Clonglong(verbose),
            )

            display(X_out[:, :, 1])
            display(U_out[:, :, 1])

            c_lcone_solve(
              pointer(X_out),
              pointer(U_out),
              Csize_t(xdim),
              Csize_t(udim),
              Csize_t(N),
              Csize_t(M),
              Clonglong(Nc),
              pointer(x0),
              pointer(f),
              pointer(fx),
              pointer(fu),
              pointer(X_prev),
              pointer(X_prev),
              pointer(Q),
              pointer(R),
              pointer(X_ref),
              pointer(U_ref),
              pointer(x_cstr_present ? lx : NaN * lx),
              pointer(x_cstr_present ? ux : NaN * ux),
              pointer(u_cstr_present ? lu : NaN * lu),
              pointer(u_cstr_present ? uu : NaN * uu),
              Cdouble(reg_x),
              Cdouble(reg_u),
              pointer(slew_present ? slew_reg : NaN * slew_reg),
              pointer(slew0_present ? slew_reg0 : NaN * slew_reg0),
              pointer(slew0_present ? slew_um1 : NaN * slew_um1),
              Clonglong(verbose),
              Cdouble(smooth_alpha),
            )

            display(X_out[:, :, 1])
            display(U_out[:, :, 1])
          end
        end
      end
    end
  end
end

precompile_c_interface()
