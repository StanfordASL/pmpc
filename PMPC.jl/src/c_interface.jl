function unwrap_args(
  xdim::Csize_t,
  udim::Csize_t,
  N::Csize_t,
  M::Csize_t,
  Nc::Clonglong,
  x0::Ptr{Cdouble},
  f::Ptr{Cdouble},
  fx::Ptr{Cdouble},
  fu::Ptr{Cdouble},
  X_prev::Ptr{Cdouble},
  U_prev::Ptr{Cdouble},
  Q::Ptr{Cdouble},
  R::Ptr{Cdouble},
  X_ref::Ptr{Cdouble},
  U_ref::Ptr{Cdouble},
  lx::Ptr{Cdouble},
  ux::Ptr{Cdouble},
  lu::Ptr{Cdouble},
  uu::Ptr{Cdouble},
  reg_x::Cdouble,
  reg_u::Cdouble,
  slew_reg::Ptr{Cdouble},
  slew_reg0::Ptr{Cdouble},
  slew_um1::Ptr{Cdouble},
  verbose::Clonglong,
)
  x0 = unsafe_wrap(Array, x0, (xdim, M))
  f = unsafe_wrap(Array, f, (xdim, N, M))
  fx = unsafe_wrap(Array, fx, (xdim, xdim, N, M))
  fu = unsafe_wrap(Array, fu, (xdim, udim, N, M))
  X_prev = unsafe_wrap(Array, X_prev, (xdim, N, M))
  U_prev = unsafe_wrap(Array, U_prev, (udim, N, M))
  Q = unsafe_wrap(Array, Q, (xdim, xdim, N, M))
  R = unsafe_wrap(Array, R, (udim, udim, N, M))
  X_ref = unsafe_wrap(Array, X_ref, (xdim, N, M))
  U_ref = unsafe_wrap(Array, U_ref, (udim, N, M))

  lx = unsafe_wrap(Array, lx, (xdim, N, M))
  ux = unsafe_wrap(Array, ux, (xdim, N, M))
  lu = unsafe_wrap(Array, lu, (udim, N, M))
  uu = unsafe_wrap(Array, uu, (udim, N, M))

  slew_reg = unsafe_wrap(Array, slew_reg, M)
  slew_reg0 = unsafe_wrap(Array, slew_reg0, M)
  slew_um1 = unsafe_wrap(Array, slew_um1, (udim, M))

  args = (x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref)

  settings = Dict{Symbol,Any}()

  settings[:reg_x] = reg_x
  settings[:reg_u] = reg_u
  settings[:Nc] = Nc

  if !(any(isnan.(lx)) || any(isnan.(ux)))
    settings[:lx] = lx
    settings[:ux] = ux
  end
  if !(any(isnan.(lu)) || any(isnan.(uu)))
    settings[:lu] = lu
    settings[:uu] = uu
  end
  if !any(isnan.(slew_reg))
    settings[:slew_reg] = slew_reg
  end
  if !(any(isnan.(slew_reg0)) || any(isnan.(slew_um1)))
    settings[:slew_um1] = slew_um1
    settings[:slew_reg0] = slew_reg0
  end
  settings[:verbose] = Bool(verbose)

  return args, settings
end


Base.@ccallable function c_lqp_solve(
  X_out::Ptr{Cdouble},
  U_out::Ptr{Cdouble},
  xdim::Csize_t,
  udim::Csize_t,
  N::Csize_t,
  M::Csize_t,
  Nc::Clonglong,
  x0::Ptr{Cdouble},
  f::Ptr{Cdouble},
  fx::Ptr{Cdouble},
  fu::Ptr{Cdouble},
  X_prev::Ptr{Cdouble},
  U_prev::Ptr{Cdouble},
  Q::Ptr{Cdouble},
  R::Ptr{Cdouble},
  X_ref::Ptr{Cdouble},
  U_ref::Ptr{Cdouble},
  lx::Ptr{Cdouble},
  ux::Ptr{Cdouble},
  lu::Ptr{Cdouble},
  uu::Ptr{Cdouble},
  reg_x::Cdouble,
  reg_u::Cdouble,
  slew_reg::Ptr{Cdouble},
  slew_reg0::Ptr{Cdouble},
  slew_um1::Ptr{Cdouble},
  verbose::Clonglong,
)::Cvoid
  (x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref), settings = unwrap_args(
    xdim,
    udim,
    N,
    M,
    Nc,
    x0,
    f,
    fx,
    fu,
    X_prev,
    U_prev,
    Q,
    R,
    X_ref,
    U_ref,
    lx,
    ux,
    lu,
    uu,
    reg_x,
    reg_u,
    slew_reg,
    slew_reg0,
    slew_um1,
    verbose,
  )

  X, U, _ = lqp_solve(
    make_probs(x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref; settings...);
    settings...,
  )
  unsafe_copyto!(X_out, pointer(X), xdim * N * M)
  unsafe_copyto!(U_out, pointer(U), udim * N * M)
  return
end

####################################################################################################


Base.@ccallable function c_lcone_solve(
  X_out::Ptr{Cdouble},
  U_out::Ptr{Cdouble},
  xdim::Csize_t,
  udim::Csize_t,
  N::Csize_t,
  M::Csize_t,
  Nc::Clonglong,
  x0::Ptr{Cdouble},
  f::Ptr{Cdouble},
  fx::Ptr{Cdouble},
  fu::Ptr{Cdouble},
  X_prev::Ptr{Cdouble},
  U_prev::Ptr{Cdouble},
  Q::Ptr{Cdouble},
  R::Ptr{Cdouble},
  X_ref::Ptr{Cdouble},
  U_ref::Ptr{Cdouble},
  lx::Ptr{Cdouble},
  ux::Ptr{Cdouble},
  lu::Ptr{Cdouble},
  uu::Ptr{Cdouble},
  reg_x::Cdouble,
  reg_u::Cdouble,
  slew_reg::Ptr{Cdouble},
  slew_reg0::Ptr{Cdouble},
  slew_um1::Ptr{Cdouble},
  verbose::Clonglong,
  smooth_alpha::Cdouble,
  solver::Cstring,
)::Cvoid
  (x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref), settings = unwrap_args(
    xdim,
    udim,
    N,
    M,
    Nc,
    x0,
    f,
    fx,
    fu,
    X_prev,
    U_prev,
    Q,
    R,
    X_ref,
    U_ref,
    lx,
    ux,
    lu,
    uu,
    reg_x,
    reg_u,
    slew_reg,
    slew_reg0,
    slew_um1,
    verbose,
  )
  settings[:smooth_alpha] = smooth_alpha
  settings[:smooth_cstr] = "logbarrier"
  settings[:solver] = unsafe_string(solver)

  X, U, _ = lcone_solve(
    make_probs(x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref; settings...);
    settings...,
  )
  unsafe_copyto!(X_out, pointer(X), xdim * N * M)
  unsafe_copyto!(U_out, pointer(U), udim * N * M)
  return
end
