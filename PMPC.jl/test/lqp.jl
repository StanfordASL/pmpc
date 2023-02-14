using PMPC

M, N, xdim, udim = 1, 10, 4, 2
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


sol = lqp_solve(x0, f, fx, fu, X_prev, U_prev, Q, R, X_ref, U_ref)
