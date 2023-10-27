# low level QP routines ########################################################
function construct_Ab!(Ap, Ai, Ax, b, f, fx, fu, X_prev, U_prev)
  N, xdim, udim = size(fx, 3), size(fx, 2), size(fu, 2)
  #nnz = N * (xdim + xdim * xdim + xdim * udim)
  Ap[1] = 1
  for i in 1:(N * udim)
    inc = xdim
    Ap[i + 1] = Ap[i] + inc
  end
  for i in (N * udim + 1):(N * udim + (N - 1) * xdim)
    inc = 1 + xdim
    Ap[i + 1] = Ap[i] + inc
  end
  for i in (N * udim + (N - 1) * xdim + 1):(N * udim + N * xdim)
    inc = 1
    Ap[i + 1] = Ap[i] + inc
  end

  l = 0
  for i in 1:N
    for j in 1:udim
      for k in 1:xdim
        l += 1
        Ai[l] = (i - 1) * xdim + k
        Ax[l] = fu[k, j, i]
      end
    end
  end
  for i in 1:(N - 1)
    for j in 1:xdim
      l += 1
      Ai[l] = (i - 1) * xdim + j
      Ax[l] = -1.0
      for k in 1:xdim
        l += 1
        Ai[l] = (i - 1) * xdim + xdim + k
        Ax[l] = fx[k, j, i + 1]
      end
    end
  end
  for j in 1:xdim
    l += 1
    Ai[l] = (N - 1) * xdim + j
    Ax[l] = -1.0
  end
  @views begin
    b[1:xdim] = -f[:, 1] + fu[:, :, 1] * U_prev[:, 1]
    for i in 2:N
      b[(xdim * (i - 1) + 1):(xdim * i)] =
        (-f[:, i] + fx[:, :, i] * X_prev[:, i - 1] + fu[:, :, i] * U_prev[:, i])
    end
  end
  return
end
# low level QP routines ########################################################


# QP representations ###########################################################
function qp_repr_Pq(prob::OCProb{T}) where {T}
  xdim, udim, N = prob.xdim, prob.udim, prob.N
  elnum, n = N * udim^2 + N * xdim^2 + 2 * N * udim, N * (xdim + udim)
  Pp = zeros(Int, n + 1) ###########################################################################
  Pi, Px = zeros(Int, elnum), zeros(T, elnum)
  k, c = 0, 0
  Pp[1] = 1
  for j in 1:N
    @inbounds for t in 1:udim
      k_old = k
      slew_reg = prob.slew_reg
      if j > 1 && slew_reg != 0.0 # slew top half
        idx = udim * (j - 2) + t
        k += 1
        Pi[k] = idx
        Px[k] = -prob.slew_reg
      end
      @inbounds for r in 1:udim # core cost and residual
        val = prob.R[r, t, j]
        if r == t
          val += prob.reg_u
          if j == 1
            val += prob.slew_reg0 + prob.slew_reg
          elseif j == N
            val += prob.slew_reg
          else
            val += 2 * prob.slew_reg
          end
        end
        if val != 0.0
          k += 1
          Pi[k] = udim * (j - 1) + r
          Px[k] = val
        end
      end
      if j < N && slew_reg != 0.0 # slew bottom half
        k += 1
        Pi[k] = udim * j + t
        Px[k] = -prob.slew_reg
      end
      c += 1
      Pp[c + 1] = Pp[c] + (k - k_old)
    end
  end
  for j in 1:N
    @inbounds for t in 1:xdim
      k_old = k
      @inbounds for r in 1:xdim
        val = prob.Q[r, t, j] + (r == t ? prob.reg_x : 0.0)
        if val != 0.0
          k += 1
          Pi[k] = N * udim + xdim * (j - 1) + r
          Px[k] = val
        end
      end
      c += 1
      Pp[c + 1] = Pp[c] + (k - k_old)
    end
  end
  q = zeros(T, n) ##################################################################################
  q[1:udim] .+= -prob.slew_reg0 * prob.slew_um1
  for j in 1:N # for U
    sidx = udim * (j - 1)
    @inbounds for r in 1:udim
      val = -prob.reg_u * prob.U_prev[r, j]
      @inbounds for t in 1:udim
        val -= prob.R[r, t, j] * prob.U_ref[t, j]
      end
      q[sidx + r] += val
    end
  end
  for j in 1:N # for X
    sidx = N * udim + xdim * (j - 1)
    @inbounds for r in 1:xdim
      val = -prob.reg_x * prob.X_prev[r, j]
      @inbounds for t in 1:xdim
        val -= prob.Q[r, t, j] * prob.X_ref[t, j]
      end
      q[sidx + r] = val
    end
  end
  resid = 0.0 ######################################################################################
  for j in 1:N # for U
    @inbounds for r in 1:udim
      resid += 0.5 * prob.reg_u * prob.U_prev[r, j]^2
      val = 0
      @inbounds for t in 1:udim
        val += prob.R[r, t, j] * prob.U_ref[t, j]
      end
      resid += 0.5 * val * prob.U_ref[r, j]
    end
  end
  for j in 1:N # for X
    @inbounds for r in 1:xdim
      resid += 0.5 * prob.reg_x * prob.X_prev[r, j]^2
      val = 0
      @inbounds for t in 1:xdim
        val += prob.Q[r, t, j] * prob.X_ref[t, j]
      end
      resid += 0.5 * val * prob.X_ref[r, j]
    end
  end
  Pi, Px = Pi[1:k], Px[1:k]
  P = SparseMatrixCSC{T, Int}(n, n, Pp, Pi, Px)
  return P, q, resid
end

function qp_repr_Ab(prob::OCProb{T}) where {T}
  xdim, udim, N = prob.xdim, prob.udim, prob.N
  numel = N * (xdim^2 + xdim * udim + xdim)
  Ap = zeros(Int, N * (xdim + udim) + 1)
  Ai, Ax = zeros(Int, numel), zeros(T, numel)
  k, c = 0, 0
  Ap[1] = 1
  for j in 1:N
    for t in 1:udim
      k_old = k
      for r in 1:xdim
        k += 1
        Ai[k] = xdim * (j - 1) + r
        Ax[k] = prob.fu[r, t, j]
      end
      c += 1
      Ap[c + 1] = Ap[c] + (k - k_old)
    end
  end
  for j in 1:N
    for t in 1:xdim
      k_old = k
      k += 1
      Ai[k] = xdim * (j - 1) + t
      Ax[k] = -1.0
      if j != N
        for r in 1:xdim
          k += 1
          Ai[k] = xdim * (j - 1) + xdim + r
          Ax[k] = prob.fx[r, t, j + 1]
        end
      end
      c += 1
      Ap[c + 1] = Ap[c] + (k - k_old)
    end
  end
  b = zeros(T, N * xdim)
  for j in 1:N
    sidx = xdim * (j - 1)
    @inbounds for r in 1:xdim
      val = -prob.f[r, j]
      @inbounds for t in 1:udim
        val += prob.fu[r, t, j] * prob.U_prev[t, j]
      end
      if j != 1
        @inbounds for t in 1:xdim
          val += prob.fx[r, t, j] * prob.X_prev[t, j - 1]
        end
      else
      end
      b[sidx + r] = val
    end
  end
  #A = SparseMatrixCSC{T, Int}(N * xdim, N * (xdim + udim), Ap, Ai, Ax)
  A = SparseMatrixCSC{T, Int}(N * xdim, N * (xdim + udim), Ap, Ai[1:Ap[end-1]], Ax[1:Ap[end-1]])
  return A, b
end

function qp_repr_Glu(prob::OCProb{T}) where {T}
  G = sparse(T(1) * I, 0, prob.N * (prob.xdim + prob.udim))
  l, u = zeros(T, 0), zeros(T, 0)
  if prob.lu != nothing && prob.uu != nothing
    G = vcat(G, sparse(T(1) * I, prob.N * prob.udim, prob.N * (prob.udim + prob.xdim)))
    l, u = [l; view(prob.lu, :)], [u; view(prob.uu, :)]
  end
  if prob.lx != nothing && prob.ux != nothing
    Gx = hcat(
      spzeros(T, prob.N * prob.xdim, prob.N * prob.udim),
      sparse(T(1) * I, prob.N * prob.xdim, prob.N * prob.xdim),
    )
    G = vcat(G, Gx)
    l, u = [l; view(prob.lx, :)], [u; view(prob.ux, :)]
  end
  return G, l, u
end
# QP representations ###########################################################
