##^# library imports ###########################################################
using Base.Threads
##$#############################################################################
##^# large QP; P, q ############################################################
function lqp_repr_Pq(probs::AA{OCProb{T}, 1}, Nc::Integer; settings...) where {T}
  xdim, udim, N, M = probs[1].xdim, probs[1].udim, probs[1].N, length(probs)
  Nc = Nc >= 0 ? Nc : N
  Nf = N - Nc
  elnum = 3 * Nc * udim^2 + M * (3 * Nf * udim^2 + N * xdim^2)

  # final cost with mixed terms ################################################
  haskey(settings, :Hf) && (elnum += (M * xdim)^2 - M * xdim^2)
  # final cost with mixed terms ################################################

  n = Nc * udim + M * (Nf * udim + N * xdim)
  Pp = zeros(Int, n + 1)
  Pi, Px = zeros(Int, elnum), zeros(T, elnum)
  k, c = 0, 0
  Pp[1] = 1
  for j in 1:Nc # for U consensus
    @inbounds for t in 1:udim
      k_old = k
      slew_reg = mapreduce(i -> probs[i].slew_reg, +, 1:M)
      if j > 1 && slew_reg != 0.0 # slew top half
        k += 1
        Pi[k] = udim * (j - 2) + t
        Px[k] = -slew_reg
      end
      @inbounds for r in 1:udim # core cost and residual
        val = 0.0
        for i in 1:M
          val += probs[i].R[r, t, j]
          if r == t
            val += probs[i].reg_u
            if j == 1
              val += probs[i].slew_reg0 + probs[i].slew_reg
            elseif j == N
              val += probs[i].slew_reg
            else
              val += 2 * probs[i].slew_reg
            end
          end
        end
        if val != 0.0
          k += 1
          Pi[k] = udim * (j - 1) + r
          Px[k] = val
        end
      end
      if j < Nc && slew_reg != 0.0 # slew bottom half
        k += 1
        Pi[k] = udim * j + t
        Px[k] = -slew_reg
      elseif j < N && slew_reg != 0.0
        for i in 1:M
          k += 1
          Pi[k] = Nc * udim + udim * Nf * (i - 1) + t
          Px[k] = -probs[i].slew_reg
        end
      end
      c += 1
      Pp[c + 1] = Pp[c] + (k - k_old)
    end
  end
  for i in 1:M # for U free
    for j in (Nc + 1):N
      @inbounds for t in 1:udim
        k_old = k
        slew_reg = probs[i].slew_reg
        if j > 1 && slew_reg != 0.0 # slew top half
          idx = (
            j - 1 > Nc ? Nc * udim + Nf * udim * (i - 1) + udim * (j - Nc - 2) + t :
            udim * (j - 2) + t
          )
          k += 1
          Pi[k] = idx
          Px[k] = -probs[i].slew_reg
        end
        @inbounds for r in 1:udim # core cost and residual
          val = probs[i].R[r, t, j]
          if r == t
            val += probs[i].reg_u
            if j == 1
              val += probs[i].slew_reg0 + probs[i].slew_reg
            elseif j == N
              val += probs[i].slew_reg
            else
              val += 2 * probs[i].slew_reg
            end
          end
          if val != 0.0
            k += 1
            Pi[k] = Nc * udim + Nf * udim * (i - 1) + udim * (j - Nc - 1) + r
            Px[k] = val
          end
        end
        if j < N && slew_reg != 0.0 # slew bottom half
          k += 1
          Pi[k] = Nc * udim + Nf * udim * (i - 1) + udim * (j - Nc) + t
          Px[k] = -probs[i].slew_reg
        end
        c += 1
        Pp[c + 1] = Pp[c] + (k - k_old)
      end
    end
  end

  # final cost with mixed terms ################################################
  Hf = get(settings, :Hf, nothing)
  # final cost with mixed terms ################################################

  offset = Nc * udim + M * Nf * udim
  for i in 1:M # for X
    for j in 1:N
      @inbounds for t in 1:xdim
        k_old = k

        # final cost with mixed terms ##########################################
        if j == N && Hf != nothing # final state
          for i_ in 1:(i - 1)
            @inbounds for r in 1:xdim
              val = Hf[xdim * (i_ - 1) + r, xdim * (i - 1) + t]
              if val != 0.0
                k += 1
                Pi[k] = offset + N * xdim * (i_ - 1) + xdim * (N - 1) + r
                Px[k] = val
              end
            end
          end
        end
        # final cost with mixed terms ##########################################

        @inbounds for r in 1:xdim
          val = probs[i].Q[r, t, j] + (r == t ? probs[i].reg_x : 0.0)

          # final cost with mixed terms ########################################
          (j == N && Hf != nothing) && (val += Hf[xdim * (i - 1) + r, xdim * (i - 1) + t])
          # final cost with mixed terms ########################################

          if val != 0.0
            k += 1
            Pi[k] = offset + N * xdim * (i - 1) + xdim * (j - 1) + r
            Px[k] = val
          end
        end

        # final cost with mixed terms ##########################################
        if j == N && Hf != nothing
          for i_ in (i + 1):M
            @inbounds for r in 1:xdim
              val = Hf[xdim * (i_ - 1) + r, xdim * (i - 1) + t]
              if val != 0.0
                k += 1
                Pi[k] = offset + N * xdim * (i_ - 1) + xdim * (N - 1) + r
                Px[k] = val
              end
            end
          end
        end
        # final cost with mixed terms ##########################################

        c += 1
        Pp[c + 1] = Pp[c] + (k - k_old)
      end
    end
  end
  q = zeros(T, n)
  q[1:udim] .+= mapreduce(i -> -probs[i].slew_reg0 * probs[i].slew_um1, +, 1:M)
  for j in 1:Nc # for U cons
    sidx = udim * (j - 1)
    for r in 1:udim
      val = 0.0
      for i in 1:M
        val -= probs[i].reg_u * probs[i].U_prev[r, j]
        @inbounds for t in 1:udim
          val -= probs[i].R[r, t, j] * probs[i].U_ref[t, j]
        end
      end
      q[sidx + r] += val
    end
  end
  for i in 1:M # for U free
    for j in (Nc + 1):N
      sidx = Nc * udim + Nf * udim * (i - 1) + udim * (j - Nc - 1)
      @inbounds for r in 1:udim
        val = -probs[i].reg_u * probs[i].U_prev[r, j]
        @inbounds for t in 1:udim
          val -= probs[i].R[r, t, j] * probs[i].U_ref[t, j]
        end
        q[sidx + r] = val
      end
    end
  end

  # final cost with mixed terms ################################################
  hf = get(settings, :hf, nothing)
  # final cost with mixed terms ################################################

  for i in 1:M # for X
    for j in 1:N
      sidx = udim * (M * Nf + Nc) + N * xdim * (i - 1) + xdim * (j - 1)
      @inbounds for r in 1:xdim
        val = -probs[i].reg_x * probs[i].X_prev[r, j]

        # final cost with mixed terms ##########################################
        (j == N && hf != nothing) && (val += hf[xdim * (i - 1) + r])
        # final cost with mixed terms ##########################################

        @inbounds for t in 1:xdim
          val -= probs[i].Q[r, t, j] * probs[i].X_ref[t, j]
        end
        q[sidx + r] = val
      end
    end
  end
  Pi, Px = Pi[1:k], Px[1:k]
  P = SpMat{T, Int}(n, n, Pp, Pi, Px)
  return P, q
end
##$#############################################################################
##^# large QP; A, b ############################################################
function lqp_repr_Ab(probs::AA{OCProb{T}, 1}, Nc::Integer) where {T}
  xdim, udim, N, M = probs[1].xdim, probs[1].udim, probs[1].N, length(probs)
  Nc = Nc >= 0 ? Nc : N
  Nf = N - Nc
  n = Nc * udim + M * (Nf * udim + N * xdim)
  m = M * N * xdim
  numel = xdim * udim * M * N + M * N * xdim + M * (N - 1) * xdim^2
  Ap = zeros(Int, M * (N * xdim + Nf * udim) + Nc * udim + 1)
  Ai = zeros(Int, numel)
  Ax = zeros(T, numel)
  k, c = 0, 0
  Ap[1] = 1
  for j in 1:Nc # for U consensus
    for t in 1:udim
      k_old = k
      for i in 1:M
        for r in 1:xdim
          k += 1
          Ai[k] = N * xdim * (i - 1) + xdim * (j - 1) + r
          Ax[k] = probs[i].fu[r, t, j]
        end
      end
      c += 1
      Ap[c + 1] = Ap[c] + (k - k_old)
    end
  end
  for i in 1:M # for U free
    for j in (Nc + 1):N
      for t in 1:udim
        k_old = k
        for r in 1:xdim
          k += 1
          Ai[k] = N * xdim * (i - 1) + xdim * (j - 1) + r
          Ax[k] = probs[i].fu[r, t, j]
        end
        c += 1
        Ap[c + 1] = Ap[c] + (k - k_old)
      end
    end
  end
  for i in 1:M # for X
    for j in 1:N
      for t in 1:xdim
        k_old = k
        k += 1
        Ai[k] = N * xdim * (i - 1) + xdim * (j - 1) + t
        Ax[k] = -1.0
        if j != N
          for r in 1:xdim
            k += 1
            Ai[k] = N * xdim * (i - 1) + xdim * (j - 1) + xdim + r
            Ax[k] = probs[i].fx[r, t, j + 1]
          end
        end
        c += 1
        Ap[c + 1] = Ap[c] + (k - k_old)
      end
    end
  end
  b = zeros(T, M * N * xdim)
  for i in 1:M
    for j in 1:N
      sidx = N * xdim * (i - 1) + xdim * (j - 1)
      val = 0.0
      @inbounds for r in 1:xdim
        val = -probs[i].f[r, j]
        @inbounds for t in 1:udim
          val += probs[i].fu[r, t, j] * probs[i].U_prev[t, j]
        end
        if j != 1
          @inbounds for t in 1:xdim
            val += probs[i].fx[r, t, j] * probs[i].X_prev[t, j - 1]
          end
        else
          #@inbounds for t in 1:xdim
          #  val += probs[i].fx[r, t, j] * probs[i].x0[t]
          #end
        end
        b[sidx + r] = val
      end
    end
  end
  A = SpMat{T, Int}(m, n, Ap, Ai, Ax)
  return A, b
end
##$#############################################################################
##^# large QP; G, l, u #########################################################
function lqp_repr_Gla(probs::AA{OCProb{T}, 1}, Nc::Integer) where {T}
  xdim, udim, N, M = probs[1].xdim, probs[1].udim, probs[1].N, length(probs)
  Nc = Nc >= 0 ? Nc : N
  Nf = N - Nc

  m = 0
  if probs[1].lu != nothing && probs[1].uu != nothing
    m += udim * Nc + M * Nf * udim
  end
  if probs[1].lx != nothing && probs[1].ux != nothing
    m += M * N * xdim
  end
  n = Nc * udim + M * (Nf * udim + N * xdim)
  Gp, Gi, Gx = zeros(Int, n + 1), zeros(Int, m), zeros(T, m)
  l, u = zeros(T, m), zeros(T, m)
  k, c = 0, 0
  Gp[1] = 1
  if probs[1].lu != nothing && probs[1].uu != nothing
    for j in 1:Nc
      for r in 1:udim
        k += 1
        Gi[k] = udim * (j - 1) + r
        Gx[k] = 1.0
        l[k] = probs[1].lu[udim * (j - 1) + r]
        u[k] = probs[1].uu[udim * (j - 1) + r]

        c += 1
        Gp[c + 1] = Gp[c] + 1
      end
    end
    for i in 1:M
      for j in (Nc + 1):N
        for r in 1:udim
          k += 1
          Gi[k] = Nc * udim + Nf * udim * (i - 1) + udim * (j - Nc - 1) + r
          Gx[k] = 1.0
          l[k] = probs[i].lu[udim * (j - 1) + r]
          u[k] = probs[i].uu[udim * (j - 1) + r]

          c += 1
          Gp[c + 1] = Gp[c] + 1
        end
      end
    end
  else
    for j in 1:Nc
      for r in 1:udim
        c += 1
        Gp[c + 1] = Gp[c]
      end
    end
    for i in 1:M
      for j in (Nc + 1):N
        for r in 1:udim
          c += 1
          Gp[c + 1] = Gp[c]
        end
      end
    end
  end
  if probs[1].lx != nothing && probs[1].ux != nothing
    for i in 1:M
      for j in 1:N
        for r in 1:xdim
          k += 1
          Gi[k] = N * xdim * (i - 1) + xdim * (j - 1) + r
          Gi[k] += probs[1].lu != nothing && probs[1].uu != nothing ? udim * (Nc + M * Nf) : 0
          Gx[k] = 1.0
          l[k] = probs[i].lx[xdim * (j - 1) + r]
          u[k] = probs[i].ux[xdim * (j - 1) + r]

          c += 1
          Gp[c + 1] = Gp[c] + 1
        end
      end
    end
  else
    for i in 1:M
      for j in 1:N
        for r in 1:xdim
          c += 1
          Gp[c + 1] = Gp[c]
        end
      end
    end
  end
  return SpMat{T, Int}(m, n, Gp, Gi, Gx), l, u
end

function split_lqp_vars(probs::AA{OCProb{T}, 1}, Nc::Integer, z::AA{T, 1}) where {T}
  xdim, udim, N, M = probs[1].xdim, probs[1].udim, probs[1].N, length(probs)
  Nc = Nc >= 0 ? Nc : N
  Nf = N - Nc
  X, U = zeros(T, xdim, N, M), zeros(T, udim, N, M)
  for i in 1:M
    for j in 1:Nc
      for r in 1:udim
        U[r, j, i] = z[udim * (j - 1) + r]
      end
    end
  end
  for i in 1:M
    for j in (Nc + 1):N
      for r in 1:udim
        U[r, j, i] = z[Nc * udim + Nf * udim * (i - 1) + udim * (j - Nc - 1) + r]
      end
    end
  end
  offset = Nc * udim + M * Nf * udim
  for i in 1:M
    for j in 1:N
      for r in 1:xdim
        X[r, j, i] = z[offset + N * xdim * (i - 1) + xdim * (j - 1) + r]
      end
    end
  end
  return X, U
end
##$#############################################################################
