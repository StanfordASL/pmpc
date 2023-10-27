
function ECOS_setup(problem::ConeProblem; settings...)
  l, q, e, G, A, c, h, b = [getproperty(problem, k) for k in [:l, :q, :e, :G, :A, :c, :h, :b]]
  G_ = G != nothing ? copy(G) : spzeros(0, length(c))
  A_ = A != nothing ? copy(A) : spzeros(0, length(c))
  c_ = copy(c)
  h_ = h != nothing ? copy(h) : zeros(0)
  b_ = b != nothing ? copy(b) : zeros(0)
  m, n = size(G_)
  p = size(A_, 1)

  # constraints must add to the size of G
  @assert (l + sum(q) + 3 * e == m) "m = $(m), but (l, sum(q), e) = $((l, sum(q), e))"
  @assert length(c_) == size(G_, 2) == size(A_, 2)
  @assert size(G_, 1) == length(h_)
  @assert size(A_, 1) == length(b_)

  # we're using G * x - h <=_K 0 but ECOS is using h - G * x <=_K 0
  G_[(l + 1):end, :] .*= -1
  h_[(l + 1):end, :] .*= -1

  Gpr, Gjc, Gir = G_.nzval, G_.colptr .- 1, G_.rowval .- 1
  Apr, Ajc, Air = A_.nzval, A_.colptr .- 1, A_.rowval .- 1

  probp = ECOS.ECOS_setup(n, m, p, l, length(q), q, e, Gpr, Gjc, Gir, Apr, Ajc, Air, c_, h_, b_)

  # set settings of the problem, loaded memory is non-modifiable
  solver_settings = Dict{Symbol, Any}(get(settings, :solver_settings, []))
  prob = unsafe_load(probp)
  old_settings = unsafe_load(prob.stgs)
  new_settings = ECOS.settings(
    [
      (
        haskey(solver_settings, property) ?
        typeof(getproperty(old_settings, property))(solver_settings[property]) :
        getproperty(old_settings, property)
      ) for property in propertynames(old_settings)
    ]...,
  )
  unsafe_store!(prob.stgs, new_settings)
  var_ref_for_gc = (Gpr, Gjc, Gir, Apr, Ajc, Air, c_, h_, b_, new_settings)
  return probp, var_ref_for_gc
end

function ECOS_solve(problem::ConeProblem; settings...)
  l, q, e, G, A, c, h, b = [getproperty(problem, k) for k in [:l, :q, :e, :G, :A, :c, :h, :b]]
  (G != nothing && !(typeof(G) <: SpMat{F64, Int})) && (problem.G = sparse(G))
  (A != nothing && !(typeof(A) <: SpMat{F64, Int})) && (problem.A = sparse(A))
  probp, var_ref_for_gc = ECOS_setup(problem; settings...)
  @assert probp != Ptr{ECOS.pwork}(0)
  status = ECOS.ECOS_solve(probp)
  #(status != 0) && (@warn("ECOS status is not optimal, status = $(status)"))
  prob = unsafe_load(probp)
  n, m, p = prob.n, prob.m, prob.p

  info = deepcopy(unsafe_load(prob.info))
  x = copy(unsafe_wrap(Array, prob.x, n))
  y = copy(unsafe_wrap(Array, prob.y, prob.p))
  z = copy(unsafe_wrap(Array, prob.z, m))
  s = copy(unsafe_wrap(Array, prob.s, m))
  ECOS.ECOS_cleanup(probp, 0)
  return ECOS_result(x, y, z, s, info)
end

function COSMO_solve(problem::ConeProblem; settings...)
  l, q, e, G, A, c, h, b = [getproperty(problem, k) for k in [:l, :q, :e, :G, :A, :c, :h, :b]]
  G = G != nothing ? G : spzeros(0, length(problem.c))
  h = h != nothing ? h : zeros(0)
  A = A != nothing ? A : spzeros(0, length(problem.c))
  b = b != nothing ? b : zeros(0)
  !(typeof(G) <: SpMat{F64, Int}) && (G = sparse(G))
  !(typeof(A) <: SpMat{F64, Int}) && (A = sparse(A))
  m, n = size(G)
  @assert (l + sum(q) + 3 * e == m) "m = $(m), but (l, sum(q), e) = $((l, sum(q), e))"
  @assert length(c) == size(G, 2) == size(A, 2)
  @assert size(G, 1) == length(h)
  @assert size(A, 1) == length(b)

  model = COSMO.Model()
  cstr = COSMO.Constraint{F64}[]

  if A != nothing
    push!(cstr, COSMO.Constraint(A, -b, COSMO.ZeroSet))
  end

  # COSMO uses the convention G x + h, but we're using G * x - h 
  k = 0
  if l > 0
    push!(cstr, COSMO.Constraint(-G[(k + 1):(k + l), :], h[(k + 1):(k + l)], COSMO.Nonnegatives))
    k += l
  end
  for q_size in q
    push!(
      cstr,
      COSMO.Constraint(G[(k + 1):(k + q_size), :], -h[(k + 1):(k + q_size)], COSMO.SecondOrderCone),
    )
    k += q_size
  end
  if e > 0
    for _ in 1:e
      push!(
        cstr,
        COSMO.Constraint(G[(k + 1):(k + 3), :], -h[(k + 1):(k + 3)], COSMO.ExponentialCone),
      )
      k += 3
    end
  end
  solver_settings = Dict{Symbol, Any}(get(settings, :solver_settings, []))
  stgs = COSMO.Settings()
  for pair in pairs(solver_settings)
    if pair.first in propertynames(stgs)
      setproperty!(stgs, pair.first, pair.second)
    end
  end
  COSMO.assemble!(model, spzeros(n, n), c, cstr; settings=stgs)

  result = COSMO.optimize!(model)
  return result
end

function JuMP_solve(problem::ConeProblem; settings...)
  l, q, e, G, A, c, h, b = [getproperty(problem, k) for k in [:l, :q, :e, :G, :A, :c, :h, :b]]
  G = G != nothing ? G : spzeros(0, length(problem.c))
  h = h != nothing ? h : zeros(0)
  A = A != nothing ? A : spzeros(0, length(problem.c))
  b = b != nothing ? b : zeros(0)
  (G != nothing && !(typeof(G) <: SpMat{F64, Int})) && (problem.G = sparse(G))
  (A != nothing && !(typeof(A) <: SpMat{F64, Int})) && (problem.A = sparse(A))
  settings = Dict{Symbol, Any}(settings...)
  get!(settings, :solver, "ecos")
  solver_settings = Dict{Symbol, Any}(get(settings, :solver_settings, []))

  if lowercase(settings[:solver]) == "ecos"
    model = JuMP.Model(ECOS.Optimizer)
    set_attribute(model, "verbose", get(settings, :verbose, false))
  elseif lowercase(settings[:solver]) == "mosek"
    #error("Not currently supported")
    model = JuMP.Model(Mosek.Optimizer)
    set_attribute(model, "QUIET", !get(settings, :verbose, false))
  elseif lowercase(settings[:solver]) == "gurobi"
    redirect_target = get(settings, :verbose, false) ? stdout : devnull
    redirect_stdout(redirect_target) do
      model = JuMP.Model(Gurobi.Optimizer)
      set_attribute(model, "OutputFlag", Int(get(settings, :verbose, false)))
    end
  else
    error("Solver is misspecified: $(settings[:solver])")
  end
  # now apply solver specific settings #############################################################
  for p in pairs(solver_settings)
    try # only if the model has such set_attribute
      set_attribute(model, p.first, p.second)
    catch e
    end
  end

  z = @variable(model, z[1:length(problem.c)])
  cstr = []
  if size(A, 1) > 0
    push!(cstr, @constraint(model, A * z - b .== 0))
  end
  k = 0
  if l > 0
    push!(cstr, @constraint(model, -G[1:(problem.l), :] * z + h[1:l] in MOI.Nonnegatives(l)))
    k += l
  end
  if length(q) > 0
    for q_size in q
      push!(
        cstr,
        @constraint(
          model,
          G[(k + 1):(k + q_size), :] * z - h[(k + 1):(k + q_size)] in MOI.SecondOrderCone(q_size)
        )
      )
      k += q_size
    end
  end
  if e > 0
    for _ in 1:e
      push!(
        cstr,
        @constraint(model, G[(k + 1):(k + 3), :] * z - h[(k + 1):(k + 3)] in MOI.ExponentialCone())
      )
      k += 3
    end
  end
  @objective(model, Min, c' * z)
  JuMP.optimize!(model)
  return (x=value.(z),)
end
