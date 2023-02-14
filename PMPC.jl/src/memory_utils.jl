const SolverType = Union{JuMPSolver, OSQPSolver}
SOLVER_STORE = Dict{String, SolverType}()

function store_solver!(solver::SolverType, name_id::String)::Nothing
  global SOLVER_STORE
  SOLVER_STORE[name_id] = solver
  return
end

function restore_solver(name_id::String)::Union{SolverType, Nothing}
  global SOLVER_STORE
  return get(SOLVER_STORE, name_id, nothing)
end

function delete_solver!(name_id::String)::Bool
  global SOLVER_STORE
  if haskey(SOLVER_STORE, name_id)
    delete!(SOLVER_STORE, name_id)
    return true
  else
    return false
  end
end

function delete_all_solvers!()::Nothing
  global SOLVER_STORE
  SOLVER_STORE = Dict{String, SolverType}()
  return
end

function generate_unique_solver_id()::String
  global SOLVER_STORE
  while true
    solver_id = @sprintf("solver_%d", rand((10^5):(10^7)))
    (!haskey(SOLVER_STORE, solver_id)) && (return solver_id)
  end
end
