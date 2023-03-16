# `pmpc`
Python-interface Particle Sequential Convex Programming Model Predictive Control (SCP PMPC) interface.

This is non-linear dynamics finite horizon MPC solver with consensus
optimization capability, support for arbitrary constraints and arbitrary cost.

# Table of Contents 
- [`pmpc`](#pmpc)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
  - [(Optional) Obtaining (a dynamically linked version of) Python](#optional-obtaining-a-dynamically-linked-version-of-python)
  - [Compilation Times and a Persistent Solver Process](#compilation-times-and-a-persistent-solver-process)
  - [Obtaining Julia](#obtaining-julia)
- [Basic Usage](#basic-usage)
  - [Defining dynamics](#defining-dynamics)
  - [Defining Cost](#defining-cost)
- [`solve` Method Arguments Glossary](#solve-method-arguments-glossary)
  - [Solver Hyperparameters](#solver-hyperparameters)
  - [Solver Settings](#solver-settings)
  - [Additional Dynamics Settings](#additional-dynamics-settings)
  - [Nonlinear Cost and Constraints](#nonlinear-cost-and-constraints)
    - [Variable Layout](#variable-layout)
  - [Misc Settings](#misc-settings)
- [Advanced Usage](#advanced-usage)
  - [Multiple solver processes](#multiple-solver-processes)
  - [Consensus Optimization for Control under Uncertainty](#consensus-optimization-for-control-under-uncertainty)
  - [Non-convex Cost Example](#non-convex-cost-example)
  - [Arbitrary Constraints](#arbitrary-constraints)
    - [Linear Constraints](#linear-constraints)
    - [Second-order Cone Constraints (SOCP)](#second-order-cone-constraints-socp)
    - [Exponential Cone Constraints](#exponential-cone-constraints)
  - [Solver selection](#solver-selection)
- [Particle (consensus/contingency) optimization](#particle-consensuscontingency-optimization)
- [Warm-start support](#warm-start-support)

# Installation

Installation can be done by cloning this repository and issuing `pip install .`

```bash
$ git clone https://github.com/StanfordASL/pmpc.git
$ cd pmpc
$ pip install .
```

*Note: you must have [julia](https://julialang.org/) in your system PATH.*

Further subsections explain some **optional** installation steps.

## (Optional) Obtaining (a dynamically linked version of) Python 

The Python module uses [pyjulia](https://github.com/JuliaPy/pyjulia) to be able
to call Julia from Python which has a limitation in that it works much better
(faster startup time) with **a Python version which is not statically linked to
libpython**. A great workaround, obtaining a python version that is not
statically linked can be easily done via [pyenv](https://github.com/pyenv/pyenv).

Instructions on how to build a python version dynamically linked to libpython can be found
- in `pyjulia` documentation [here](https://pyjulia.readthedocs.io/en/stable/troubleshooting.html?highlight=shared#ultimate-fix-build-your-own-python)
- or in `pyenv` documentation [here](https://github.com/pyenv/pyenv/wiki#how-to-build-cpython-with---enable-shared)

With a working version of `pyenv`, an example might be
```bash
$ env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.9.13
$ pyenv virtualenv 3.9.13 {your venv name} 
$ pyenv activate {your venv name}
```

## Compilation Times and a Persistent Solver Process

A large downside of using a Julia for the core solver is that compilation needs
to occur every time a new process is launched. This is exacerbated by
[pyjulia](https://github.com/JuliaPy/pyjulia) which does not work well with
not-dynamically linked python interpreters (most commonly used).

To overcome compilation times, we can start a solver process once and the library can
use that solver process repeatedly, even through script restarts.

Use
```bash
$ python3 -m pmpc.remote
```
to start a persistent solver process.

Next, from your script
```python
>>> from pmpc.remote import solve # instead of `from pmpc import solve`
```

## Obtaining Julia

You can download [Julia](https://julialang.org/), the programming language and interpreter, from [here](https://julialang.org/downloads/).

Make sure Julia is in your `PATH`.

# Basic Usage

The solver is capable of MPC consensus optimization for several system instantiations. For the basic usage, we'll focus on a single system MPC.

A basic MPC problem is defined using the dynamics and a quadratic cost

## Defining dynamics

- `x0` the initial state of shape

where
```python
np.shape(x0) == (xdim,)
```

- `f, fx, fu = f_fx_fu_fn(xt, ut)` an affine dynamics linearization, such that
$$x^{(i+1)} \approx f^{(i)} + f_x^{(i)} (x^{(i)} - \tilde{x}^{(i)}) + f_u^{(i)} (u^{(i)} - \tilde{u}^{(i)}) $$
where 
```python
np.shape(xt) == (N, xdim)
np.shape(ut) == (N, udim)
np.shape(f) == (N, xdim)
np.shape(fx) == (N, xdim, xdim)
np.shape(fu) == (N, xdim, udim)
```

## Defining Cost

- `X_ref, Q` a reference and quadratic weight matrix for state cost
- `U_ref, R` a reference and quadratic weight matrix for control cost

The cost is given as 

$$J = \sum_{i=0}^N 
\frac{1}{2} (x^{(i+1)} - x_\text{ref}^{(i+1)}) Q^{(i)} (x^{(i+1)} - x_\text{ref}^{(i+1)}) + 
\frac{1}{2} (u^{(i)} - u_\text{ref}^{(i)}) R^{(i)} (u^{(i)} - u_\text{ref}^{(i)})$$

*Note: Initial state, x0, is assumed constant and thus does not feature in the cost.*

*Note: When handling controls, we'll always have `np.shape(U) == (N, udim)`*

*Note: When handling states, we'll have either `np.shape(X) == (N + 1, xdim)` with `x0` included at the beginning or `np.shape(X) == (N, xdim)` with `x0` NOT INCLUDED. `X[:, -1]` always refers to the state N, whereas `U[:, -1]` always refers to control N - 1.*

Thus, an example call would be

```python
>>> import pmpc
>>> X, U, debug = pmpc.solve(f_fx_fu_fn, Q, R, x0, X_ref, U_ref)
>>> help(pmpc.solve)
```

Take a look at
- `tests/simple.py` for simple usage
- `tests/dubins_car.py` for defining dynamics

# `solve` Method Arguments Glossary

## Solver Hyperparameters

The solver has two scalar hyperparamters, the dynamics linearization deviation penalty for states and controls

$$
J_\text{deviation} = \sum_{i=0}^N \frac{1}{2} 
\rho_x (x^{(i+1)} - x_\text{prev}^{(i+1)})^T (x^{(i+1)} - x_\text{prev}^{(i+1)})
+ \rho_u (u^{(i)} - u_\text{prev}^{(i)})^T (u^{(i)} - u_\text{prev}^{(i)})
$$

- `reg_x` - state deviation in-between SCP iterations regularization
- `reg_u` - control deviation in-between SCP iterations regularization

Higher values will slow evolution between SCP iterations and will require more
SCP iterations to converge to a solution, but will avoid instability in the
solution if the dynamics are not sufficiently smooth.

## Solver Settings

- `verbose` - whether to print iteration status (user-facing)
- `debug` - whether to print very low-level debugging information (developer-facing)
- `max_it` - maximum number of SCP iterations to perform (can be fewer if tolerance met earlier)
- `time_limit` - the time limit in seconds for SCP iteration
- `res_tol` - deviation tolerance past which solution is accepted (measure of convergence)
- `slew_rate` - the quadratic penalty between time-consecutive controls (encourages smooth controls)
- `u_slew` - the previous action taken to align the first plan action with (useful for smooth receding horizon control)

## Additional Dynamics Settings

- `X_prev` - previous state solution (guess), $x^{(i)} ~~ \forall i \in [1, \dots, N]$, `shape = (N, xdim)`
- `U_prev` - previous control solution (guess),  $u^{(i)} ~~ \forall i \in [0, \dots, N - 1]$, `shape = (N, udim)`
- `x_l` - state lower box constraints, $x^{(i)} ~~ \forall i \in [1, \dots, N]$, `shape = (N, xdim)`
- `x_u` - state upper box constraints, $x^{(i)} ~~ \forall i \in [1, \dots, N]$, `shape = (N, xdim)`
- `u_l` - control lower box constraints, $u^{(i)} ~~ \forall i \in [0, \dots, N - 1]$, `shape = (N, udim)`
- `u_u` - control upper box constraints, $u^{(i)} ~~ \forall i \in [0, \dots, N - 1]$, `shape = (N, udim)`

## Nonlinear Cost and Constraints

The solver supports custom arbitrary cost via each-SCP-iteration cost linearization and custom constraints via each-SCP-iteration constraint reformulation into any convex-cone constraint.

- `lin_cost_fn` is an optional callable which allows specifying a custom cost, it
should take arbitrary `X`, `U` and return a tuple
    - `cx`, the linearization of the cost with respect to the state, `np.shape(cx) == (N, xdim) or cx is None`
    - `cu`, the linearization of the cost with respect to the controls, `np.shape(cu) == (N, udim) or cu is None`

I highly recommend using an auto-diff library to produce the linearizations to avoid unnecessary bugs.

- `extra_cstrs_fns` is an optional callable which returns **a list** of conic constraints given arbitrary `X, U`
  - a conic constraint $G z - h \in \mathcal{K}$, (e.g., $A z - b \leq 0$) consists of a tuple of 8 elements
    - `l` - the number of non-positive orthant constraints (an integer)
    - `q` - a list with the sizes of second order cone constraints (list of integers)
    - `e` - the number of exponential cone constraints (integer)
    - `G_left` - the G matrix referring to existing variables
    - `G_right` - the G matrix for new variables to introduce
    - `h` - the right hand side vector for the constraint
    - `c_left` - the additive linear cost augmentation for existing variables
    - `c_right` - the linear (minimization) cost for new variables to introduce

 *Note: `lin_cost_fn` is expected to return a tuple, but `extra_cstrs_fns` must be a list of functions!*

### Variable Layout

The variable layout is control variables (`# = N udim`) followed by state variables (`# = N xdim`).

For consensus optimization the layout is 
- consensus controls (`# = Nc udim`)
- free controls (`# = M (N - Nc) udim`)
- states (`# = M N xdim`)

## Misc Settings

- `solver_settings` - a dictionary of settings to pass to the lower-level Julia solver
- `solver_state` - a previous solver state to pass to the lower-level Julia solver
- `filter_method` - whether to apply SCP solution low-pass-like filtering to avoid SCP solution oscillation
- `filter_window` - how large a SCP iteration window to pick for SCP solution filtering
- `filter_it0` - what iteration of SCP to start applying filtering from
- `return_min_viol` - whether to return the minimum deviation solution (measured from previous), not the last one
- `min_viol_it0` - what iteration to start looking for minimum deviation solution

# Advanced Usage

## Multiple solver processes

Launching multiple solver processes can be useful when many problems need to be solved. This presents some challenges
- the solver needs to be able to discover existing, ready persistent worker processes
  - we use [redis](https://redis.io/) here
- batch of problem specifications need to be fed to a solution function
  - we redefine the optimal control problem slightly as a set of keyword arguments for the `solve` function, positional arguments are now keywords arguments

Launch
```bash
$ python3 -m pmpc.remote --help

usage: remote.py [-h] [--port PORT] [--verbose] [--worker-num WORKER_NUM]
                 [--redis-host REDIS_HOST] [--redis-port REDIS_PORT]
                 [--redis-password REDIS_PASSWORD]

optional arguments:
  -h, --help            show this help message and exit
  --port PORT, -p PORT  TCP port on which to start the server
  --verbose, -v
  --worker-num WORKER_NUM, -n WORKER_NUM
                        Number of workers to start, 0 means number equal to
                        physical CPU cores.
  --redis-host REDIS_HOST
                        Redis host
  --redis-port REDIS_PORT
                        Redis port
  --redis-password REDIS_PASSWORD
                        Redis password

$ python3 -m pmpc.remote --worker-num 0 --redis-password my_redis_password
```
then
```python
from pmpc.remote import solve_problems

problems = [generate_problem(...) for _ in range(32)]
solutions = solve_problems(problems)
```

We solve the discovery problem by making use of [redis](https://redis.io/), an in-memory key-value store (a database). **You need to have a working `redis-server` running** (e.g., `sudo apt install redis-server`).

*Note: Since our worker processes are TCP based, they can be run on any network connected computer (which has the copy of the python environment). As long as we can ensure that all computers can access a redis database where workers register.*

We solve the problem specification by redefining the optimal control problem slightly. All arguments to the `solve` function are now keyword arguments **a problem is defined as a dictionary `Dict[str, Any]`**. 

The `solve_problems` method solves a list of dictionary problem definitions and returns a list of solutions.

## Consensus Optimization for Control under Uncertainty

For consensus control, simply 
- batch the system instantiations into an extra (first) dimension
  - all problem data now obtains a size `M` first dimensions
- specify `Nc` as a field in `solver_settings`, the length of the consensus horizon

TODO more detailed explanation

## Non-convex Cost Example

```python
N, xdim, udim = 10, 3, 2
X_ref = np.random.rand(N, xdim)

def lin_cost_fn(X, U):
    # cost is np.sum((X - X_ref) ** 2)
    cx = 2 * (X - X_ref)
    cu = None
    return cx, cu
```

## Arbitrary Constraints

Arbitrary convex (cone) constraints can be introduced in a canonical form via a callaback which recomputes them at every SCP iteration. This allows to encode non-convex constraints via their SCP convexification.

The canonical form for a convex problem is given by

$$
\begin{aligned}
& \text{minimize} && c^T z \\
& \text{such that} && A_i z \leq_{\mathcal{K}} b_i ~~ \forall i
\end{aligned}
$$

where $A z \leq_{\mathcal{K}} b$ refers to a cone constraint. The three supported cone constraints are

### Linear Constraints

Simply $A z \leq b$ 

### Second-order Cone Constraints (SOCP)
Mathematically
$||A_{2:n} z - b_{2:n}||_2 \leq a_1^T z - b_1$
or in code
```python
np.linalg.norm(A[1:, :] @ z - b[1:]) <= A[0, :].T @ z - b[0]
```

### Exponential Cone Constraints

The exponential cone is defined as a 3 output matrix expression (the image of
the cone matrix is of dimension 3). We follow the convention from [JuMP.jl](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Exponential-Cone)

$$
K_\text{exp} = \lbrace (x, y, z) \in \mathbb{R}^3 ~~~~ : ~~~~ y e^{x / y} \leq z, ~~~~ y \geq 0 \rbrace
$$

for

$$A v - b = (x, y, z)$$

so $A \in \mathbb{R}^{n \times 3}$ and $b \in \mathbb{R}^3$.

## Solver selection

The underlying convex solver can be selected by passing a composite keyword argument

```python
sol = solve(
  ...,
  solver_settings = dict(solver="ecos") # for example, or "cosmo", "osqp", "mosek"
)
```

- `ECOS` - FREE - very fast and very general (used by default)
- `OSQP` - FREE - very fast, but only supports linear constraints
- `COSMO` - FREE - very general, but it tends to run very slowly
- `Mosek` - NOT FREE - extremely fast, but requires a (not free) license
  - requires: a Mosek license

# Particle (consensus/contingency) optimization

TODO

# Warm-start support

*Warm-start* in SCP MPC can refer to either
- warm-starting the SCP procedure through a good `X_prev, U_prev` - this is supported
- warm-starting the underlying convex solver - not supported

Warm-starting of the SCP procedure by providing a good `X_prev, U_prev` guess is supported and very much encouraged for good SCP performance!

Warm-starting of the underlying convex solver is currently not supported, as it does not lead to a noticeable
performance improvement on problems we tested the solver on.
