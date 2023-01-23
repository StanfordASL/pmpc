# `pmpc`
High-level Python Particle Sequential Convex Programming Model Predictive Control (SCP PMPC) interface.

This is non-linear dynamics finite horizon MPC solver with feasibility TODO

# Table of Contents 
- [`pmpc`](#pmpc)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
  - [Obtaining (a dynamically linked version of) Python](#obtaining-a-dynamically-linked-version-of-python)
  - [Obtaining Julia](#obtaining-julia)
  - [Obtaining `pyjulia`](#obtaining-pyjulia)
  - [Installation of this package](#installation-of-this-package)
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
  - [Consensus Optimization for Control under Uncertainty](#consensus-optimization-for-control-under-uncertainty)
  - [Non-convex Cost Example](#non-convex-cost-example)
  - [Arbitrary Constraints Example](#arbitrary-constraints-example)
  - [Convex solver selection](#convex-solver-selection)

# Installation

This package consists of the high level Python interface `pmpc` and the low level solver utilities `PMPC.jl`.

The best way to clone this repository is to do so with the low-level `PMPC.jl` module included
```bash
$ git clone --recursive https://github.com/StanfordASL/pmpc.git
```
or, alternatively
```bash
$ git clone https://github.com/StanfordASL/pmpc.git
$ cd pmpc
$ git submodule update --init
```

## Obtaining (a dynamically linked version of) Python 

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

## Obtaining Julia

You can download [Julia](https://julialang.org/), the programming language and interpreter, from [here](https://julialang.org/downloads/).

Make sure Julia is in your `PATH`.

## Obtaining `pyjulia`

Once you have the desired version of Python, install `pyjulia`
```bash
$ pip install julia
$ python3
>>> import julia
>>> julia.install()
...
>>> from julia import Main as jl
>>> import math
>>> assert (jl.sin(2.0) - math.sin(2.0)) < 1e-9
```

## Installation of this package

From the root of this project
```bash
$ pip install . # to install the pmpc Python module
``` 
to install the Python package `pmpc`.

From the root of this project
```bash
$ jl -e 'using Pkg; Pkg.develop(PackageSpec(path="PMPC.jl"))' 
```
to install the Julia core solver `PMPC.jl`.

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

and a quadratic cost
- `X_ref, Q` a reference and quadratic weight matrix for state cost
- `U_ref, R` a reference and quadratic weight matrix for control cost
The cost is given as 
$$J = \sum_{i=0}^N 
\frac{1}{2} (x^{(i+1)} - x_\text{ref}^{(i+1)}) Q^{(i)} (x^{(i+1)} - x_\text{ref}^{(i+1)}) + 
\frac{1}{2} (u^{(i)} - u_\text{ref}^{(i)}) R^{(i)} (u^{(i)} - u_\text{ref}^{(i)})$$

*Note: Initial state, x0, is assumed constant and thus does not feature in the cost.*

*Note: When handling controls, we'll always have np.shape(U) == (N, udim)*

*Note: When handling states, we'll have np.shape(X) == (N + 1, xdim) with x0 included at the beginning or np.shape(X) == (N, xdim) with x0 NOT INCLUDED. X[:, -1] always refers to the state N, whereas U[:, -1] always refers to control N - 1.*

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
$$J_\text{deviation} = \sum_{i=0}^N \frac{1}{2} 
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
- `method` - which solver type to use: `"cone"` for general cone support and `"qp"` for only quadratic programs
  - `"qp"` is slightly faster, but supports only linear inequalities
  - `"cone"` used by default

## Additional Dynamics Settings

- `X_prev` - previous state solution (guess), $x^{(i)} ~~ \forall i \in [1, \dots, N]$, `shape = (N, xdim)`
- `U_prev` - previous control solution (guess),  $u^{(i)} ~~ \forall i \in [0, \dots, N - 1]$, `shape = (N, udim)`
- `x_l` - state lower box constraints, $x^{(i)} ~~ \forall i \in [1, \dots, N]$, `shape = (N, xdim)`
- `x_u` - state upper box constraints, $x^{(i)} ~~ \forall i \in [1, \dots, N]$, `shape = (N, xdim)`
- `u_l` - control lower box constraints, $u^{(i)} ~~ \forall i \in [0, \dots, N - 1]$, `shape = (N, udim)`
- `u_u` - control upper box constraints, $u^{(i)} ~~ \forall i \in [0, \dots, N - 1]$, `shape = (N, udim)`

## Nonlinear Cost and Constraints

The solver supports custom arbitrary cost via each-SCP-iteration cost linearization and custom constraints via each-SCP-iteration constraint reformulation into any convex-cone constraint.

- `cost_fn` is an optional callable which allows specifying a custom cost, it
should take arbitrary `X`, `U` and return a tuple
    - `cx`, the linearization of the cost with respect to the state, `np.shape(cx) == (N, xdim) or cx is None`
    - `cu`, the linearization of the cost with respect to the controls, `np.shape(cu) == (N, udim) or cu is None`

I highly recommend using an auto-diff library to produce the linearizations to avoid unnecessary bugs.

- `extra_cstrs_fn` is an optional callable which returns **a list** of conic constraints given arbitrary `X, U`
  - a conic constraint $G z - h \in \mathcal{K}$, (e.g., $A z - b \leq 0$) consists of a tuple of 8 elements
    - `l` - the number of non-positive orthant constraints (an integer)
    - `q` - a list with the sizes of second order cone constraints (list of integers)
    - `e` - the number of exponential cone constraints (integer)
    - `G_left` - the G matrix referring to existing variables
    - `G_right` - the G matrix for new variables to introduce
    - `h` - the right hand side vector for the constraint
    - `c_left` - the additive linear cost augmentation for existing variables
    - `c_right` - the linear (minimization) cost for new variables to introduce

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

def cost_fn(X, U):
    # cost is np.sum((X - X_ref) ** 2)
    cx = 2 * (X - X_ref)
    cu = None
    return cx, cu
```

## Arbitrary Constraints Example

TODO

## Convex solver selection

TODO