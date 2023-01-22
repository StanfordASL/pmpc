# `pmpc`
High-level Python Particle Sequential Convex Programming Model Predictive Control (SCP PMPC) interface.

This is non-linear dynamics finite horizon MPC solver with feasibility TODO


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

---

# Basic Usage

The solver is capable of MPC consensus optimization for several system instantiations. For the basic usage, we'll focus on a single system MPC.

A basic MPC problem is defined using the dynamics and a quadratic cost

## Defining dynamics

- `x0` the initial state of shape $\mathbb{R}^{x}$
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

## Hyperparameters

The solver has two scalar hyperparamters, the dynamics linearization deviation penalty for states and controls
$$ 

# Advanced Usage

TODO

## Non-convex cost

TODO

## Arbitrary Constraints

TODO

## Convex solver selection

TODO
