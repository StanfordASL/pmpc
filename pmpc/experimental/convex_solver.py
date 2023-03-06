import os
from enum import Enum
from typing import NamedTuple
from jfi import init

jaxm = init()

from jaxopt import base


class LineSearch(Enum):
    backtracking = 0
    scan = 1
    binary_search = 2


class ConvexState(NamedTuple):
    best_params: jaxm.jax.Array
    best_loss: jaxm.jax.Array
    value: jaxm.jax.Array


class ConvexSolver(base.IterativeSolver):
    def __init__(
        self,
        fun,
        maxiter=100,
        tol=1e-3,
        verbose=False,
        jit=True,
        maxls=20,
        min_stepsize=1e-7,
        max_stepsize=1e2,
        reg0=1e-6,
        linesearch="scan",
    ):
        self.maxiter, self.maxls = maxiter, maxls
        self.min_stepsize, self.max_stepsize = min_stepsize, max_stepsize
        self.tol, self.reg0 = tol, reg0
        self.verbose = verbose
        assert linesearch.lower() in ["scan", "backtracking", "binary_search"]
        if linesearch.lower() == "scan":
            self.linesearch = LineSearch.scan
        elif linesearch.lower() == "backtracking":
            self.linesearch = LineSearch.backtracking
        elif linesearch.lower() == "binary_search":
            self.linesearch = LineSearch.binary_search

        self.f_fn = fun

        def g_fn(params, *args, **kw):
            g = jaxm.grad(self.f_fn)(params, *args, **kw)
            return g.reshape((params.size,))

        def h_fn(params, *args, **kw):
            H = jaxm.hessian(self.f_fn)(params, *args, **kw)
            return H.reshape((params.size, params.size))

        self.g_fn, self.h_fn = g_fn, h_fn

        if jit:
            self.f_fn = jaxm.jit(self.f_fn)
            self.g_fn = jaxm.jit(self.g_fn)
            self.h_fn = jaxm.jit(self.h_fn)

    def init_state(self, params, *args, **kw):
        best_loss = self.f_fn(params, *args, **kw)
        return ConvexState(params, best_loss, best_loss)

    def update(self, params, state, *args, **kw):
        g, H = self.g_fn(params, *args, **kw), self.h_fn(params, *args, **kw)
        # dp = -jaxm.linalg.solve(H + self.reg0 * jaxm.eye(H.shape[-1]), g).reshape(params.shape)
        dp = -jaxm.scipy.linalg.cho_solve(
            jaxm.scipy.linalg.cho_factor(H + self.reg0 * jaxm.eye(H.shape[-1])), g
        ).reshape(params.shape)
        if self.linesearch == LineSearch.scan:
            lower_ls = max(round(0.7 * self.maxls), 1)
            bets_low = jaxm.logspace(jaxm.log10(self.min_stepsize), 0.0, lower_ls)
            bets_up = jaxm.logspace(0.0, jaxm.log10(self.max_stepsize), self.maxls - lower_ls)[1:]
            bets = jaxm.cat([bets_low, bets_up], -1)
            losses = jaxm.jax.vmap(lambda bet: self.f_fn(params + bet * dp, *args, **kw))(bets)
            losses = jaxm.where(jaxm.isnan(losses), jaxm.inf, losses)
            idx = jaxm.argmin(losses)
            bet, new_loss = bets[idx], losses[idx]
            new_params = params + bet * dp
        elif self.linesearch == LineSearch.backtracking:

            def cond_fn(step):
                step_not_too_small = step >= self.min_stepsize
                not_better_loss = self.f_fn(params + step * dp, *args, **kw) > state.best_loss
                return jaxm.logical_and(step_not_too_small, not_better_loss)

            def body_fn(step):
                return step * 0.7

            step_size = jaxm.jax.lax.while_loop(cond_fn, body_fn, jaxm.ones(()))
            new_params = params + step_size * dp
            new_loss = self.f_fn(new_params, *args, **kw)
        elif self.linesearch == LineSearch.binary_search:

            def body_fn(i, val):
                betl, betr, fl, fr = val
                betm = (betl + betr) / 2.0
                fm = self.f_fn(params + betm * dp, *args, **kw)
                cond = fl < fr
                betr, betl = jaxm.where(cond, betm, betr), jaxm.where(cond, betl, betm)
                fr, fl = jaxm.where(cond, fm, fr), jaxm.where(cond, fl, fm)
                return (betl, betr, fl, fr)

            betl, betr = self.min_stepsize * 1e2, 1e0
            fl = self.f_fn(params + betl * dp, *args, **kw)
            fr = self.f_fn(params + betr * dp, *args, **kw)
            betl, betr, fl, fr = jaxm.jax.lax.fori_loop(
                0, jaxm.maximum(self.maxls - 2, 1), body_fn, (betl, betr, fl, fr)
            )
            cond = fl < fr
            new_params = jaxm.where(cond, params + betl * dp, params + betr * dp)
            new_loss = jaxm.where(cond, fl, fr)

        best_params = jaxm.where(new_loss <= state.best_loss, new_params, state.best_params)
        best_loss = jaxm.where(new_loss <= state.best_loss, new_loss, state.best_loss)
        state = ConvexState(best_params, best_loss, best_loss)
        new_params = state.best_params
        return new_params, state
