import math, pdb, os, sys, pickle

import torch, numpy as np
from torch.autograd.functional import jacobian


def f(x, u, p):
    return car(x, u, p)


def fx(x, u, p):
    if x.ndim == 1:
        Jx = jacobian(lambda x: car(x, u, p), x)
    else:
        dims = tuple(i for i in range(x.ndim - 1))
        Jx = jacobian(lambda x: torch.sum(car(x, u, p), dims), x)
    xdim = x.shape[-1]
    Jx = Jx.reshape((xdim, -1, xdim)).transpose(0, 1).reshape(x.shape + (xdim,))
    return Jx


def fu(x, u, p):
    if u.ndim == 1:
        Ju = jacobian(lambda u: car(x, u, p), u)
    else:
        dims = tuple(i for i in range(u.ndim - 1))
        Ju = jacobian(lambda u: torch.sum(car(x, u, p), dims), u)
    xdim, udim = x.shape[-1], u.shape[-1]
    Ju = Ju.reshape((xdim, -1, udim)).transpose(0, 1).reshape(x.shape + (udim,))
    return Ju


def f_np(x, u, p):
    args = torch.as_tensor(x), torch.as_tensor(u), torch.as_tensor(p)
    return f(*args).cpu().detach().numpy()


def fx_np(x, u, p):
    args = torch.as_tensor(x), torch.as_tensor(u), torch.as_tensor(p)
    return fx(*args).cpu().detach().numpy()


def fu_np(x, u, p):
    args = torch.as_tensor(x), torch.as_tensor(u), torch.as_tensor(p)
    return fu(*args).cpu().detach().numpy()


def car(x, u, p):
    """
    unicycle car dynamics, 4 states, 2 actions
    x1: position x
    x2: position y
    x3: speed (local frame)
    x4: orientation angle

    u1: acceleration
    u2: turning speed (independent of velocity)
    """
    assert x.shape[-1] == 4 and u.shape[-1] == 2
    v_scale, w_scale, T = p[..., 0], p[..., 1], p[..., 2]
    eps = 1e-6
    u1, u2 = v_scale * u[..., 0], w_scale * -u[..., 1]
    u1 = u1 + torch.where(u1 >= 0.0, eps, -eps)
    u2 = u2 + torch.where(u2 >= 0.0, eps, -eps)

    x0, y0, v0, th0 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    xp1 = (
        x0
        + (
            u2 * torch.sin(T * u2 + th0) * v0
            + T * u1 * u2 * torch.sin(T * u2 + th0)
            + u1 * torch.cos(T * u2 + th0)
        )
        / u2 ** 2
        - (torch.sin(th0) * u2 * v0 + torch.cos(th0) * u1) / u2 ** 2
    )
    xp2 = (
        y0
        - (
            u2 * torch.cos(T * u2 + th0) * v0
            - u1 * torch.sin(T * u2 + th0)
            + T * u1 * u2 * torch.cos(T * u2 + th0)
        )
        / u2 ** 2
        + (torch.cos(th0) * u2 * v0 - torch.sin(th0) * u1) / u2 ** 2
    )
    xp3 = v0 + T * u1
    xp4 = T * u2 + th0
    xp = torch.stack([xp1, xp2, xp3, xp4], -1)
    return xp


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    dirname = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(dirname, ".."))

    import sysid
    import matplotlib.pyplot as plt

    with open("data/uni_1s.pkl", "rb") as fp:
        data = pickle.load(fp)
    idxs, u_old = [], np.array(math.inf)
    for (i, z) in enumerate(data):
        u = np.array(z["command"])
        if np.linalg.norm(u - u_old) > 0.0:
            idxs.append(i)
            u_old = u
    data = np.array(data)[np.array(idxs)]

    X = torch.stack([torch.tensor(z["state"]) for z in data])
    U = torch.stack([torch.tensor(z["command"]) for z in data])
    X, U, Xp = X[:-1, :], U[:-1, :], X[1:, :]

    # visualize ################################
    def visualize(X, label="x", **kw):
        for r in range(X.shape[-1]):
            plt.plot(X[:, r], label="%s%d" % (label, r + 1), **kw)
        plt.legend()

    l1, P1, fwd_fn1 = sysid.fit_params_dynamics(f, X, U, Xp, 2, max_it=10 ** 3)
    l2, P2, fwd_fn2 = sysid.fit_params_nn(
        X, U, Xp, 2, max_it=10 ** 4, reg=1e-3
    )

    sysid.visualize_error_dist(fwd_fn1, X, U, Xp)
    plt.title("dynamics")
    sysid.visualize_error_dist(fwd_fn2, X, U, Xp)
    plt.title("nn")

    plt.figure()
    visualize(Xp, label="x")
    visualize(fwd_fn2(X, U), label="x", ls="--")

    plt.draw_all()
    plt.pause(1e-1)

    pdb.set_trace()
