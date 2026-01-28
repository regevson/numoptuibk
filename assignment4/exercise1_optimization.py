# import plotille  # for plotting on the console, might be useful for debugging
import functools

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp

STARTS = [
    jnp.array([-4.0, 4.0]),
    jnp.array([0.0, 4.0]),
    jnp.array([2.0, 0.0]),
    jnp.array([-3.0, -1.0]),
    jnp.array([3.0, -1.0]),
    jnp.array([3.0, 3.0]),
    jnp.array([0, 0]),
]


def flatv(vector):
    # convert vector to flat vector of shap [n]
    v = vector.squeeze()
    assert_flat(v), f"you didn't pass a vector: {vector.shape}"
    return v


def rowv(vector):
    # convert vector to a rowvector of shape [1, n]
    return flatv(vector)[None, :]


def colv(vector):
    # convert vector to a column vector of shape [n, 1]
    return flatv(vector)[:, None]


def assert_flat(vector):
    # raise error if the vector is NOT flat.
    assert len(vector.shape) == 1


def himmelblau(xy):
    x, y = jnp.split(xy.flatten(), 2, axis=0)
    (val,) = (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2
    return val + 1.0


def booth(xy):
    x, y = jnp.split(xy.flatten(), 2, axis=0)
    (val,) = (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
    return val + 1.0


def yourfunction(xy):
    x, y = jnp.split(xy.flatten(), 2, axis=0)
    (val,) = jnp.sin(x) * jnp.cos(y) + (x - y) ** 2
    return val + 1.0


def mergexy(x, y, func):
    return func(jnp.hstack([x, y]))


def gradient_descent(func, x0, steps=10, alpha=0.01, log=True, minimum=None, maximum=None):
    losses = []
    xlog = []
    x = x0
    for i in range(steps):
        x -= alpha * jax.grad(fun=func)(x)
        if log:
            losses.append(func(x))
            xlog.append(x)
        if maximum is not None and any(x > maximum):
            break
        if minimum is not None and any(x < minimum):
            break
    if minimum is not None or maximum is not None:
        x = jnp.clip(x, minimum, maximum)
    return x, losses, xlog


def newton_raphson(func, x0, steps=10, alpha=0.5, log=True, minimum=None, maximum=None):
    losses = []
    xlog = []
    x = x0
    for i in range(steps):
        y, grad_x = jax.value_and_grad(func)(x)
        grad_x = jnp.where(grad_x == 0, 1e-10, grad_x)
        x = x - alpha * (y / grad_x)
        if log:
            losses.append(func(x))
            xlog.append(x)
        if maximum is not None and any(x > maximum):
            break
        if minimum is not None and any(x < minimum):
            break
    if minimum is not None or maximum is not None:
        x = jnp.clip(x, minimum, maximum)
    return x, losses, xlog


def contour_map(func, minimum=-5, maximum=5):
    X = onp.linspace(minimum, maximum, 100)
    Y = onp.linspace(minimum, maximum, 100)
    X, Y = onp.meshgrid(X, Y)
    xyfunc = functools.partial(mergexy, func=func)
    Z = jax.vmap(jax.vmap(xyfunc))(X, Y).squeeze()
    vmin = Z.flatten().min()
    vmax = Z.flatten().max()
    plt.contour(X, Y, Z, levels=onp.geomspace(vmin, vmax, 60), alpha=0.5)
    # plt.contour(X, Y, Z, levels=60, alpha=0.5)
    plt.grid()
    plt.colorbar()


def argmin_alpha(func, x, p, min_alpha=0.00001, max_alpha=0.1):
    alpha = max_alpha
    while func(x + alpha * p) > func(x):
        alpha *= 0.5
        if alpha < min_alpha:
            break
    return alpha


def bfgs(
    func,
    x0,
    steps=10,
    min_alpha=0.00001,
    max_alpha=0.1,
    log=True,
    minimum=None,
    maximum=None,
):
    losses = []
    xlog = []
    x = x0
    (xdim,) = x.squeeze().shape
    Binv = jnp.eye(xdim)
    for i in range(steps):
        grad_x = jax.grad(func)(x)

        p_flat = -Binv @ grad_x
        p = p_flat.reshape(-1, 1)
        x_vec = x.reshape(-1, 1)

        alpha = argmin_alpha(func, x, p_flat, min_alpha, max_alpha)

        s = alpha * p
        x_new = x_vec + s

        grad_new = jax.grad(func)(x_new.squeeze())
        y = (grad_new - grad_x).reshape(-1, 1)

        x = x_new.squeeze()

        sty = (s.T @ y)[0, 0]
        sty = jnp.where(sty == 0, 1e-10, sty)

        term1_scalar = sty + (y.T @ Binv @ y)[0, 0]
        term1 = (term1_scalar * (s @ s.T)) / (sty**2)

        term2 = ((Binv @ y) @ s.T + (s @ y.T) @ Binv) / sty

        Binv += term1 - term2

        if log:
            losses.append(func(x))
            xlog.append(x)
        if maximum is not None and any(x > maximum):
            break
        if minimum is not None and any(x < minimum):
            break
    if minimum is not None or maximum is not None:
        x = jnp.clip(x, minimum, maximum)
    return x, losses, xlog


if __name__ == "__main__":
    funcs_limits = [(himmelblau, 10), (booth, 5), (yourfunction, 5)]
    for func, limit in funcs_limits:
        # x0 = STARTS[limit % len(STARTS) - 1]
        # onp.random.seed(123) # fixing the seed (-> not random)
        # a random starting point:
        x0 = onp.random.uniform(-limit, limit, size=(2,))
        x_gd, losses_gd, xlog_gd = gradient_descent(func, x0, alpha=0.1)
        x_nr, losses_nr, xlog_nr = newton_raphson(func, x0, alpha=0.1)
        x_bfgs, losses_bfgs, xlog_bfgs = bfgs(func, x0, max_alpha=0.1)

        plt.subplot(2, 1, 1)
        contour_map(func, minimum=-limit, maximum=limit)

        def addplot(log, limit=limit, label=""):
            log = onp.vstack([x0] + log)
            log = onp.clip(log, -limit, limit)
            plt.plot(*log.T, marker="x", label=label)

        addplot(xlog_gd, label="GD")
        addplot(xlog_nr, label="NR")
        addplot(xlog_bfgs, label="BFGS")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(onp.vstack(losses_gd), label="GD")
        plt.plot(onp.vstack(losses_nr), label="NR")
        plt.plot(onp.vstack(losses_bfgs), label="BFGS")
        plt.legend()
        plt.yscale("log")
        plt.savefig(f"result_{func.__name__}.png", dpi=400)
        plt.clf()
