"""Akima spline interpolation for smooth trajectory generation in GTMP.

Implements LayerPPoly (piecewise polynomial) and LayerAkima1DInterpolator
for constructing and evaluating smooth splines over dream point layers.
"""
from functools import partial
from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import jit


def _horner_eval(c: jax.Array, x: jax.Array) -> jax.Array:
    """Evaluate polynomial using Horner's method.

    Parameters
    ----------
    c : jax.Array, shape (..., k)
        Polynomial coefficients in descending order.
    x : jax.Array, shape (...)
        Points to evaluate at.

    Returns
    -------
    jax.Array
        Polynomial values.
    """
    result = c[..., 0]
    for i in range(1, c.shape[-1]):
        result = result * x + c[..., i]
    return result


def _poly_deriv_cubic(c: jax.Array, nu: int = 1) -> jax.Array:
    """Compute derivative coefficients for cubic polynomial.

    For a cubic polynomial with coefficients [a, b, c, d] (degree 3),
    the first derivative has coefficients [3a, 2b, c].

    Parameters
    ----------
    c : jax.Array, shape (..., 4)
        Cubic polynomial coefficients.
    nu : int
        Order of derivative.

    Returns
    -------
    jax.Array
        Derivative coefficients.
    """
    if nu == 0:
        return c
    # First derivative: [3a, 2b, c]
    c1 = jnp.stack([3 * c[..., 0], 2 * c[..., 1], c[..., 2]], axis=-1)
    if nu == 1:
        return c1
    # Second derivative: [6a, 2b]
    c2 = jnp.stack([6 * c[..., 0], 2 * c[..., 1]], axis=-1)
    if nu == 2:
        return c2
    # Third derivative: [6a]
    return 6 * c[..., 0:1]


def asarray_inexact(x):
    """Convert to inexact (float) JAX array."""
    x = jnp.asarray(x)
    dtype = x.dtype
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.result_type(x, jnp.array(1.0))
    return x.astype(dtype)


class LayerPPoly(eqx.Module):
    """Piecewise polynomial in terms of coefficients and breakpoints.

    The polynomial between ``x[i]`` and ``x[i + 1]`` is written in the
    local power basis::

        S = sum(c[m, i] * (xp - x[i])**(k-m) for m in range(k+1))

    where ``k`` is the degree of the polynomial.

    Parameters
    ----------
    c : jax.Array, shape (k, m, ...)
        Polynomial coefficients, order `k` and `m` intervals.
    x : jax.Array, shape (m+1,)
        Polynomial breakpoints.
    extrapolate : bool or 'periodic'
        Extrapolation behavior.
    """

    _c: jax.Array
    _x: jax.Array
    _extrapolate: Union[bool, str] = eqx.field(static=True)

    def __init__(self, c: jax.Array, x: jax.Array, extrapolate: Union[bool, str] = None):
        c = asarray_inexact(c)
        x = asarray_inexact(x)
        if extrapolate is None:
            extrapolate = True
        elif extrapolate != "periodic":
            extrapolate = bool(extrapolate)
        self._extrapolate = extrapolate
        self._x = x
        self._c = c

    @property
    def c(self) -> jax.Array:
        """Spline coefficients, shape (order, knots-1, ...)."""
        return self._c

    @property
    def x(self) -> jax.Array:
        """Knot values, shape (knots,)."""
        return self._x

    @property
    def extrapolate(self) -> Union[bool, str]:
        """Extrapolation mode."""
        return self._extrapolate

    @classmethod
    def construct_fast(cls, c: jax.Array, x: jax.Array, extrapolate: Union[bool, str] = None):
        """Construct without input validation (fast path)."""
        self = object.__new__(cls)
        object.__setattr__(self, "_c", c)
        object.__setattr__(self, "_x", x)
        object.__setattr__(self, "_extrapolate", extrapolate)
        return self

    @partial(jit, static_argnames=("nu", "extrapolate"))
    def __call__(self, x: jax.Array, i: int, j: int, nu: int = 0, extrapolate: Union[bool, str] = None):
        """Evaluate the piecewise polynomial or its derivative."""
        if extrapolate is None:
            extrapolate = self.extrapolate
        x = asarray_inexact(x)
        x_shape = x.shape
        x = x.flatten()

        if extrapolate == "periodic":
            x = self.x[0] + (x - self.x[0]) % (self.x[-1] - self.x[0])
            extrapolate = False

        t_i = jnp.clip(jnp.searchsorted(self.x, x, side="right"), 1, len(self.x) - 1)
        t = x - self.x[t_i - 1]
        c = self.c[:, t_i - 1, i, j]  # (order, num_points, ...)

        # Derivative + evaluation using direct operations
        c_t = c.T  # (num_points, ..., order)
        if nu > 0:
            c_t = _poly_deriv_cubic(c_t, nu)
        y = _horner_eval(c_t, t).T

        y = y.reshape(x_shape + self.c.shape[4:])

        if not extrapolate:
            mask = jnp.logical_or(x > self.x[-1], x < self.x[0])
            y = jnp.where(mask, jnp.nan, y.T).T

        return y

    @partial(jit, static_argnames=("nu", "num_points"))
    def get_spline_interpolation(self, path_id: jax.Array, num_points: int = 5, nu: int = 0) -> jax.Array:
        """Get the spline for a given path.

        Parameters
        ----------
        path_id : jax.Array
            Path index through the layers.
        num_points : int
            Number of evaluation points per segment.
        nu : int
            Derivative order.

        Returns
        -------
        jax.Array
            Spline values along the path.
        """
        assert len(path_id) == self.x.shape[0]
        x = jnp.linspace(0, 1, num_points + 1)[:-1]
        dim = self.c.shape[2]

        def get_segment(t: int, i: int, j: int) -> jax.Array:
            c = self.c[:, t, :, i, j]  # (order, dim)
            c_expanded = jnp.broadcast_to(c[:, None, :], (c.shape[0], num_points, c.shape[1]))
            c_t = c_expanded.transpose(1, 2, 0)  # (num_points, dim, order)
            if nu > 0:
                c_t = _poly_deriv_cubic(c_t, nu)
            y = _horner_eval(c_t, x[:, None])  # (num_points, dim)
            return y

        idx = jnp.arange(len(path_id) - 1, dtype=jnp.int32)
        cu, ne = path_id[:-1], path_id[1:]
        spline = jax.vmap(get_segment, in_axes=(0, 0, 0))(idx, cu, ne).reshape(-1, dim)
        return spline

    @partial(jit, static_argnames="nu")
    def get_spline(self, path_id: jax.Array, nu: int = 0) -> jax.Array:
        """Get raw spline coefficients for a given path."""
        assert len(path_id) == self.x.shape[0]

        def get_segment(t: int, i: int, j: int) -> jax.Array:
            return self.c[:, t, :, i, j]

        idx = jnp.arange(len(path_id) - 1, dtype=jnp.int32)
        cu, ne = path_id[:-1], path_id[1:]
        return jax.vmap(get_segment, in_axes=(0, 0, 0))(idx, cu, ne)

    @partial(jit, static_argnames=("nu", "num_points"))
    def get_spline_grid_interpolation(self, num_points: int = 5, nu: int = 0) -> jax.Array:
        """Evaluate splines over all source→target combinations per layer.

        Returns
        -------
        tuple of (points_s_1, points_layers, points_final_g)
        """
        x = jnp.linspace(0, 1, num_points + 1)[:-1]
        M, dim, N = self.c.shape[1:4]
        layer_idx = jnp.arange(N, dtype=jnp.int32)
        X, Y = jnp.meshgrid(layer_idx, layer_idx)
        s_id, t_id = Y.ravel(), X.ravel()

        def get_segment(t: int, i: int, j: int) -> jax.Array:
            c = self.c[:, t, :, i, j]  # (order, dim)
            c_expanded = jnp.broadcast_to(c[:, None, :], (c.shape[0], num_points, c.shape[1]))
            c_t = c_expanded.transpose(1, 2, 0)  # (num_points, dim, order)
            if nu > 0:
                c_t = _poly_deriv_cubic(c_t, nu)
            y = _horner_eval(c_t, x[:, None])  # (num_points, dim)
            return y

        def get_segment_layers(t: int) -> jax.Array:
            points = jax.vmap(get_segment, in_axes=(None, 0, 0))(t, s_id, t_id)
            return points.reshape(N, N, num_points, dim)

        points_s_1 = jax.vmap(get_segment, in_axes=(None, None, 0))(0, 0, layer_idx)
        mid_idx = jnp.arange(1, M - 1, dtype=jnp.int32)
        points_layers = jax.vmap(get_segment_layers)(mid_idx)
        points_final_g = jax.vmap(get_segment, in_axes=(None, 0, None))(M, layer_idx, 0)
        return points_s_1, points_layers, points_final_g

    def derivative(self, nu: int = 1):
        """Construct piecewise polynomial representing the derivative."""
        if nu < 0:
            return self.antiderivative(-nu)
        if nu == 0:
            c2 = self.c
        else:
            c2 = jnp.vectorize(lambda x: jnp.polyder(x, nu), signature="(n)->(m)")(self.c.T).T

        if c2.shape[0] == 0:
            c2 = jnp.zeros((1,) + c2.shape[1:], dtype=c2.dtype)
        return self.construct_fast(c2, self.x, self.extrapolate)

    def antiderivative(self, nu: int = 1):
        """Construct piecewise polynomial representing the antiderivative."""
        if nu <= 0:
            return self.derivative(-nu)
        if nu == 0:
            c2 = self.c
        else:
            c2 = self.c
            for _ in range(nu):
                c2 = jnp.vectorize(jnp.polyint, signature="(n)->(m)")(c2.T).T
                dx = jnp.diff(self.x)
                z = jnp.vectorize(jnp.polyval, signature="(n),()->()")(c2.T, dx).T
                c2 = c2.at[-1, 1:].add(jnp.cumsum(z, axis=0)[:-1])

        if self.extrapolate == "periodic":
            extrapolate = False
        else:
            extrapolate = self.extrapolate
        return self.construct_fast(c2, self.x, extrapolate)


class LayerAkima1DInterpolator(LayerPPoly):
    """Akima interpolator for smooth trajectory generation.

    Constructs piecewise cubic polynomials through dream points with
    continuous first derivatives, using the modified Akima method.
    """

    def __init__(
        self,
        x: jax.Array,
        q_s: jax.Array,
        q_l: jax.Array,
        q_g: jax.Array,
        extrapolate: Union[bool, str] = None,
    ):
        q_s = jnp.atleast_2d(q_s)
        q_g = jnp.atleast_2d(q_g)
        M, N = q_l.shape[0], q_l.shape[1]
        dydx = df_akima(x, q_s, q_l, q_g)  # (M + 2, N, N, D)
        dydx = jnp.moveaxis(dydx, -1, 1)  # (M + 2, D, N, N)
        dx = jnp.diff(x)  # (M + 1,)

        # Compute Akima spline coefficients
        y = jnp.concatenate(
            [
                jnp.broadcast_to(q_s[None, ...], (1, N, q_s.shape[-1])),
                q_l,
                jnp.broadcast_to(q_g[None, ...], (1, N, q_g.shape[-1])),
            ],
            axis=0,
        )  # (M + 2, N, D)
        dxr = dx.reshape([dx.shape[0]] + [1] * y.ndim)

        y_l, y_r = y[:-1], y[1:]
        dydx_l, dydx_r = dydx[:-1] * dxr, dydx[1:] * dxr

        # Use broadcasting for derivative averaging across source/target pairs
        dydx_l_mean = dydx_l.mean(axis=(-1, -2), keepdims=True)  # (M+1, D, 1, 1)
        dydx_r_mean = dydx_r.mean(axis=(-1, -2), keepdims=True)  # (M+1, D, 1, 1)

        ai = y_l[:, :, None, :].repeat(N, axis=2)  # (M+1, N, N, D)
        ai = jnp.moveaxis(ai, -1, 1)  # (M+1, D, N, N)
        diff = (y_r[:, None, :, :] - y_l[:, :, None, :]) / dxr
        diff = jnp.moveaxis(diff, -1, 1)  # (M+1, D, N, N)
        # Broadcast mean derivatives to full (M+1, D, N, N) shape
        bi = jnp.broadcast_to(dydx_l_mean, ai.shape)
        ci = 3 * diff - 2 * dydx_l_mean - dydx_r_mean
        di = -2 * diff + dydx_l_mean + dydx_r_mean
        c = jnp.stack([di, ci, bi, ai], axis=-1)  # (M+1, D, N, N, 4)
        # Handle non-uniform spacing
        c = c / (dx[:, None, None, None, None] ** jnp.arange(4)[::-1])
        c = jnp.moveaxis(c, -1, 0)  # (4, M+1, D, N, N)
        super().__init__(c, x, extrapolate=extrapolate)


def df_akima(x: jax.Array, q_s: jax.Array, q_l: jax.Array, q_g: jax.Array) -> jax.Array:
    """Compute Akima spline derivatives at breakpoints.

    Parameters
    ----------
    x : jax.Array, shape (M+2,)
        Breakpoint positions.
    q_s : jax.Array, shape (N_s, D)
        Start configuration(s).
    q_l : jax.Array, shape (M, N, D)
        Layer dream points.
    q_g : jax.Array, shape (N_g, D)
        Goal configuration(s).

    Returns
    -------
    jax.Array, shape (M+2, N, N, D)
        Derivatives at each breakpoint.
    """
    dx = jnp.diff(x)
    dim = q_s.shape[-1]
    M, N = q_l.shape[0], q_l.shape[1]

    # Determine slopes between breakpoints
    mask = dx == 0
    dx_safe = jnp.where(mask, 1, dx)
    dxi = jnp.where(mask, 0.0, 1 / dx_safe)

    m = jnp.zeros((M + 5, N, N, dim))

    diff_s = q_l[0] - q_s  # (N, D)
    sources, targets = q_l[:-1], q_l[1:]
    diff_l = targets[:, None, :, :] - sources[:, :, None, :]  # (M-1, N, N, D)
    diff_g = q_g - q_l[-1]  # (N, D)

    m = m.at[2].set(diff_s * dxi[0, None])
    m = m.at[3:-3].set(diff_l * dxi[1:-1, None, None, None])
    m = m.at[-3].set(diff_g * dxi[-1, None])

    # Add two additional points on each side (Akima boundary condition)
    m = m.at[1].set(2.0 * m[2] - m[3])
    m = m.at[0].set(2.0 * m[1] - m[2])
    m = m.at[-2].set(2.0 * m[-3] - m[-4])
    m = m.at[-1].set(2.0 * m[-2] - m[-3])

    # Compute derivatives using modified Akima weights
    m_mean = m.mean(axis=(-2, -3), keepdims=True)
    dm = jnp.abs(m_mean[1:] - m[:-1])
    pm = jnp.abs(m_mean[1:] + m[:-1])
    f1 = dm[2:] + 0.5 * pm[2:]
    f2 = dm[:-2] + 0.5 * pm[:-2]
    m2 = m[1:-2]
    m3 = m[2:-1]
    f12 = f1 + f2
    mask = f12 > 1e-9 * jnp.max(f12, initial=-jnp.inf)
    df = (f1 * m2 + f2 * m3) / jnp.where(mask, f12, 1.0)
    df = jnp.where(mask, df, 0.5 * (m[3:] + m[:-3]))
    return df  # (M + 2, N, N, D)
