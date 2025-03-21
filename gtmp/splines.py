from functools import partial
from typing import Union, Tuple

import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import jit


A_CUBIC = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [-3, 3, -2, -1],
    [2, -2, 1, 1],
])


def isbool(x):
    return isinstance(x, bool) or (hasattr(x, "dtype") and (x.dtype == bool))


def errorif(cond, err=ValueError, msg=""):
    if cond:
        raise err(msg)


def asarray_inexact(x):
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
    c : ndarray, shape (k, m, ...)
        Polynomial coefficients, order `k` and `m` intervals.
    x : ndarray, shape (m+1,)
        Polynomial breakpoints. Must be sorted in either increasing or
        decreasing order.
    extrapolate : bool or 'periodic', optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. Default is True.

    """

    _c: jax.Array
    _x: jax.Array
    _extrapolate: Union[bool, str] = eqx.field(static=True)

    def __init__(
        self,
        c: jax.Array,
        x: jax.Array,
        extrapolate: Union[bool, str] = None,
    ):
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
        """Array of spline coefficients, shape(order, knots-1, ...)."""
        return self._c

    @property
    def x(self) -> jax.Array:
        """Array of knot values, shape(knots)."""
        return self._x

    @property
    def extrapolate(self) -> Union[bool, str]:
        """Whether to extrapolate beyond domain of known values."""
        return self._extrapolate

    @classmethod
    def construct_fast(
        cls,
        c: jax.Array,
        x: jax.Array,
        extrapolate: Union[bool, str] = None,
    ):

        self = object.__new__(cls)
        object.__setattr__(self, "_c", c)
        object.__setattr__(self, "_x", x)
        object.__setattr__(self, "_extrapolate", extrapolate)
        return self

    @partial(jit, static_argnames=("nu", "extrapolate"))
    def __call__(self, x: jax.Array, i: int, j: int, nu: int = 0, extrapolate: Union[bool, str] = None):
        """Evaluate the piecewise polynomial or its derivative.

        Parameters
        ----------
        x : array_like
            Points to evaluate the interpolant at.
        nu : int, optional
            Order of derivative to evaluate. Must be non-negative.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used.
            If None (default), use `self.extrapolate`.

        Returns
        -------
        y : array_like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.
        """
        if extrapolate is None:
            extrapolate = self.extrapolate
        x = asarray_inexact(x)
        x_shape, x_ndim = x.shape, x.ndim
        x = x.flatten()

        # With periodic extrapolation we map x to the segment
        # [self.x[0], self.x[-1]].
        if extrapolate == "periodic":
            x = self.x[0] + (x - self.x[0]) % (self.x[-1] - self.x[0])
            extrapolate = False

        t_i = jnp.clip(jnp.searchsorted(self.x, x, side="right"), 1, len(self.x) - 1)

        t = x - self.x[t_i - 1]
        c = self.c[:, t_i - 1, i, j]

        c = jnp.vectorize(lambda x: jnp.polyder(x, nu), signature="(n)->(m)")(c.T).T
        y = jnp.vectorize(jnp.polyval, signature="(n),()->()")(c.T, t).T

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
        path_id : array_like
            Path index.
        num_points : int, optional
            Number of points to evaluate the spline at. Default is 20.
        nu : int, optional
            Order of derivative to evaluate. Default is 0.

        Returns
        -------
        spline : array_like
            Spline values for the given path.
        """
        assert len(path_id) == self.x.shape[0]
        x = jnp.linspace(0, 1, num_points + 1)[:-1]
        dim = self.c.shape[2]

        def get_segment(t: int, i: int, j: int) -> jax.Array:

            c = self.c[:, t, None, :, i, j].repeat(num_points, axis=1)
            c = jnp.vectorize(lambda x: jnp.polyder(x, nu), signature="(n)->(m)")(c.T)
            y = jnp.vectorize(jnp.polyval, signature="(n),()->()")(c, x).T
            return y

        idx = jnp.arange(len(path_id) - 1, dtype=jnp.int32)
        cu, ne = path_id[:-1], path_id[1:]
        spline = jax.vmap(get_segment, in_axes=(0, 0, 0))(idx, cu, ne).reshape(-1, dim)
        return spline

    @partial(jit, static_argnames="nu")
    def get_spline(self, path_id: jax.Array, nu: int = 0) -> jax.Array:
        """Get the spline for a given path.

        Parameters
        ----------
        path_id : array_like
            Path index.
        num_points : int, optional
            Number of points to evaluate the spline at. Default is 20.
        nu : int, optional
            Order of derivative to evaluate. Default is 0.

        Returns
        -------
        spline : array_like
            Spline values for the given panum_segments,th.
        """
        assert len(path_id) == self.x.shape[0]

        def get_segment(t: int, i: int, j: int) -> jax.Array:
            return self.c[:, t, :, i, j]

        idx = jnp.arange(len(path_id) - 1, dtype=jnp.int32)
        cu, ne = path_id[:-1], path_id[1:]
        spline = jax.vmap(get_segment, in_axes=(0, 0, 0))(idx, cu, ne)
        return spline  # (M + 1, k, d)

    @partial(jit, static_argnames=("nu", "num_points"))
    def get_spline_grid_interpolation(self, num_points: int = 5, nu: int = 0) -> jax.Array:
        """Get the spline for a given path.

        Parameters
        ----------
        path_id : array_like
            Path index.
        num_points : int, optional
            Number of points to evaluate the spline at. Default is 20.
        nu : int, optional
            Order of derivative to evaluate. Default is 0.

        Returns
        -------
        spline : array_like
            Spline values for the given path.
        """
        x = jnp.linspace(0, 1, num_points + 1)[:-1]
        M, dim, N = self.c.shape[1:4]
        layer_idx = jnp.arange(N, dtype=jnp.int32)
        X, Y = jnp.meshgrid(layer_idx, layer_idx)
        pairwise_idx = jnp.stack([X.ravel(), Y.ravel()])
        s_id, t_id = pairwise_idx[1], pairwise_idx[0]

        def get_segment(t: int, i: int, j: int) -> jax.Array:
            c = self.c[:, t, None, :, i, j].repeat(num_points, axis=1)
            c = jnp.vectorize(lambda x: jnp.polyder(x, nu), signature="(n)->(m)")(c.T)
            y = jnp.vectorize(jnp.polyval, signature="(n),()->()")(c, x).T
            return y  # (num_points, dim)
        
        def get_segment_layers(t: int) -> jax.Array:
            points = jax.vmap(get_segment, in_axes=(None, 0, 0))(t, s_id, t_id)  # (N*N, num_points, dim)
            return points.reshape(N, N, num_points, dim)
        
        points_s_1 = jax.vmap(get_segment, in_axes=(None, None, 0))(0, 0, layer_idx)  # (N, num_points, dim)
        mid_idx = jnp.arange(1, M - 1, dtype=jnp.int32)
        points_layers = jax.vmap(get_segment_layers)(mid_idx)  # (M - 2, N, N, num_points, dim)
        points_final_g = jax.vmap(get_segment, in_axes=(None, 0, None))(M, layer_idx, 0)  # (N, num_points, dim)
        return points_s_1, points_layers, points_final_g

    def derivative(self, nu: int = 1):
        """Construct a new piecewise polynomial representing the derivative.

        Parameters
        ----------
        nu : int, optional
            Order of derivative to evaluate. Default is 1, i.e., compute the
            first derivative. If negative, the antiderivative is returned.

        Returns
        -------
        pp : PPoly
            Piecewise polynomial of order k2 = k - n representing the derivative
            of this polynomial.

        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.
        """
        if nu < 0:
            return self.antiderivative(-nu)

        if nu == 0:
            c2 = self.c.copy()
        else:
            c2 = jnp.vectorize(lambda x: jnp.polyder(x, nu), signature="(n)->(m)")(
                self.c.T
            ).T

        if c2.shape[0] == 0:
            # derivative of order 0 is zero
            c2 = jnp.zeros((1,) + c2.shape[1:], dtype=c2.dtype)

        return self.construct_fast(c2, self.x, self.extrapolate)

    def antiderivative(self, nu: int = 1):
        """Construct a new piecewise polynomial representing the antiderivative.

        Antiderivative is also the indefinite integral of the function,
        and derivative is its inverse operation.

        Parameters
        ----------
        nu : int, optional
            Order of antiderivative to evaluate. Default is 1, i.e., compute
            the first integral. If negative, the derivative is returned.

        Returns
        -------
        pp : PPoly
            Piecewise polynomial of order k2 = k + n representing
            the antiderivative of this polynomial.

        Notes
        -----
        The antiderivative returned by this function is continuous and
        continuously differentiable to order n-1, up to floating point
        rounding error.

        If antiderivative is computed and ``self.extrapolate='periodic'``,
        it will be set to False for the returned instance. This is done because
        the antiderivative is no longer periodic and its correct evaluation
        outside of the initially given x interval is difficult.
        """
        if nu <= 0:
            return self.derivative(-nu)

        if nu == 0:
            c2 = self.c.copy()
        else:
            c2 = self.c.copy()
            for _ in range(nu):
                c2 = jnp.vectorize(jnp.polyint, signature="(n)->(m)")(c2.T).T
                # need to patch up continuity
                dx = jnp.diff(self.x)
                z = jnp.vectorize(jnp.polyval, signature="(n),()->()")(c2.T, dx).T
                c2 = c2.at[-1, 1:].add(jnp.cumsum(z, axis=0)[:-1])

        if self.extrapolate == "periodic":
            extrapolate = False
        else:
            extrapolate = self.extrapolate

        return self.construct_fast(c2, self.x, extrapolate)



class LayerAkima1DInterpolator(LayerPPoly):
    """Akima interpolator.
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

        # compute akima spline coefficients
        y = jnp.concatenate(
            [jnp.repeat(q_s[None, ...], N, axis=1), q_l, jnp.repeat(q_g[None, ...], N, axis=1)], axis=0
        )  # (M + 2, N, D)
        dxr = dx.reshape([dx.shape[0]] + [1] * y.ndim)

        y_l, y_r = y[:-1], y[1:]
        dydx_l, dydx_r = dydx[:-1], dydx[1:]
        dydx_l = dydx_l.mean(axis=(-1, -2), keepdims=True)  # (M + 1, D, 1, 1)
        dydx_l = dydx_l.repeat(N, axis=-1).repeat(N, axis=-2)  # (M + 1, D, N, N)
        dydx_r = dydx_r.mean(axis=(-1, -2), keepdims=True)  # (M + 1, D, 1, 1)
        ai = y_l[:, :, None, :].repeat(N, axis=2)  # (M + 1, N, N, D)
        ai = jnp.moveaxis(ai, -1, 1)  # (M + 1, D, N, N)
        bi = dydx_l  # (M + 1, N, N, D)
        diff = (y_r[:, None, :, :] - y_l[:, :, None, :]) / dxr
        diff = jnp.moveaxis(diff, -1, 1)  # (M + 1, D, N, N)
        ci = 3 * diff - 2 * dydx_l - dydx_r  # (M + 1, D, N, N)
        ci = ci / dxr
        di = -2 * diff + dydx_l + dydx_r  # (M + 1, D, N, N)
        di = di / jnp.square(dxr)
        c = jnp.stack([di, ci, bi, ai], axis=-1)  # (M + 1, D, N, N, 4)
        # handle non-uniform spacing
        c = c / (dx[:, None, None, None, None] ** jnp.arange(4)[::-1])
        c = jnp.moveaxis(c, -1, 0)  # (4, M + 1, D, N, N)
        super().__init__(c, x, extrapolate=extrapolate)


def df_akima(x: jax.Array, q_s: jax.Array, q_l: jax.Array, q_g: jax.Array) -> jax.Array:
    dx = jnp.diff(x)
    dim = q_s.shape[-1]
    M, N = q_l.shape[0], q_l.shape[1]

    # determine slopes between breakpoints
    mask = dx == 0
    dx = jnp.where(mask, 1, dx)
    dxi = jnp.where(mask, 0.0, 1 / dx)
    m = jnp.empty((M + 5, N, N, dim))  # two additional points on the left and right

    diff_s = q_l[0] - q_s  # (N, D)
    sources, targets = q_l[:-1], q_l[1:]
    diff_l = targets[:, None, :, :] - sources[:, :, None, :]  # (M - 1, N, N, D)
    diff_g = q_g - q_l[-1]  # (N, D)

    m = m.at[2].set(diff_s * dxi[0, None])
    m = m.at[3:-3].set(diff_l * dxi[1:-1, None, None, None])
    m = m.at[-3].set(diff_g * dxi[-1, None])

    # add two additional points on the left
    m = m.at[1].set(2.0 * m[2] - m[3])
    m = m.at[0].set(2.0 * m[1] - m[2])
    # and on the right
    m = m.at[-2].set(2.0 * m[-3] - m[-4])
    m = m.at[-1].set(2.0 * m[-2] - m[-3])

    # df = derivative of f at x (spline derivatives)
    m_mean = m.mean(axis=(-2, -3), keepdims=True)  # (M + 5, D)
    dm = jnp.abs(m_mean[1:] - m[:-1])  # (M + 4, N, N, D)
    pm = jnp.abs(m_mean[1:] + m[:-1])  # (M + 4, N, N, D)
    f1 = dm[2:] + 0.5 * pm[2:]  # (M + 2, N, N, D)
    f2 = dm[:-2] + 0.5 * pm[:-2]  # (M + 2, N, N, D)
    m2 = m[1:-2]  # (M + 2, N, N, D)
    m3 = m[2:-1]  # (M + 2, N, N, D)
    f12 = f1 + f2
    mask = f12 > 1e-9 * jnp.max(f12, initial=-jnp.inf)
    df = (f1 * m2 + f2 * m3) / jnp.where(mask, f12, 1.0)
    df = jnp.where(mask, df, 0.5 * (m[3:] + m[:-3]))
    return df  # (M + 2, N, N, D)
