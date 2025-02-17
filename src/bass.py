from collections.abc import Callable, Sequence
import warnings

import numpy as np
from numpy.typing import NDArray
from numpy.polynomial.hermite import hermgauss
from scipy.special import ndtr, ndtri
from scipy.interpolate import PchipInterpolator, CubicSpline

from .data import Marginal


EPS = np.finfo(np.float64).eps
FloatArray = NDArray[np.float64]
Floats = FloatArray | float


def conv_heat_gauss_hermite(
    t: Floats,
    func: Callable[[Floats], Floats],
    n_quad: int = 20,
) -> Callable[[Floats], FloatArray]:
    nodes, weights = hermgauss(n_quad)

    def f(x):
        x_b, t_b = np.broadcast_arrays(x, t)
        target_shape = (-1, *(1,) * x_b.ndim)
        x_b, t_b = x_b[None], t_b[None]

        scale = np.sqrt(2 * t_b + EPS)
        nodes_scaled = nodes.reshape(target_shape) * scale
        weights_reshaped = weights.reshape(target_shape)

        y = x_b - nodes_scaled
        func_vals = func(y)
        return np.sum(func_vals * weights_reshaped / np.sqrt(np.pi), axis=0)

    return f


class AuxCDF:
    def __init__(
        self,
        grid: FloatArray,
        vals: FloatArray,
    ):
        dg = np.diff(grid, prepend=0).clip(EPS)
        dg[0] = grid[0]
        grid = np.cumsum(dg)

        dv = np.diff(vals, prepend=0).clip(EPS)
        dv[0] = vals[0]
        vals = np.cumsum(dv)

        self._w_grid = grid
        self._cdf = PchipInterpolator(grid, vals)
        self._icdf = PchipInterpolator(vals, grid)

    @classmethod
    def build_initial_approx(
        cls,
        m1: Marginal,
        m2: Marginal,
        grid_points: int = 300,
        interp_bounds_stds: float = 5,
    ):
        assert interp_bounds_stds < -ndtri(EPS)

        raw_w_grid = np.linspace(-interp_bounds_stds, interp_bounds_stds, grid_points)
        u_grid = ndtr(raw_w_grid)

        integrand = np.sqrt(
            m2.dicdf(u_grid) / (m1.iicdf(u_grid) - m2.iicdf(u_grid) + EPS)
        )
        integral = CubicSpline(u_grid, integrand).antiderivative()
        icdf_vals = np.sqrt((m2.tenor - m1.tenor) / 2) * (
            integral(u_grid) - integral(1 / 2)
        )

        w_grid = raw_w_grid * np.sqrt(m1.tenor)
        vals = PchipInterpolator(icdf_vals, u_grid)(w_grid)
        return cls(w_grid, vals)

    @property
    def grid(self):
        return np.copy(self._w_grid)

    def __call__(self, w):
        return self._cdf(w)

    def icdf(self, u):
        return self._icdf(u)


def apply_operator(cdf: AuxCDF, m1: Marginal, m2: Marginal) -> AuxCDF:
    dt = m2.tenor - m1.tenor
    assert dt > 0

    kf = conv_heat_gauss_hermite(dt, cdf)
    fkf = lambda u: m2.icdf(kf(u))
    kfkf = conv_heat_gauss_hermite(dt, fkf)
    fkfkf = lambda x: m1.cdf(kfkf(x))

    new_vals = fkfkf(cdf.grid)
    new_cdf = AuxCDF(cdf.grid, new_vals)
    return new_cdf


def solve_fixed_point(
    m1: Marginal,
    m2: Marginal,
    l_inf_tol: float = 1e-3,
    max_iters: int = 100,
    grid_points: int = 300,
    interp_bounds_stds: float = 5,
) -> AuxCDF:

    l_inf = np.inf
    cdf = AuxCDF.build_initial_approx(m1, m2, grid_points, interp_bounds_stds)
    for _ in range(max_iters):
        new_cdf = apply_operator(cdf, m1, m2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            l_inf = np.abs(new_cdf(cdf.grid) - cdf(cdf.grid)).max()
        cdf = new_cdf

        if l_inf < l_inf_tol:
            break
    else:
        warnings.warn(
            f"Maximum number of iterations reached, error = {l_inf:.1e}", RuntimeWarning
        )

    return cdf


class BassLV:
    def __init__(
        self,
        gauss_hermite_quad_points: int = 20,
        fixed_point_solve_tol: float = 1e-3,
        fixed_point_solve_max_iters: int = 100,
        aux_cdf_grid_points: int = 100,
        aux_cdf_grid_bound_stds: float = 5,
    ):
        self._n_quad = gauss_hermite_quad_points
        self._tol = fixed_point_solve_tol
        self._max_iters = fixed_point_solve_max_iters
        self._aux_grid_sz = aux_cdf_grid_points
        self._aux_bound = aux_cdf_grid_bound_stds

    def build(self, marginals: Sequence[Marginal]):

        def f_01(t: Floats, w: Floats) -> FloatArray:
            conv_f = conv_heat_gauss_hermite(
                marginals[0].tenor - t,
                lambda x: marginals[0].icdf(ndtr(x / np.sqrt(marginals[0].tenor))),
                n_quad=self._n_quad,
            )
            return conv_f(w)

        self._funcs = [f_01]
        self._marginals = marginals
        self._aux_cdfs = []
        for i in range(len(marginals) - 1):
            m1, m2 = marginals[i], marginals[i + 1]
            cdf = solve_fixed_point(
                m1, m2, self._tol, self._max_iters, self._aux_grid_sz, self._aux_bound
            )
            kf = conv_heat_gauss_hermite(m2.tenor - m1.tenor, cdf)
            fkf = lambda u: m2.icdf(kf(u))
            kfkf = lambda t: conv_heat_gauss_hermite(m2.tenor - t, fkf)

            def func(t: Floats, w: Floats) -> FloatArray:
                return kfkf(t)(w)

            self._funcs.append(func)
            self._aux_cdfs.append(cdf)

        self._aux_cdfs.append(None)

        return self

    def sample(
        self, t: FloatArray, n_paths: int, rng: np.random.Generator | None = None
    ) -> FloatArray:  # of shape (n_paths, len(t))
        assert np.ndim(t) == 1

        rng = rng or np.random.default_rng()
        dB_t = rng.normal(size=(n_paths, len(t))) * np.sqrt(np.diff(t, prepend=0))
        B_t = np.cumsum(dB_t, axis=1)
        S_t = np.full((n_paths, len(t)), np.nan)

        t_lb = 0
        B_start = 0
        W_start = 0
        for func, m, aux_cdf in zip(self._funcs, self._marginals, self._aux_cdfs):

            # retrieve relevant times and B_t values
            fr = np.searchsorted(t, t_lb, "left")
            to = np.searchsorted(t, m.tenor, "right")
            t_cur = t[fr:to]
            B_t_cur = B_t[:, fr:to]

            W_t_cur = W_start + B_t_cur - B_start
            S_t[:, fr:to] = func(t_cur, W_t_cur)

            if to == len(t):
                break

            # Brownian bridge
            t1, t_b, t2 = t[to - 1], m.tenor, t[to]
            B1, B2 = B_t[:, to - 1], B_t[:, to]
            mean = B1 * (t2 - t_b) / (t2 - t1 + EPS) + B2 * (t_b - t1) / (t2 - t1 + EPS)
            std = np.sqrt((t2 - t_b) * (t_b - t1) / (t2 - t1 + EPS))
            B_start = rng.normal(loc=mean, scale=std)[:, None]

            W_start = aux_cdf.icdf(m.cdf(func(t_b, B_start)))
            t_lb = m.tenor

        return S_t


def build_lv_model(
    marginals: Sequence[Marginal],
    gauss_hermite_quad_points: int = 20,
    fixed_point_solve_tol: float = 1e-3,
    fixed_point_solve_max_iters: int = 100,
) -> Callable[[Floats, Floats], Floats]:

    def f_01(t: Floats, w: Floats) -> FloatArray:
        conv_f = conv_heat_gauss_hermite(
            marginals[0].tenor - t,
            lambda x: marginals[0].icdf(ndtr(x / np.sqrt(marginals[0].tenor))),
            n_quad=gauss_hermite_quad_points,
        )
        return conv_f(w)

    funcs = [f_01]
    borders = np.array([marginals[0].tenor])
    for i in range(len(marginals) - 1):
        m1, m2 = marginals[i], marginals[i + 1]
        cdf = solve_fixed_point(
            m1, m2, fixed_point_solve_tol, fixed_point_solve_max_iters
        )
        kf = conv_heat_gauss_hermite(m2.tenor - m1.tenor, cdf)
        fkf = lambda u: m2.icdf(kf(u))
        kfkf = lambda t: conv_heat_gauss_hermite(m2.tenor - t, fkf)

        def func(t: Floats, w: Floats) -> FloatArray:
            return kfkf(t)(w)

        funcs.append(func)
        borders = np.append(borders, m2.tenor)

    def markov_func(t: Floats, w: Floats) -> Floats:

        res_shape = np.broadcast_shapes(np.shape(t), np.shape(w))
        res = np.full(res_shape, np.nan)

        func_idx_for_t = np.searchsorted(borders, t)
        for i in range(len(marginals)):
            mask = func_idx_for_t == i
            t_safe = np.clip(t, a_max=borders[i])
            res[mask] = funcs[i](t_safe, w)[mask]

        return res

    return markov_func
