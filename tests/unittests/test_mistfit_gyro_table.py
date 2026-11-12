"""Tabulated 2D rotation likelihood (gyro_teff_grid, gyro_age_grid, gyro_loglike).

Synthetic grids only; no MIST data is needed.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.infer.util import log_density
from scipy.interpolate import RegularGridInterpolator

from jaxstar.mistfit import MistFit, MistModel
from jaxstar.mistfit.inference import _interp_table_2d
from test_mist_tracks import native_fields, LATENTS, OPTIONS
from test_mistfit_age_table import case, grid_path, track_path, track_precision  # noqa: F401

TEFF = np.array([4000., 4500., 5200., 6000.])
AGE = np.array([0., 0.5, 1.5, 4., 14.])
RNG = np.random.default_rng(3)
TABLE = -RNG.uniform(0., 6., size=(TEFF.size, AGE.size))


def reference(teff, age, teff_grid=TEFF, age_grid=AGE, table=TABLE):
    """Bilinear interpolation, clamped to the end values, through scipy."""
    interpolator = RegularGridInterpolator((teff_grid, age_grid), table)
    return interpolator(np.stack([np.clip(teff, teff_grid[0], teff_grid[-1]),
                                  np.clip(age, age_grid[0], age_grid[-1])], axis=-1))


def test_bilinear_is_exact_at_nodes():
    teff, age = np.meshgrid(TEFF, AGE, indexing='ij')
    actual = _interp_table_2d(teff, age, TEFF, AGE, TABLE)
    np.testing.assert_allclose(actual, TABLE, rtol=0., atol=1e-13)


def test_bilinear_midpoints_by_hand():
    # Midpoint of a cell is the mean of its four corners.
    teff = (TEFF[:-1] + TEFF[1:]) / 2.
    age = (AGE[:-1] + AGE[1:]) / 2.
    mid_t, mid_a = np.meshgrid(teff, age, indexing='ij')
    corners = (TABLE[:-1, :-1] + TABLE[1:, :-1] + TABLE[:-1, 1:] + TABLE[1:, 1:]) / 4.
    np.testing.assert_allclose(_interp_table_2d(mid_t, mid_a, TEFF, AGE, TABLE), corners, atol=1e-13)


def test_bilinear_matches_scipy_inside_and_clamps_outside():
    teff = RNG.uniform(3000., 7000., size=200)
    age = RNG.uniform(-2., 20., size=200)
    np.testing.assert_allclose(_interp_table_2d(teff, age, TEFF, AGE, TABLE),
                               reference(teff, age), rtol=0., atol=1e-12)
    # Outside in one axis only, and the four corners: end values.
    np.testing.assert_allclose(_interp_table_2d(3000., AGE[2], TEFF, AGE, TABLE), TABLE[0, 2], atol=1e-13)
    np.testing.assert_allclose(_interp_table_2d(TEFF[1], 99., TEFF, AGE, TABLE), TABLE[1, -1], atol=1e-13)
    np.testing.assert_allclose(_interp_table_2d(9000., -1., TEFF, AGE, TABLE), TABLE[-1, 0], atol=1e-13)


def test_bilinear_works_under_jit_and_vmap():
    teff = jnp.array([4200., 5000., 8000.])
    age = jnp.array([0.2, 3., 1.])
    actual = jax.jit(jax.vmap(lambda t, a: _interp_table_2d(t, a, TEFF, AGE, TABLE)))(teff, age)
    np.testing.assert_allclose(actual, reference(np.asarray(teff), np.asarray(age)), atol=1e-12)


def test_bilinear_gradients_are_finite_inside_and_outside_the_grid():
    def value(point):
        return _interp_table_2d(point[0], point[1], TEFF, AGE, TABLE)

    inside = np.asarray(jax.grad(value)(jnp.array([4800., 1.0])))
    # Slope of the cell by hand: derivative of the bilinear form at the point.
    i, j = 1, 1
    tx, ty = (4800. - TEFF[i]) / (TEFF[i + 1] - TEFF[i]), (1.0 - AGE[j]) / (AGE[j + 1] - AGE[j])
    d_teff = ((1 - ty) * (TABLE[i + 1, j] - TABLE[i, j]) + ty * (TABLE[i + 1, j + 1] - TABLE[i, j + 1])
              ) / (TEFF[i + 1] - TEFF[i])
    d_age = ((1 - tx) * (TABLE[i, j + 1] - TABLE[i, j]) + tx * (TABLE[i + 1, j + 1] - TABLE[i + 1, j])
             ) / (AGE[j + 1] - AGE[j])
    np.testing.assert_allclose(inside, [d_teff, d_age], rtol=1e-12)
    for point in ([3000., 1.0], [4800., 30.], [9000., -5.]):
        outside = np.asarray(jax.grad(value)(jnp.array(point)))
        assert np.isfinite(outside).all()
    # Clamped to a constant in a direction that leaves the grid.
    assert np.asarray(jax.grad(value)(jnp.array([3000., 1.0])))[0] == 0.


def test_minus_inf_nodes_give_nonfinite_gradients_so_callers_must_floor():
    table = TABLE.copy()
    table[0, 0] = -np.inf
    point = jnp.array([4250., 0.25])  # in the cell that has the -inf corner
    assert not np.isfinite(jax.grad(lambda p: _interp_table_2d(p[0], p[1], TEFF, AGE, table))(point)).all()
    floored = np.maximum(table, -700.)
    assert np.isfinite(jax.grad(lambda p: _interp_table_2d(p[0], p[1], TEFF, AGE, floored))(point)).all()


def test_table_adds_its_bilinear_value_at_the_model_teff_and_age(case):
    density, age = case
    teff = float(density()[1]['teff']['value'])
    # Grid chosen so the model point is inside a cell and the table has structure in both axes.
    teff_grid = np.array([teff - 500., teff - 100., teff + 300.])
    age_grid = np.array([age - 1., age + 0.5, age + 2.])
    table = np.array([[-1., -2., -4.], [-3., -2.5, -6.], [-0.5, -5., -2.]])
    expected = reference(teff, age, teff_grid, age_grid, table)
    difference = (density(gyro_teff_grid=teff_grid, gyro_age_grid=age_grid, gyro_loglike=table)[0]
                  - density()[0])
    np.testing.assert_allclose(difference, expected, rtol=0., atol=1e-10)


def test_table_constant_in_teff_equals_the_age_table(case):
    density, _ = case
    age_grid = np.array([0., 1., 3., 14.])
    age_loglike = np.array([-4., -1., -2.5, -9.])
    teff_grid = np.array([3000., 5000., 9000.])
    table = np.tile(age_loglike, (teff_grid.size, 1))
    one_d = density(age_grid=age_grid, age_loglike=age_loglike)[0]
    two_d = density(gyro_teff_grid=teff_grid, gyro_age_grid=age_grid, gyro_loglike=table)[0]
    np.testing.assert_allclose(two_d, one_d, rtol=0., atol=1e-12)


def test_table_outside_the_grid_takes_end_values_with_finite_gradients(case):
    density, age = case
    teff = float(density()[1]['teff']['value'])
    teff_grid = np.array([teff + 1000., teff + 2000.])  # model Teff below the grid
    age_grid = np.array([age + 1., age + 2.])  # model age below the grid
    table = np.array([[-1., -2.], [-3., -4.]])
    kwargs = dict(gyro_teff_grid=teff_grid, gyro_age_grid=age_grid, gyro_loglike=table)
    np.testing.assert_allclose(density(**kwargs)[0] - density()[0], -1., atol=1e-10)

    def difference(vector):
        point = dict(zip(density.parameters, vector))
        return density(point, **kwargs)[0] - density(point)[0]

    gradient = jax.grad(difference)(jnp.array(list(density.parameters.values())))
    assert np.isfinite(np.asarray(gradient)).all()
    np.testing.assert_allclose(np.asarray(gradient), 0., atol=1e-12)


def test_prot_and_table_are_exclusive(case):
    density, _ = case
    with pytest.raises(ValueError, match='prot'):
        density(prot=10., gyro_teff_grid=TEFF, gyro_age_grid=AGE, gyro_loglike=TABLE)


@pytest.mark.parametrize('options,message', [
    (dict(gyro_teff_grid=TEFF), 'together'),
    (dict(gyro_age_grid=AGE), 'together'),
    (dict(gyro_loglike=TABLE), 'together'),
    (dict(gyro_teff_grid=TEFF, gyro_age_grid=AGE), 'together'),
    (dict(gyro_teff_grid=TEFF, gyro_age_grid=AGE, gyro_loglike=TABLE.T), 'shape'),
    (dict(gyro_teff_grid=TEFF, gyro_age_grid=AGE, gyro_loglike=TABLE[:, 0]), 'shape'),
    (dict(gyro_teff_grid=TEFF[:1], gyro_age_grid=AGE, gyro_loglike=TABLE[:1]), 'at least 2 points'),
    (dict(gyro_teff_grid=TEFF, gyro_age_grid=AGE[:1], gyro_loglike=TABLE[:, :1]), 'at least 2 points'),
    (dict(gyro_teff_grid=TEFF[::-1], gyro_age_grid=AGE, gyro_loglike=TABLE), 'strictly increasing'),
    (dict(gyro_teff_grid=TEFF, gyro_age_grid=AGE[::-1], gyro_loglike=TABLE), 'strictly increasing'),
    (dict(gyro_teff_grid=TEFF, gyro_age_grid=np.where(AGE > 1, np.nan, AGE), gyro_loglike=TABLE), 'finite'),
])
def test_argument_checks(case, options, message):
    density, _ = case
    with pytest.raises(ValueError, match=message):
        density(**options)


def test_track_without_a_teff_value_has_no_density_under_the_table(track_path):
    data = native_fields(track_path)
    present = np.ones(data['age_gyr'].shape, dtype=bool)
    for name, values in data.items():
        if values.shape[:3] == present.shape and name == 'logTeff':
            values[:, :, 1:3] = np.nan  # hole in Teff only, around the test point
    np.savez(track_path, **data, track_present=present,
             metadata_json=np.array(json.dumps(dict(kind='track', format_version=2))))
    fit = MistFit(model=MistModel.from_tracks(track_path), observations={'parallax': (6.7, .1)})
    options = dict(OPTIONS, nodata=True)
    table = dict(gyro_teff_grid=TEFF, gyro_age_grid=AGE, gyro_loglike=TABLE)
    assert np.isfinite(log_density(fit.model, (), options, LATENTS)[0])
    value = log_density(fit.model, (), dict(options, **table), LATENTS)[0]
    assert np.isneginf(value)
    gradient = jax.grad(lambda p: log_density(fit.model, (), dict(options, **table), dict(zip(LATENTS, p)))[0])(
        jnp.array(list(LATENTS.values())))
    assert np.isfinite(np.asarray(gradient)).all()
