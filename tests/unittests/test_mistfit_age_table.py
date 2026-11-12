"""Tabulated age log-likelihood (age_grid, age_loglike) in both inference models.

Synthetic grids only; no MIST data is needed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.infer.util import log_density

from jaxstar.mistfit import MistFit, MistModel, MistObservations
from test_mistmodel import grid_path, OPTIONS as ISO_OPTIONS, PARAMS as ISO_PARAMS
from test_mist_tracks import (track_path, track_precision, LATENTS as TRACK_LATENTS,
                              OPTIONS as TRACK_OPTIONS)

AGE_OBS, AGE_ERR = 2.0, 0.3
GRID = np.linspace(0., 14., 14001)  # 1 Myr spacing
GAUSSIAN = -0.5 * ((AGE_OBS - GRID) / AGE_ERR) ** 2


@pytest.fixture(params=['isochrone', 'track'])
def case(request, grid_path, track_path):
    """(density(point=None, **options), age at the test point) for each model kind."""
    if request.param == 'isochrone':
        observations = MistObservations(['teff', 'parallax'], [5400., 8.4], [70., .1])
        fit = MistFit(grid_path, observations=observations)
        parameters = {'age': 10 ** ISO_PARAMS['logage'] / 1e9,
                      **{k: v for k, v in ISO_PARAMS.items() if k != 'logage'}}
        base = dict(ISO_OPTIONS, linear_age=True)
    else:
        observations = MistObservations(['teff', 'parallax'], [6100., 6.7], [120., .1])
        fit = MistFit(model=MistModel.from_tracks(track_path), observations=observations)
        parameters = dict(TRACK_LATENTS)
        base = dict(TRACK_OPTIONS, linear_age=True)

    def density(point=parameters, **options):
        return log_density(fit.model, (), dict(base, **options), point)

    density.parameters = parameters
    age = float(density()[1]['age']['value'])
    return density, age


def test_table_equal_to_gaussian_matches_age_obs(case):
    density, _ = case
    builtin = density(age_obs=AGE_OBS, age_err=AGE_ERR)[0]
    table = density(age_grid=GRID, age_loglike=GAUSSIAN)[0]
    # Linear interpolation of a quadratic with 1 Myr spacing is accurate to ~1e-6.
    np.testing.assert_allclose(table, builtin, rtol=0., atol=1e-5)
    assert float(table) != float(density()[0])


def test_table_adds_its_interpolated_value(case):
    density, age = case
    grid, loglike = np.array([0., 1., 3., 14.]), np.array([-4., -1., -2.5, -9.])
    expected = np.interp(age, grid, loglike)
    difference = density(age_grid=grid, age_loglike=loglike)[0] - density()[0]
    np.testing.assert_allclose(difference, expected, rtol=0., atol=1e-12)


def test_table_takes_end_values_outside_the_grid(case):
    density, age = case
    low = np.array([age + 1., age + 2.])
    high = np.array([age - 2., age - 1.])
    baseline = density()[0]
    np.testing.assert_allclose(density(age_grid=low, age_loglike=[-1., -3.])[0] - baseline, -1., atol=1e-12)
    np.testing.assert_allclose(density(age_grid=high, age_loglike=[-1., -3.])[0] - baseline, -3., atol=1e-12)


def test_table_gradient_is_finite_and_follows_the_slope(case):
    density, _ = case
    grid, loglike = np.array([0., 14.]), np.array([0., -14.])  # slope of -1 per Gyr

    def difference(vector):
        point = dict(zip(density.parameters, vector))
        return density(point, age_grid=grid, age_loglike=loglike)[0] - density(point)[0]

    vector = jnp.array(list(density.parameters.values()))
    gradient = np.asarray(jax.grad(difference)(vector))
    assert np.isfinite(gradient).all()
    if 'age' in density.parameters:  # isochrone: age is a latent, so the slope is exact
        np.testing.assert_allclose(gradient[list(density.parameters).index('age')], -1., atol=1e-12)
    else:  # track: age depends on mass, FeH and EEP through the track
        assert np.any(gradient != 0.)


@pytest.mark.parametrize('options,message', [
    (dict(age_grid=GRID), 'together'),
    (dict(age_loglike=GAUSSIAN), 'together'),
    (dict(age_grid=GRID, age_loglike=GAUSSIAN, age_obs=2.), 'not both'),
    (dict(age_grid=GRID, age_loglike=GAUSSIAN, age_err=.3), 'not both'),
    (dict(age_grid=GRID, age_loglike=GAUSSIAN[:-1]), 'same shape'),
    (dict(age_grid=[1.], age_loglike=[0.]), 'at least 2 points'),
    (dict(age_grid=np.ones((2, 2)), age_loglike=np.ones((2, 2))), '1D array'),
    (dict(age_grid=[2., 1.], age_loglike=[0., 0.]), 'strictly increasing'),
    (dict(age_grid=[1., 1.], age_loglike=[0., 0.]), 'strictly increasing'),
    (dict(age_grid=[1., np.nan], age_loglike=[0., 0.]), 'finite'),
])
def test_argument_checks(case, options, message):
    density, _ = case
    with pytest.raises(ValueError, match=message):
        density(**options)
