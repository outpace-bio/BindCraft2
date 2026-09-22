import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bindcraft.ipsae import d0, ipsae, ipsae_by_residue, soft_ipsae

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
BINDER_RESIDUES = 12
# From upstream ipsae.py v4 on the committed fixture at cutoffs 10 and 15.
# Regenerate with tests/fixtures/make_ipsae_fixture.py; see its docstring.
UPSTREAM_BINDER_TO_TARGET = 0.047426
UPSTREAM_TARGET_TO_BINDER = 0.020000
UPSTREAM_MAX = 0.047426


@pytest.fixture
def fixture_pae():
    with open(os.path.join(FIXTURE_DIR, 'ipsae_pae.json')) as handle:
        return jnp.asarray(json.load(handle)['pae'])


@pytest.fixture
def chain_masks(fixture_pae):
    residues = fixture_pae.shape[0]
    binder = jnp.arange(residues) < BINDER_RESIDUES
    return binder.astype(jnp.float32), (~binder).astype(jnp.float32)


def test_matches_upstream_on_both_directions(fixture_pae, chain_masks):
    binder, target = chain_masks
    forward = float(ipsae_by_residue(fixture_pae, binder, target).max())
    reverse = float(ipsae_by_residue(fixture_pae, target, binder).max())
    assert forward == pytest.approx(UPSTREAM_BINDER_TO_TARGET, abs=1e-6)
    assert reverse == pytest.approx(UPSTREAM_TARGET_TO_BINDER, abs=1e-6)


def test_matches_upstream_max(fixture_pae, chain_masks):
    binder, target = chain_masks
    assert float(ipsae(fixture_pae, binder, target)) == pytest.approx(UPSTREAM_MAX, abs=1e-6)


def test_symmetrized_pae_would_disagree(fixture_pae, chain_masks):
    """Guards the af2.py:158 vs :159 distinction: ipSAE must read the raw matrix."""
    binder, target = chain_masks
    symmetrized = (fixture_pae + fixture_pae.T) / 2
    assert float(ipsae(symmetrized, binder, target)) != pytest.approx(UPSTREAM_MAX, abs=1e-3)


def test_d0_matches_calc_d0_array_across_the_clamp():
    assert [round(float(d0(jnp.asarray(float(n)))), 4) for n in (25, 26, 27, 28)] == [1.0, 1.0, 1.0389, 1.1157]


def test_empty_interface_is_zero_with_finite_gradient(chain_masks):
    binder, target = chain_masks
    pae = jnp.full((BINDER_RESIDUES + 30, BINDER_RESIDUES + 30), 30.0)
    assert float(ipsae(pae, binder, target)) == 0.0
    gradient = jax.grad(lambda matrix: soft_ipsae(matrix, binder, target))(pae)
    assert bool(jnp.isfinite(gradient).all())


def test_soft_ipsae_has_live_gradient_at_a_wide_cutoff(chain_masks):
    binder, target = chain_masks
    pae = jnp.full((BINDER_RESIDUES + 30, BINDER_RESIDUES + 30), 30.0)
    value = soft_ipsae(pae, binder, target, pae_cutoff=31.0)
    gradient = jax.grad(lambda matrix: soft_ipsae(matrix, binder, target, pae_cutoff=31.0))(pae)
    assert float(value) > 0.0
    assert float(jnp.abs(gradient).sum()) > 0.0


def test_max_takes_the_reverse_direction_when_it_leads():
    """A target that locates the binder confidently while the binder does not reciprocate."""
    residues = 40
    binder = (jnp.arange(residues) < 10).astype(jnp.float32)
    target = (jnp.arange(residues) >= 10).astype(jnp.float32)
    pae = np.full((residues, residues), 30.0)
    pae[10:40, 0:10] = 1.0
    pae = jnp.asarray(pae)
    forward = float(ipsae_by_residue(pae, binder, target).max())
    reverse = float(ipsae_by_residue(pae, target, binder).max())
    assert reverse > forward
    assert float(ipsae(pae, binder, target)) == pytest.approx(reverse, abs=1e-6)


def test_by_residue_is_zero_outside_the_from_mask(fixture_pae, chain_masks):
    binder, target = chain_masks
    values = ipsae_by_residue(fixture_pae, binder, target)
    assert values.shape == (fixture_pae.shape[0],)
    assert float(jnp.abs(values[BINDER_RESIDUES:]).max()) == 0.0
