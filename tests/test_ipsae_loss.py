import jax.numpy as jnp

from bindcraft.ipsae import soft_ipsae
from bindcraft.loss import REGISTERED_LOSSES, LOSS_TARGET_WEIGHTING, annealed_pae_cutoff, build_losses


def test_ipsae_loss_is_registered_and_targets_the_bound_state():
    assert 'ipsae_loss' in REGISTERED_LOSSES
    assert LOSS_TARGET_WEIGHTING['ipsae_loss'] == 'binds_target'


def test_registering_the_loss_makes_its_weight_a_known_setting():
    """known_campaign_settings builds weights_* from REGISTERED_LOSSES, so this is only
    true once the decorator above has run. Task 3 deliberately does not assert it."""
    from bindcraft.settings import known_campaign_settings
    assert 'weights_ipsae_loss' in known_campaign_settings()


def test_ipsae_loss_is_off_by_default():
    assert 'ipsae_loss' not in build_losses({})


def test_ipsae_loss_switches_on_with_a_weight():
    assert 'ipsae_loss' in build_losses({'weights_ipsae_loss': 0.05})


def test_cutoff_anneals_from_warmup_down_to_the_paper_value():
    soft_sequence = jnp.asarray(1.0 / 20.0)
    hard_sequence = jnp.asarray(1.0)
    assert float(annealed_pae_cutoff(soft_sequence, 10.0, 30.0)) > 28.0
    assert float(annealed_pae_cutoff(hard_sequence, 10.0, 30.0)) == 10.0


def test_a_wide_cutoff_scores_above_the_paper_cutoff_on_a_dispersed_interface():
    """The reason the anneal exists: at cutoff 10 a step-0 trajectory scores exactly zero."""
    residues = 40
    binder = (jnp.arange(residues) < 10).astype(jnp.float32)
    target = 1.0 - binder
    pae = jnp.full((residues, residues), 22.0)
    assert float(soft_ipsae(pae, binder, target, pae_cutoff=10.0)) == 0.0
    assert float(soft_ipsae(pae, binder, target, pae_cutoff=30.0)) > 0.0


import jax


def test_the_loss_has_a_live_gradient_on_a_dispersed_interface():
    residues = 40
    binder = (jnp.arange(residues) < 10).astype(jnp.float32)
    target = 1.0 - binder
    pae = jnp.full((residues, residues), 22.0)
    gradient = jax.grad(lambda matrix: 1 - soft_ipsae(matrix, binder, target, pae_cutoff=30.0))(pae)
    assert bool(jnp.isfinite(gradient).all())
    assert float(jnp.abs(gradient).sum()) > 0.0
