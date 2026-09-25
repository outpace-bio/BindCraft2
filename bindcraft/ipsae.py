"""ipSAE, the interface score of Dunbrack 2025 (biorxiv 2025.02.10.637595).

Reproduces the `d0res` variant that upstream ipsae.py reports as its headline `ipSAE`
column, reduced as the max over both chain directions. The score reads only a PAE matrix
and two chain masks; no coordinates are involved, because upstream's `dist_cutoff` feeds
pDockQ and the diagnostic residue counts rather than ipSAE itself.

Pass the RAW predicted aligned error. Upstream runs on AlphaFold's asymmetric matrix, so a
symmetrized one gives a different number.
"""
import jax
import jax.numpy as jnp
from jax import Array

DEFAULT_PAE_CUTOFF = 10.0
#mirrors ipsae_loss's `warmup_cutoff` default in bindcraft/loss.py -- exists so the reported
#metrics can show what the loss sees at the permissive end of its anneal. If the two ever
#diverge, this observation stops being meaningful.
WARMUP_PAE_CUTOFF = 30.0
#upstream calc_d0_array clamps the length rather than branching on it, which is what the
#d0res variant uses; the scalar calc_d0 branches at L > 27 and is for d0chn and d0dom only
D0_MINIMUM_LENGTH = 26.0
D0_MINIMUM = 1.0


def d0(length: Array) -> Array:
    return jnp.maximum(D0_MINIMUM, 1.24 * jnp.cbrt(jnp.maximum(D0_MINIMUM_LENGTH, length) - 15.0) - 1.8)


def ipsae_by_residue(pae: Array, from_mask: Array, to_mask: Array, pae_cutoff: float=DEFAULT_PAE_CUTOFF) -> Array:
    """One ipSAE value per residue of the aligned chain, zero outside from_mask."""
    pair_mask = (from_mask[:, None] * to_mask[None, :]) * (pae < pae_cutoff)
    scored_residues = jax.lax.stop_gradient(pair_mask.sum(-1))
    aligned_error = 1.0 / (1.0 + jnp.square(pae / d0(scored_residues)[:, None]))
    return jnp.where(scored_residues > 0, (aligned_error * pair_mask).sum(-1) / jnp.maximum(scored_residues, 1.0), 0.0)


def _both_directions(pae: Array, binder_mask: Array, target_mask: Array, pae_cutoff: float) -> Array:
    return jnp.concatenate([ipsae_by_residue(pae, binder_mask, target_mask, pae_cutoff),
                            ipsae_by_residue(pae, target_mask, binder_mask, pae_cutoff)])


def ipsae(pae: Array, binder_mask: Array, target_mask: Array, pae_cutoff: float=DEFAULT_PAE_CUTOFF) -> Array:
    """The reported score: upstream's `max` row for the binder/target chain pair."""
    return _both_directions(pae, binder_mask, target_mask, pae_cutoff).max()


def soft_ipsae(pae: Array, binder_mask: Array, target_mask: Array, pae_cutoff: float=DEFAULT_PAE_CUTOFF, temperature: float=0.1) -> Array:
    """The gradient variant, with the max softened so more than one residue carries signal."""
    from bindcraft.loss import soft_maximum
    return soft_maximum(_both_directions(pae, binder_mask, target_mask, pae_cutoff), temperature)
