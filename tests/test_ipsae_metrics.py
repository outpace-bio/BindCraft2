import jax.numpy as jnp

from bindcraft.af2 import binder_residue_mask


def test_binder_mask_marks_only_binder_chains():
    mask = binder_residue_mask(('binder', 'target'), (3, 4))
    assert mask.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_binder_mask_handles_multi_chain_binders_and_multi_chain_targets():
    mask = binder_residue_mask(('binder_a', 'binder_b', 'target', 'target_2'), (2, 2, 3, 1))
    assert mask.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_binder_mask_is_all_binder_for_a_binder_alone_state():
    assert binder_residue_mask(('binder',), (5,)).tolist() == [1.0] * 5


def test_binder_mask_follows_chain_name_order_not_sorted_order():
    """concatenate_chain_arrays walks chain_names in the order given, so the mask must too."""
    mask = binder_residue_mask(('target', 'binder'), (2, 3))
    assert mask.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


import json
import os

import jax
import pytest

from bindcraft.af2 import alphafold_prediction_metrics
from bindcraft.ipsae import ipsae

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')


def test_ipsae_is_invariant_to_padding_given_gated_masks():
    """ipsae itself ignores padding once the masks are already gated. The wiring that does
    the gating is covered by test_metrics_gating_excludes_padding below."""
    with open(os.path.join(FIXTURE_DIR, 'ipsae_pae.json')) as handle:
        pae = jnp.asarray(json.load(handle)['pae'])
    residues = pae.shape[0]
    binder = (jnp.arange(residues) < 12).astype(jnp.float32)
    unpadded = float(ipsae(pae, binder, 1.0 - binder))

    padding = 8
    padded_pae = jnp.pad(pae, [[0, padding], [0, padding]])
    padded_binder = jnp.pad(binder, [0, padding])
    seq_mask = jnp.pad(jnp.ones(residues, dtype=jnp.float32), [0, padding])
    padded = float(ipsae(padded_pae, padded_binder * seq_mask, (1.0 - padded_binder) * seq_mask))
    assert padded == unpadded


PAE_BINS = 64
PAE_MAX = 32.0


def _pae_head(pae_bins):
    """One-hot PAE logits, so the decoded matrix is exactly the chosen bin centres."""
    breaks = jnp.linspace(0.0, PAE_MAX, PAE_BINS)[:-1]
    return {'predicted_aligned_error': {'logits': jax.nn.one_hot(pae_bins, PAE_BINS) * 30.0,
                                        'breaks': breaks}}


def _pae_bin(value):
    breaks = jnp.linspace(0.0, PAE_MAX, PAE_BINS)[:-1]
    width = breaks[1] - breaks[0]
    centres = jnp.append(breaks + width / 2, breaks[-1] + 1.5 * width)
    return int(jnp.argmin(jnp.abs(centres - value)))


def test_metrics_gating_excludes_padding():
    """Padding decodes to PAE 0, which is under any cutoff. Without the `* seq_mask` gating
    in alphafold_prediction_metrics every padded column joins the target and the score
    inflates -- measured at 0.939 against a true 0.064. This asserts the gating is present."""
    binder_residues, target_residues, padding = 6, 10, 5
    residues = binder_residues + target_residues
    interface, elsewhere = _pae_bin(4.0), _pae_bin(26.0)
    pae_bins = (jnp.full((residues, residues), elsewhere)
                .at[:binder_residues, binder_residues:].set(interface)
                .at[binder_residues:, :binder_residues].set(interface))
    binder = (jnp.arange(residues) < binder_residues).astype(jnp.float32)
    seq_mask = jnp.ones(residues, dtype=jnp.float32)
    asym_id = (jnp.arange(residues) >= binder_residues).astype(jnp.int32)

    unpadded = alphafold_prediction_metrics(_pae_head(pae_bins), seq_mask, asym_id, binder)

    padded_bins = jnp.zeros((residues + padding, residues + padding), dtype=jnp.int32).at[:residues, :residues].set(pae_bins)
    padded = alphafold_prediction_metrics(
        _pae_head(padded_bins),
        jnp.pad(seq_mask, [0, padding]),
        jnp.pad(asym_id, [0, padding], constant_values=2),
        jnp.pad(binder, [0, padding]))

    assert float(unpadded['ipsae']) > 0.0
    assert float(padded['ipsae']) == pytest.approx(float(unpadded['ipsae']), abs=1e-7)
    assert float(padded['ipsae_per_residue'][residues:].max()) == 0.0
