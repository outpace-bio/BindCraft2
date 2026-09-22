import jax.numpy as jnp

from bindcraft.af2 import chain_residue_masks


def test_chain_masks_split_binder_from_target():
    binder = chain_residue_masks(('binder', 'target'), (3, 4), binder=True)
    target = chain_residue_masks(('binder', 'target'), (3, 4), binder=False)
    assert binder.tolist() == [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
    assert target.tolist() == [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]]


def test_chain_masks_keep_every_chain_on_its_own_row():
    """The whole point of the split: two binder copies and two target chains stay apart,
    because ipSAE is defined on an ordered chain pair and pooling inflates the reverse
    direction."""
    binder = chain_residue_masks(('binder_a', 'binder_b', 'target', 'target_2'), (2, 2, 3, 1), binder=True)
    target = chain_residue_masks(('binder_a', 'binder_b', 'target', 'target_2'), (2, 2, 3, 1), binder=False)
    assert binder.tolist() == [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
    assert target.tolist() == [[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]


def test_chain_masks_give_a_binder_alone_state_no_target_rows():
    binder = chain_residue_masks(('binder',), (5,), binder=True)
    target = chain_residue_masks(('binder',), (5,), binder=False)
    assert binder.tolist() == [[1.0] * 5]
    assert target.shape == (0, 5)


def test_chain_masks_follow_chain_name_order_not_sorted_order():
    """concatenate_chain_arrays walks chain_names in the order given, so the masks must too."""
    binder = chain_residue_masks(('target', 'binder'), (2, 3), binder=True)
    target = chain_residue_masks(('target', 'binder'), (2, 3), binder=False)
    assert binder.tolist() == [[0.0, 0.0, 1.0, 1.0, 1.0]]
    assert target.tolist() == [[1.0, 1.0, 0.0, 0.0, 0.0]]


import json
import os

import jax
import pytest

from bindcraft.af2 import alphafold_prediction_metrics, interface_asym_ids
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


def _pae_centres():
    breaks = jnp.linspace(0.0, PAE_MAX, PAE_BINS)[:-1]
    width = breaks[1] - breaks[0]
    return jnp.append(breaks + width / 2, breaks[-1] + 1.5 * width)


def _pae_bin(value):
    return int(jnp.argmin(jnp.abs(_pae_centres() - value)))


def _pae_bins_of(pae):
    """Recover bin indices from a matrix whose values are already decoded bin centres."""
    return jnp.argmin(jnp.abs(pae[:, :, None] - _pae_centres()[None, None, :]), axis=-1)


def _metrics(pae_bins, seq_mask, chain_names, chain_lengths, padding=0):
    """Calls alphafold_prediction_metrics exactly as af2.py does, padding included."""
    pad_columns = lambda masks: jnp.pad(masks, [[0, 0], [0, padding]])
    return alphafold_prediction_metrics(
        _pae_head(pae_bins),
        seq_mask=seq_mask,
        interface_asym_id=jnp.pad(interface_asym_ids(chain_names, chain_lengths), [0, padding], constant_values=len(chain_names)),
        binder_chain_masks=pad_columns(chain_residue_masks(chain_names, chain_lengths, binder=True)),
        target_chain_masks=pad_columns(chain_residue_masks(chain_names, chain_lengths, binder=False)))


def test_metrics_gating_excludes_padding():
    """Padding decodes to PAE 0, which is under any cutoff. Without the `* seq_mask` gating
    in alphafold_prediction_metrics every padded column joins the target and the score
    inflates -- measured at 0.939 against a true 0.064. This asserts the gating is present."""
    binder_residues, target_residues, padding = 6, 10, 5
    residues = binder_residues + target_residues
    chain_names, chain_lengths = ('binder', 'target'), (binder_residues, target_residues)
    #the two off-diagonal blocks differ, so this fixture also distinguishes the raw PAE from
    #the symmetrized one; see test_metrics_read_the_raw_asymmetric_pae below
    forward, reverse, elsewhere = _pae_bin(4.0), _pae_bin(9.0), _pae_bin(26.0)
    pae_bins = (jnp.full((residues, residues), elsewhere)
                .at[:binder_residues, binder_residues:].set(forward)
                .at[binder_residues:, :binder_residues].set(reverse))
    seq_mask = jnp.ones(residues, dtype=jnp.float32)

    unpadded = _metrics(pae_bins, seq_mask, chain_names, chain_lengths)

    padded_bins = jnp.zeros((residues + padding, residues + padding), dtype=jnp.int32).at[:residues, :residues].set(pae_bins)
    padded = _metrics(padded_bins, jnp.pad(seq_mask, [0, padding]), chain_names, chain_lengths, padding=padding)

    assert float(unpadded['ipsae']) > 0.0
    assert float(padded['ipsae']) == pytest.approx(float(unpadded['ipsae']), abs=1e-7)
    assert float(padded['ipsae_per_residue'][residues:].max()) == 0.0


def test_metrics_read_the_raw_asymmetric_pae():
    """af2.py must feed ipSAE the decoded `pae`, never `metrics['pae']`, which is symmetrized.

    This is the branch's headline correctness constraint: the spec records 0.047426 raw
    against 0.033419 symmetrized on the golden fixture. The two off-diagonal blocks here are
    4.0 one way and 9.0 the other, so the swap is visible; a symmetric fixture cannot see it."""
    binder_residues, target_residues = 6, 10
    residues = binder_residues + target_residues
    chain_names, chain_lengths = ('binder', 'target'), (binder_residues, target_residues)
    pae_bins = (jnp.full((residues, residues), _pae_bin(26.0))
                .at[:binder_residues, binder_residues:].set(_pae_bin(4.0))
                .at[binder_residues:, :binder_residues].set(_pae_bin(9.0)))
    seq_mask = jnp.ones(residues, dtype=jnp.float32)
    metrics = _metrics(pae_bins, seq_mask, chain_names, chain_lengths)

    binder = (jnp.arange(residues) < binder_residues).astype(jnp.float32)
    raw_pae = jnp.take(_pae_centres(), pae_bins)
    raw_answer = float(ipsae(raw_pae, binder, 1.0 - binder))
    symmetrized_answer = float(ipsae((raw_pae + raw_pae.T) / 2, binder, 1.0 - binder))

    assert abs(raw_answer - symmetrized_answer) / raw_answer > 0.25
    assert float(metrics['ipsae']) == pytest.approx(raw_answer, abs=1e-6)
    assert float(metrics['ipsae']) != pytest.approx(symmetrized_answer, abs=1e-6)


# Upstream ipsae.py v4 on tests/fixtures/ipsae_multichain_{pae.json,model.pdb} at cutoffs 10
# and 15, ipSAE column of the three `max` rows. Regenerate with
# tests/fixtures/make_ipsae_multichain_fixture.py; see its docstring.
UPSTREAM_BINDER_A_TARGET = 0.041179
UPSTREAM_BINDER_B_TARGET = 0.020824
UPSTREAM_BINDER_A_BINDER_B = 0.113578
# The same fixture scored with the binder copies POOLED into one mask, which is what af2.py
# used to do. Pooling lets a target residue count its good pairs against both copies at once,
# so n0res goes 20 -> 40, d0 goes 1.0000 -> 1.8258 and every pTM term lifts. Pinned as the
# wrong answer.
POOLED_BINDER_MASK = 0.095718
MULTICHAIN_CHAIN_NAMES = ('binder_0', 'binder_1', 'target')
MULTICHAIN_CHAIN_LENGTHS = (20, 20, 30)


@pytest.fixture
def multichain_pae():
    with open(os.path.join(FIXTURE_DIR, 'ipsae_multichain_pae.json')) as handle:
        return jnp.asarray(json.load(handle)['pae'])


def test_multichain_ipsae_matches_upstream_per_chain_pair(multichain_pae):
    """Two binder copies against one target, checked against upstream's per-pair `max` rows.

    Upstream emits one row per ordered chain pair, so the reported score is the max over
    (binder chain, target chain) pairs: max(0.041179, 0.020824). Pooling the copies gives
    0.095718 on this fixture, +132% and permissive, so oligomer designs would score better
    than upstream says."""
    metrics = _metrics(_pae_bins_of(multichain_pae), jnp.ones(70, dtype=jnp.float32),
                       MULTICHAIN_CHAIN_NAMES, MULTICHAIN_CHAIN_LENGTHS)
    assert float(metrics['ipsae']) == pytest.approx(max(UPSTREAM_BINDER_A_TARGET, UPSTREAM_BINDER_B_TARGET), abs=1e-6)
    assert float(metrics['ipsae']) != pytest.approx(POOLED_BINDER_MASK, abs=1e-6)


def test_multichain_ipsae_ignores_the_binder_to_binder_pair(multichain_pae):
    """The two copies pack against each other more confidently (0.113578) than either binds
    the target. The reported score is the binder/target interface, so that pair is not in the
    reduction."""
    metrics = _metrics(_pae_bins_of(multichain_pae), jnp.ones(70, dtype=jnp.float32),
                       MULTICHAIN_CHAIN_NAMES, MULTICHAIN_CHAIN_LENGTHS)
    assert UPSTREAM_BINDER_A_BINDER_B > UPSTREAM_BINDER_A_TARGET
    assert float(metrics['ipsae']) != pytest.approx(UPSTREAM_BINDER_A_BINDER_B, abs=1e-6)


def test_multichain_per_residue_stays_pooled_across_every_binder_copy(multichain_pae):
    """sequence_optimization's interface_ipsae mutation weighting wants one value per binder
    residue over ALL binder chains, so this array must NOT follow the per-pair reduction: a
    per-pair array would leave the losing copy at zero and the weighting would never mutate
    it."""
    metrics = _metrics(_pae_bins_of(multichain_pae), jnp.ones(70, dtype=jnp.float32),
                       MULTICHAIN_CHAIN_NAMES, MULTICHAIN_CHAIN_LENGTHS)
    per_residue = metrics['ipsae_per_residue']
    assert per_residue.shape == (70,)
    assert float(per_residue[:20].max()) == pytest.approx(0.020824, abs=1e-6)
    assert float(per_residue[20:40].max()) == pytest.approx(0.012498, abs=1e-6)
    assert float(per_residue[40:].max()) == 0.0


def test_a_binder_alone_state_reports_zero_ipsae():
    """No target chain means no chain pair to reduce over."""
    residues = 8
    metrics = _metrics(jnp.full((residues, residues), _pae_bin(2.0)),
                       jnp.ones(residues, dtype=jnp.float32), ('binder',), (residues,))
    assert float(metrics['ipsae']) == 0.0
    assert float(jnp.abs(metrics['ipsae_per_residue']).max()) == 0.0
