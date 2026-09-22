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

from bindcraft.ipsae import ipsae

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')


def test_padding_residues_score_as_neither_chain():
    """A padded prediction must give the same ipSAE as the unpadded one."""
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
