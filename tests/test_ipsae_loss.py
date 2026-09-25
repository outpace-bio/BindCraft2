import jax.numpy as jnp

from bindcraft.ipsae import soft_ipsae
from bindcraft.loss import REGISTERED_LOSSES, LOSS_TARGET_WEIGHTING, annealed_pae_cutoff, build_losses, ipsae_loss


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


# ---------------------------------------------------------------------------
# End-to-end coverage that calls the REGISTERED ipsae_loss, not soft_ipsae
# directly, so the mask construction (is_binder_chain -> binder/target masks)
# and the anneal wiring (sequence_hardness -> annealed_pae_cutoff -> soft_ipsae)
# both have regression coverage. The two tests above only exercise soft_ipsae
# and would still pass if either piece of wiring were broken.
# ---------------------------------------------------------------------------

from bindcraft.protein import AMINO_ACIDS, ATOM_NAMES, Protein, StructurePrediction

BINDER_RESIDUES = 6
TARGET_RESIDUES = 8


def _toy_protein(sequence: jnp.ndarray) -> Protein:
    """A Protein with nothing resolved -- ipsae_loss never reads atoms/atom_mask/residue_index,
    only sequence (for sequence_hardness) and flags (for real_residue_weights)."""
    residues = sequence.shape[0]
    return Protein(sequence=sequence,
                   atoms=jnp.zeros((residues, len(ATOM_NAMES), 3)),
                   atom_mask=jnp.zeros((residues, len(ATOM_NAMES)), dtype=bool),
                   flags=jnp.zeros((residues,), dtype=jnp.uint8),
                   residue_index=jnp.arange(residues, dtype=jnp.int32))


def _toy_complex(binder_sequence: jnp.ndarray, pae: jnp.ndarray):
    """A minimal ('complex' -> {'binder', 'target'}) ProteinStates/StructurePredictions pair,
    sized just enough for chain_residue_slices, amino_acid_probabilities, real_residue_weights
    and metrics['pae'] to all have something real to operate on."""
    protein_complex = {'binder': _toy_protein(binder_sequence),
                        'target': _toy_protein(jnp.zeros((TARGET_RESIDUES, len(AMINO_ACIDS))))}
    protein_states = {'complex': protein_complex}
    predictions = {'complex': StructurePrediction(protein_complex=protein_complex, metrics={'pae': pae})}
    return protein_states, predictions


def _uniform_binder_sequence():
    """All-zero logits: softmax is exactly uniform, so sequence_hardness is exactly 1/20 --
    a maximally soft, step-0-like sequence."""
    return jnp.zeros((BINDER_RESIDUES, len(AMINO_ACIDS)))


def _one_hot_binder_sequence():
    """A discrete, fully-hardened sequence: already a valid probability distribution, so
    amino_acid_probabilities passes it through unchanged and sequence_hardness is exactly 1.0."""
    return jax.nn.one_hot(jnp.zeros(BINDER_RESIDUES, dtype=jnp.int32), len(AMINO_ACIDS))


def _dispersed_pae():
    """PAE 22 everywhere: inside the annealed warmup cutoff (~30) but outside the paper's
    fixed cutoff (10) -- the exact step-0 regime the anneal exists to fix."""
    residues = BINDER_RESIDUES + TARGET_RESIDUES
    return jnp.full((residues, residues), 22.0)


def test_ipsae_loss_scores_below_one_on_a_confident_cross_chain_interface():
    """Catches Mutation A (binder_mask filled for every chain, so target_mask is all-zero and
    the loss pins at exactly 1.0 no matter how confident the interface is). Uses a one-hot
    binder sequence so the anneal settles at the paper's cutoff of 10, independent of Mutation B."""
    residues = BINDER_RESIDUES + TARGET_RESIDUES
    pae = jnp.full((residues, residues), 25.0)
    pae = pae.at[:BINDER_RESIDUES, BINDER_RESIDUES:].set(2.0).at[BINDER_RESIDUES:, :BINDER_RESIDUES].set(2.0)
    protein_states, predictions = _toy_complex(_one_hot_binder_sequence(), pae)
    assert float(ipsae_loss(protein_states, predictions)) < 1.0


def test_ipsae_loss_has_a_live_gradient_on_a_soft_sequence_via_the_anneal():
    """Catches Mutation B (cutoff = pae_cutoff, i.e. no anneal): at a fixed cutoff of 10 this
    dispersed PAE (22) fails the mask everywhere, the loss pins at exactly 1.0, and its gradient
    with respect to the PAE matrix is exactly zero. Only the live anneal (cutoff -> ~30 on this
    maximally soft sequence) lets any signal through."""
    def loss_from_pae(pae):
        protein_states, predictions = _toy_complex(_uniform_binder_sequence(), pae)
        return ipsae_loss(protein_states, predictions)

    pae = _dispersed_pae()
    assert float(loss_from_pae(pae)) < 1.0
    gradient = jax.grad(loss_from_pae)(pae)
    assert bool(jnp.isfinite(gradient).all())
    assert float(jnp.abs(gradient).sum()) > 0.0


def test_a_harder_binder_sequence_tightens_the_cutoff_and_raises_the_loss():
    """Also catches Mutation B: without the anneal, hardening the sequence has no effect on the
    cutoff, so the hard and soft sequences would score identically (both pinned at 1.0 on this
    dispersed PAE). With the anneal wired up, hardening tightens the cutoff toward 10, the mask
    stops passing, and the loss strictly increases -- pinning the direction of the anneal."""
    pae = _dispersed_pae()
    soft_states, soft_predictions = _toy_complex(_uniform_binder_sequence(), pae)
    hard_states, hard_predictions = _toy_complex(_one_hot_binder_sequence(), pae)
    soft_loss = float(ipsae_loss(soft_states, soft_predictions))
    hard_loss = float(ipsae_loss(hard_states, hard_predictions))
    assert hard_loss > soft_loss
