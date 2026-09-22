import jax.numpy as jnp
import pytest

from bindcraft.campaign_output import LEADING_CONFIDENCE_COLUMNS
from bindcraft.MPNN_stage import REACHABLE_CONFIDENCE_BOUNDS
from bindcraft.protein import QUALITY_METRIC_TYPES
from bindcraft.rank import LOWER_IS_BETTER_METRICS, MODALITY_METRICS
from bindcraft.score import PREDICTED_METRICS


def test_ipsae_is_a_reported_metric():
    assert 'i_pSAE' in PREDICTED_METRICS
    assert 'i_pSAE' in LEADING_CONFIDENCE_COLUMNS


def test_ipsae_keeps_its_ipTM_neighbour():
    assert 'i_pTM' in PREDICTED_METRICS
    assert 'i_pTM' in LEADING_CONFIDENCE_COLUMNS


def test_detarget_ipsae_is_lower_is_better():
    assert 'i_pSAE_detarget' in LOWER_IS_BETTER_METRICS
    assert 'i_pSAE' not in LOWER_IS_BETTER_METRICS


def test_ipsae_is_documented_in_the_rank_glossary():
    assert 'i_pSAE' in MODALITY_METRICS['every campaign']
    assert 'i_pSAE_detarget' in MODALITY_METRICS['multi-target, cross-reactivity and detargeting']


def test_ipsae_has_a_quality_metric_type_and_an_ensemble_bound():
    assert QUALITY_METRIC_TYPES[('i_pSAE', 'global')] == 'ipTM'
    assert REACHABLE_CONFIDENCE_BOUNDS['ipsae'] == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Mutation weighting (Half B): interface_confidence_weights must actually read
# whichever per-residue metric its `metric` argument names, not a hardcoded
# 'iptm_per_residue'. Everything below constructs a real two-chain complex
# (a binder chain that geometrically faces a target chain) and carries BOTH
# iptm_per_residue and ipsae_per_residue with deliberately asymmetric values
# (not mirror images of each other -- see module docstring on each test) so a
# single mis-keyed read cannot land on the right answer by coincidence.
# ---------------------------------------------------------------------------

from bindcraft.protein import AMINO_ACIDS, ATOM_NAMES, ATOM_INDEX, Protein, StructurePrediction
from bindcraft.sequence_optimization import INTERFACE_WEIGHTING_METRICS, SemigreedySequenceSampler, interface_confidence_weights

BINDER_RESIDUES = 4
TARGET_RESIDUES = 1


def _chain_protein(residue_count: int, cb_coordinates: list[tuple[float, float, float]]) -> Protein:
    """A Protein resolved only at CB, placed at the given coordinates. Sequence/flags/residue_index
    are never read by interface_confidence_weights, only atoms/atom_mask (for the pseudo-beta
    interface mask) and length."""
    atoms = jnp.zeros((residue_count, len(ATOM_NAMES), 3))
    atoms = atoms.at[:, ATOM_INDEX['CB']].set(jnp.asarray(cb_coordinates))
    atom_mask = jnp.zeros((residue_count, len(ATOM_NAMES)), dtype=bool)
    atom_mask = atom_mask.at[:, ATOM_INDEX['CB']].set(True)
    return Protein(sequence=jnp.zeros((residue_count, len(AMINO_ACIDS))),
                   atoms=atoms,
                   atom_mask=atom_mask,
                   flags=jnp.zeros((residue_count,), dtype=jnp.uint8),
                   residue_index=jnp.arange(residue_count, dtype=jnp.int32))


def _binder_target_prediction(iptm_per_residue: list[float], ipsae_per_residue: list[float]) -> StructurePrediction:
    """binder (4 residues, all within 8 Angstrom of the target) + target (1 residue), so
    pseudo_beta_interface_mask marks every binder residue as interface and no target residue."""
    protein_complex = {'binder': _chain_protein(BINDER_RESIDUES, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)]),
                        'target': _chain_protein(TARGET_RESIDUES, [(0.0, 0.0, 0.0)])}
    metrics = {'iptm_per_residue': jnp.asarray(iptm_per_residue + [0.5]),
               'ipsae_per_residue': jnp.asarray(ipsae_per_residue + [0.5]),
               'plddt': jnp.ones((BINDER_RESIDUES + TARGET_RESIDUES,))}
    return StructurePrediction(protein_complex=protein_complex, metrics=metrics)


# iptm is worst (least confident, i.e. highest resulting weight) at residue 0; ipsae is worst
# at residue 3. Not a reversal of one another (reversing the iptm list gives
# [0.90, 0.50, 0.30, 0.10], which is not the ipsae list below), so a read that silently landed
# on the wrong array could not coincidentally reproduce the right argmax by symmetry.
IPTM_PER_RESIDUE = [0.10, 0.30, 0.50, 0.90]
IPSAE_PER_RESIDUE = [0.80, 0.20, 0.95, 0.05]


def test_interface_confidence_weights_reads_the_requested_metric():
    prediction = _binder_target_prediction(IPTM_PER_RESIDUE, IPSAE_PER_RESIDUE)

    weights_by_iptm = interface_confidence_weights(prediction, metric='iptm_per_residue')
    weights_by_ipsae = interface_confidence_weights(prediction, metric='ipsae_per_residue')

    # iptm's least-confident binder residue is index 0; ipsae's is index 3.
    assert int(jnp.argmax(weights_by_iptm['binder'])) == 0
    assert int(jnp.argmax(weights_by_ipsae['binder'])) == 3
    assert float(weights_by_iptm['binder'][0]) == pytest.approx(1.6364, abs=1e-3)
    assert float(weights_by_ipsae['binder'][3]) == pytest.approx(1.9, abs=1e-3)
    assert not jnp.allclose(weights_by_iptm['binder'], weights_by_ipsae['binder'])


def test_interface_confidence_weights_still_defaults_to_iptm():
    """Backward compatibility: calling with no metric= must behave exactly as it did before
    this metric argument existed."""
    prediction = _binder_target_prediction(IPTM_PER_RESIDUE, IPSAE_PER_RESIDUE)
    assert jnp.allclose(interface_confidence_weights(prediction)['binder'],
                         interface_confidence_weights(prediction, metric='iptm_per_residue')['binder'])


def test_both_interface_weightings_are_available():
    assert INTERFACE_WEIGHTING_METRICS['interface_iptm'] == 'iptm_per_residue'
    assert INTERFACE_WEIGHTING_METRICS['interface_ipsae'] == 'ipsae_per_residue'


def test_select_best_sequence_uses_the_configured_weighting_metric():
    """The call site inside select_best_sequence must resolve the metric name through
    INTERFACE_WEIGHTING_METRICS[self.mutation_weighting], not a hardcoded 'iptm_per_residue'."""
    prediction = _binder_target_prediction(IPTM_PER_RESIDUE, IPSAE_PER_RESIDUE)
    predictions = {'complex': prediction}
    design_loss = jnp.asarray(0.0)

    sampler_iptm = SemigreedySequenceSampler(mutation_weighting='interface_iptm')
    sampler_iptm.select_best_sequence(design_loss, predictions)
    assert int(jnp.argmax(sampler_iptm.chain_interface_weights['binder'])) == 0

    sampler_ipsae = SemigreedySequenceSampler(mutation_weighting='interface_ipsae')
    sampler_ipsae.select_best_sequence(design_loss, predictions)
    assert int(jnp.argmax(sampler_ipsae.chain_interface_weights['binder'])) == 3

    assert not jnp.allclose(sampler_iptm.chain_interface_weights['binder'], sampler_ipsae.chain_interface_weights['binder'])


def test_interface_weights_gating_also_recognizes_interface_ipsae():
    """interface_weights() (used during propose_sequence_mutation) gates on membership in
    INTERFACE_WEIGHTING_METRICS, not equality with the single string 'interface_iptm', so
    'interface_ipsae' must also use the stored chain_interface_weights rather than falling
    back to a uniform 1.0."""
    sampler = SemigreedySequenceSampler(mutation_weighting='interface_ipsae')
    sampler.chain_interface_weights = {'binder': jnp.asarray([2.0, 0.5])}
    shared_chains = {'binder': _chain_protein(2, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])}
    weights = sampler.interface_weights(('binder',), shared_chains)
    assert jnp.allclose(weights, jnp.asarray([2.0, 0.5]))
