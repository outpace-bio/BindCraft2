import jax.numpy as jnp
import pytest

from bindcraft.target_schedule import MultitargetSchedule
from bindcraft.protein import StructurePrediction


def _schedule(metric):
    return MultitargetSchedule(targets={}, target_objectives={}, iterations=10, confidence_metric=metric)


def test_schedule_defaults_to_iptm():
    assert MultitargetSchedule(targets={}, target_objectives={}, iterations=10).confidence_metric == 'iptm'


def test_schedule_accepts_ipsae():
    assert _schedule('ipsae').confidence_metric == 'ipsae'


def _prediction(iptm, ipsae):
    return StructurePrediction(protein_complex={},
                               metrics={'iptm': jnp.asarray(iptm), 'ipsae': jnp.asarray(ipsae)})


def _schedule_with_one_target(metric):
    return MultitargetSchedule(targets={'t': object()}, target_objectives={'t': 'binds'},
                               iterations=10, confidence_metric=metric)


def test_the_stage_peak_follows_the_configured_metric():
    """The two rounds disagree about which is better, so the peak reveals which metric is read.

    Breaks symmetry by setting round 2 to (0.95, 0.50) so individual mis-keyed reads are caught."""
    rounds = [{'t': _prediction(0.20, 0.90)},   # good ipSAE, poor ipTM
              {'t': _prediction(0.95, 0.50)}]   # good ipTM, okay ipSAE

    by_iptm = _schedule_with_one_target('iptm')
    for round_predictions in rounds:
        by_iptm.record_stage_peak(round_predictions)
    assert float(by_iptm.stage_peak_predictions['t'].metrics['iptm']) == pytest.approx(0.95, abs=1e-6)

    by_ipsae = _schedule_with_one_target('ipsae')
    for round_predictions in rounds:
        by_ipsae.record_stage_peak(round_predictions)
    assert float(by_ipsae.stage_peak_predictions['t'].metrics['ipsae']) == pytest.approx(0.90, abs=1e-6)


def test_the_swap_decision_follows_the_configured_metric():
    """line 220 feeds target_transition.reached, i.e. the actual swap. With the default
    swap threshold of 0.5, a prediction that is confident by ipTM and weak by ipSAE must
    swap targets under 'iptm' and hold under 'ipsae'."""
    confident_by_iptm_only = {'a': _prediction(0.90, 0.10)}

    by_iptm = MultitargetSchedule(targets={'a': object(), 'b': object()},
                                  target_objectives={'a': 'binds', 'b': 'binds'},
                                  iterations=100, confidence_metric='iptm', swap_patience=99)
    by_iptm.select_protein_states({'a': {}, 'b': {}}, confident_by_iptm_only)
    assert by_iptm.active_target_index == 1
    assert by_iptm.iterations_on_target == 0

    by_ipsae = MultitargetSchedule(targets={'a': object(), 'b': object()},
                                   target_objectives={'a': 'binds', 'b': 'binds'},
                                   iterations=100, confidence_metric='ipsae', swap_patience=99)
    by_ipsae.select_protein_states({'a': {}, 'b': {}}, confident_by_iptm_only)
    assert by_ipsae.active_target_index == 0
    assert by_ipsae.iterations_on_target == 1
