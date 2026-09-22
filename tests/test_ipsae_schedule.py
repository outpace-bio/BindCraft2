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

    This fails if any of the three metric reads in the scheduler is left on metrics['iptm']."""
    rounds = [{'t': _prediction(0.20, 0.90)},   # good ipSAE, poor ipTM
              {'t': _prediction(0.90, 0.20)}]   # good ipTM, poor ipSAE

    by_iptm = _schedule_with_one_target('iptm')
    for round_predictions in rounds:
        by_iptm.record_stage_peak(round_predictions)
    assert float(by_iptm.stage_peak_predictions['t'].metrics['iptm']) == pytest.approx(0.90, abs=1e-6)

    by_ipsae = _schedule_with_one_target('ipsae')
    for round_predictions in rounds:
        by_ipsae.record_stage_peak(round_predictions)
    assert float(by_ipsae.stage_peak_predictions['t'].metrics['ipsae']) == pytest.approx(0.90, abs=1e-6)
