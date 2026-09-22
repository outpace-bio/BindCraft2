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


from bindcraft.settings import BinderDesignSettings, BinderSettings, TargetSettings
from bindcraft.target_schedule import build_target_schedule, objective_gated_reached


def _built_schedule(settings):
    """build_target_schedule off a real BinderDesignSettings, with one on- and one off-target.

    This covers the settings reads at build_target_schedule itself, which nothing else drives:
    hardcoding confidence_metric='iptm' there used to leave the whole suite green."""
    design_settings = BinderDesignSettings(
        targets=[TargetSettings('on', 'on.pdb', weight=1.0), TargetSettings('off', 'off.pdb', weight=-1.0)],
        binder=BinderSettings(lengths=(50,)), settings=settings)
    target_states = {state.name: {state.target_chain: object()} for state in design_settings.prepared_states}
    return build_target_schedule(design_settings, target_states, 100, 'screen')


def test_build_target_schedule_reads_the_swap_metric():
    assert _built_schedule({}).confidence_metric == 'iptm'
    assert _built_schedule({'multitarget_swap_metric': 'ipsae'}).confidence_metric == 'ipsae'


def test_an_ipsae_campaign_gets_the_ipsae_detarget_ceiling():
    """max_detarget_ipsae was registered and asserted by a test but nothing read it: the
    schedule hardcoded max_detarget_iptm."""
    schedule = _built_schedule({'multitarget_swap_metric': 'ipsae', 'max_detarget_ipsae': 0.05, 'max_detarget_iptm': 0.4})
    assert schedule.max_detarget_confidence == 0.05


def test_an_iptm_campaign_still_gets_the_iptm_detarget_ceiling():
    schedule = _built_schedule({'multitarget_swap_metric': 'iptm', 'max_detarget_iptm': 0.35, 'max_detarget_ipsae': 0.05})
    assert schedule.max_detarget_confidence == 0.35


def test_the_detarget_ceiling_defaults_are_unchanged():
    """The safety promise: a campaign with no ipSAE settings behaves exactly as it did."""
    schedule = _built_schedule({})
    assert schedule.confidence_metric == 'iptm'
    assert schedule.max_detarget_confidence == 0.4


def test_an_ipsae_detarget_visit_is_decided_against_the_ipsae_ceiling():
    """The defect this fixes. ipSAE runs an order of magnitude below ipTM, so an ipTM ceiling
    of 0.4 is met on the very first round of every detarget visit and detargeting silently
    stops happening. With the ipSAE ceiling wired, a binder still on the off-target holds the
    visit and one that has come off releases it."""
    schedule = _built_schedule({'multitarget_swap_metric': 'ipsae', 'max_detarget_ipsae': 0.05, 'max_detarget_iptm': 0.4})
    assert objective_gated_reached(schedule, 'detarget', 0.30, 0.0) is False
    assert objective_gated_reached(schedule, 'detarget', 0.04, 0.0) is True


def test_an_iptm_detarget_visit_keeps_its_old_decision():
    schedule = _built_schedule({'max_detarget_iptm': 0.4})
    assert objective_gated_reached(schedule, 'detarget', 0.55, 0.0) is False
    assert objective_gated_reached(schedule, 'detarget', 0.30, 0.0) is True
