"""ipSAE gates the settled structure, not the trajectory.

The A/B evaluation on this branch settled it: ipTM is the better signal while the
trajectory is still moving, and ipSAE is the better filter once the structure has
settled. So the confidence gates that run during optimisation are the ipTM ones, and
ipSAE gates only at `final`. `default.json` already ships that policy -- every
non-final min_ipsae_* is unset and every non-final min_iptm_* is 0.5 -- and these
tests stop a campaign overriding against it.

Bare `max_detarget_ipsae` is deliberately NOT covered: it carries no stage suffix and
is the multitarget swap-scheduler ceiling read at target_schedule.py:24, a different
mechanism from the per-stage gates.
"""
import pytest

from bindcraft.settings import load_settings

TRAJECTORY_STAGES = ('screen', 'refine', 'anneal', 'harden', 'mutate')
IPSAE_GATES = [f'{prefix}_{stage}' for prefix in ('min_ipsae', 'max_detarget_ipsae') for stage in TRAJECTORY_STAGES]


@pytest.mark.parametrize('setting_name', IPSAE_GATES)
def test_an_ipsae_gate_on_a_trajectory_stage_is_rejected(setting_name):
    with pytest.raises(ValueError) as raised:
        load_settings({setting_name: 0.5})
    message = str(raised.value)
    assert setting_name in message
    assert 'min_ipsae_final' in message


@pytest.mark.parametrize('setting_name', IPSAE_GATES)
def test_explicitly_unsetting_a_trajectory_ipsae_gate_is_allowed(setting_name):
    """A null is the campaign saying 'no gate here', which is the policy, not a breach of it."""
    assert load_settings({setting_name: None}).get(setting_name) is None


@pytest.mark.parametrize('stage', TRAJECTORY_STAGES)
def test_the_iptm_gate_on_a_trajectory_stage_is_untouched(stage):
    """ipTM is the signal the trajectory is supposed to be gated on."""
    assert load_settings({f'min_iptm_{stage}': 0.6})[f'min_iptm_{stage}'] == pytest.approx(0.6)


def test_ipsae_at_final_is_the_supported_place_for_it():
    assert load_settings({'min_ipsae_final': 0.8})['filters']['i_pSAE']['threshold'] == pytest.approx(0.8)


def test_the_detarget_ipsae_ceiling_at_final_is_allowed():
    assert load_settings({'max_detarget_ipsae_final': 0.3})['max_detarget_ipsae_final'] == pytest.approx(0.3)


def test_the_unsuffixed_detarget_ipsae_ceiling_is_a_different_setting_and_is_allowed():
    """max_detarget_ipsae without a stage is the multitarget swap ceiling, not a stage gate."""
    assert load_settings({'max_detarget_ipsae': 0.4})['max_detarget_ipsae'] == pytest.approx(0.4)


def test_the_shipped_defaults_obey_the_policy():
    settings = load_settings({})
    for stage in TRAJECTORY_STAGES:
        assert settings.get(f'min_ipsae_{stage}') is None
        assert settings.get(f'max_detarget_ipsae_{stage}') is None
    assert settings['min_ipsae_final'] == pytest.approx(0.7)
    assert settings['min_iptm_anneal'] == pytest.approx(0.5)
