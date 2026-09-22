import pytest

from bindcraft.settings import CAMPAIGN_SETTING_NAMES, FINAL_CONFIDENCE_FILTERS, known_campaign_settings


def test_final_ipsae_filter_is_registered():
    assert FINAL_CONFIDENCE_FILTERS['min_ipsae_final'] == 'i_pSAE'


def test_detarget_ipsae_ceiling_is_a_campaign_setting():
    assert 'max_detarget_ipsae' in CAMPAIGN_SETTING_NAMES
    assert 'multitarget_swap_metric' in CAMPAIGN_SETTING_NAMES


def test_per_stage_ipsae_settings_are_known():
    names = known_campaign_settings()
    for stage in ('screen', 'refine', 'anneal', 'harden', 'mutate', 'final'):
        assert f'min_ipsae_{stage}' in names
        assert f'max_detarget_ipsae_{stage}' in names


def test_existing_iptm_settings_still_known():
    names = known_campaign_settings()
    assert 'min_iptm_final' in names
    assert 'min_iptm_anneal' in names
    assert 'max_detarget_iptm' in CAMPAIGN_SETTING_NAMES
    assert FINAL_CONFIDENCE_FILTERS['min_iptm_final'] == 'i_pTM'


def test_min_ipsae_final_builds_a_working_filter():
    """Regression: settings.py:482 assumed the filters block already declared i_pSAE."""
    from bindcraft.settings import load_settings
    settings = load_settings({'min_ipsae_final': 0.5})
    assert settings['filters']['i_pSAE']['threshold'] == 0.5
    assert settings['filters']['i_pSAE'].get('higher', True) is True


def test_ipsae_final_filter_is_inert_by_default():
    """ipSAE must gate nothing unless a campaign asks. A null threshold is skipped."""
    from bindcraft.settings import load_settings
    assert load_settings({})['filters']['i_pSAE']['threshold'] is None


from bindcraft.preflight import undecided_avoidance


class _Settings:
    def __init__(self, settings, prepared_states=()):
        self.settings = settings
        self.prepared_states = prepared_states


class _State:
    def __init__(self, name, objective):
        self.name, self.objective = name, objective


def test_an_off_target_with_no_ceiling_is_flagged():
    """The control: without a ceiling of either kind, the warning must fire."""
    undecided = _Settings({}, prepared_states=(_State('offtarget', 'detarget'),))
    assert 'offtarget' in undecided_avoidance(undecided)


def test_an_ipsae_ceiling_decides_an_off_target():
    decided = _Settings({'max_detarget_ipsae_harden': 0.3},
                        prepared_states=(_State('offtarget', 'detarget'),))
    assert undecided_avoidance(decided) == ''


def test_an_iptm_ceiling_still_decides_an_off_target():
    """The pre-existing path must keep working."""
    decided = _Settings({'max_detarget_iptm_harden': 0.4},
                        prepared_states=(_State('offtarget', 'detarget'),))
    assert undecided_avoidance(decided) == ''


from bindcraft.filters import design_stage_filters


def test_design_stage_filters_sets_ipsae_polarity_and_threshold_per_state():
    """Guards against a copy-paste polarity inversion in the ipSAE stage-filter block:
    on-target must be a floor (higher=True), a detarget state must be a ceiling (higher=False)."""
    design_settings = _Settings(
        {'min_ipsae_harden': 0.5, 'max_detarget_ipsae_harden': 0.3},
        prepared_states=(_State('offtarget', 'detarget'),),
    )
    protein_states = {'complex': {}, 'offtarget': {}}
    stage_filters = design_stage_filters(design_settings, protein_states, 'harden', plddt=False)

    on_target = stage_filters['i_pSAE.complex']
    assert on_target.higher is True
    assert on_target.threshold == 0.5

    detarget = stage_filters['i_pSAE.offtarget']
    assert detarget.higher is False
    assert detarget.threshold == 0.3


import jax.numpy as jnp

from bindcraft.protein import StructurePrediction


def test_stage_filters_evaluate_with_ipsae_loss_params_configured():
    """Regression: the ipSAE stage filter used to be bound from `losses.ipsae_loss.params`.

    ipsae_loss carries pae_cutoff, warmup_cutoff and temperature; ipsae_metric accepts none of
    them, so a campaign that tuned the loss validated at load and then died at the first filter
    evaluation with "ipsae_metric() got an unexpected keyword argument 'temperature'". The stage
    filter is built unconditionally, so it fired even with no ipSAE threshold and the loss at
    weight 0."""
    design_settings = _Settings(
        {'losses': {'ipsae_loss': {'weight': 0.0,
                                   'params': {'pae_cutoff': 8.0, 'warmup_cutoff': 25.0, 'temperature': 0.2}}}},
        prepared_states=(_State('complex', 'target'),),
    )
    protein_states = {'complex': {}}
    predictions = {'complex': StructurePrediction(protein_complex={},
                                                  metrics={'iptm': jnp.asarray(0.7), 'ipsae': jnp.asarray(0.12)})}
    stage_filters = design_stage_filters(design_settings, protein_states, 'harden', plddt=False)
    assert stage_filters['i_pSAE'].function(protein_states, predictions) == pytest.approx(0.12, abs=1e-6)
    assert stage_filters['i_pTM'].function(protein_states, predictions) == pytest.approx(0.7, abs=1e-6)


def test_stage_filters_evaluate_with_iptm_loss_params_configured():
    """The ipTM twin of the regression above, so the pattern stays pinned on both blocks."""
    design_settings = _Settings({'losses': {'iptm_loss': {'weight': 1.0, 'params': {}}}},
                                prepared_states=(_State('complex', 'target'),))
    protein_states = {'complex': {}}
    predictions = {'complex': StructurePrediction(protein_complex={},
                                                  metrics={'iptm': jnp.asarray(0.7), 'ipsae': jnp.asarray(0.12)})}
    stage_filters = design_stage_filters(design_settings, protein_states, 'harden', plddt=False)
    assert stage_filters['i_pTM'].function(protein_states, predictions) == pytest.approx(0.7, abs=1e-6)
