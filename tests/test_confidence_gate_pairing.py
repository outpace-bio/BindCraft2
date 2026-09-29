"""A confidence metric that also has a per-stage spelling is gated twice over.

`min_iptm_final` / `min_ipsae_final` reach the final stage through
`design_stage_filters`, while `filters.i_pTM` / `filters.i_pSAE` reach the campaign
through `settings['filters']`. The two are read by different code and resolve
conflicts in opposite directions, so the settled pattern -- the one main has run
since i_pTM held the gating seat -- is that the gating metric declares both halves
and they agree. These tests pin that a campaign cannot declare half of it.

Metrics with no per-stage spelling (pTM, i_pAE, Unbound_Binder_pLDDT) are not paired
and are deliberately not covered here.
"""
import pytest

from bindcraft.settings import load_settings

PAIRS = [('min_ipsae_final', 'i_pSAE'), ('min_iptm_final', 'i_pTM')]
PAIR_IDS = [metric for _setting, metric in PAIRS]


@pytest.mark.parametrize('setting_name,metric', PAIRS, ids=PAIR_IDS)
def test_declaring_only_the_filters_half_is_rejected(setting_name, metric):
    """The confounder: filters.<metric> alone leaves min_<metric>_final at its default,
    and the final stage reads that default rather than the threshold the campaign asked for."""
    with pytest.raises(ValueError) as raised:
        load_settings({'filters': {metric: {'threshold': 0.9}}})
    assert setting_name in str(raised.value) and metric in str(raised.value)


@pytest.mark.parametrize('setting_name,metric', PAIRS, ids=PAIR_IDS)
def test_declaring_only_the_stage_half_is_accepted_and_propagates(setting_name, metric):
    """min_<metric>_final alone is safe: the propagation step copies it into the filters
    block, so the two halves end up in agreement without the campaign restating it."""
    settings = load_settings({setting_name: 0.8})
    assert settings['filters'][metric]['threshold'] == pytest.approx(0.8)


@pytest.mark.parametrize('setting_name,metric', PAIRS, ids=PAIR_IDS)
def test_declaring_both_halves_in_agreement_is_accepted(setting_name, metric):
    """The published pattern, as main writes it for i_pTM."""
    settings = load_settings({setting_name: 0.8, 'filters': {metric: {'threshold': 0.8}}})
    assert settings['filters'][metric]['threshold'] == pytest.approx(0.8)
    assert settings[setting_name] == pytest.approx(0.8)


@pytest.mark.parametrize('setting_name,metric', PAIRS, ids=PAIR_IDS)
def test_declaring_both_halves_in_disagreement_is_rejected(setting_name, metric):
    """settings.py resolves this in favour of the filters block and filters.py resolves it
    in favour of the stage setting, so whichever value wins depends on which reads it."""
    with pytest.raises(ValueError) as raised:
        load_settings({setting_name: 0.8, 'filters': {metric: {'threshold': 0.9}}})
    message = str(raised.value)
    assert setting_name in message and metric in message
    assert '0.8' in message and '0.9' in message


def test_the_shipped_defaults_are_paired():
    """Zero blast radius: default.json already agrees on both pairs, so a campaign that
    overrides neither is untouched by the check."""
    settings = load_settings({})
    assert settings['filters']['i_pSAE']['threshold'] == pytest.approx(settings['min_ipsae_final'])
    assert settings['filters']['i_pTM']['threshold'] == settings.get('min_iptm_final')


def test_an_unpaired_metric_may_still_be_set_from_its_filters_block_alone():
    """pTM has no min_ptm_harden equivalent, so nothing shadows it and the filters block
    alone stays the correct way to set it."""
    settings = load_settings({'filters': {'pTM': {'threshold': 0.6}}})
    assert settings['filters']['pTM']['threshold'] == pytest.approx(0.6)
