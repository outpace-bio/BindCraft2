"""A null *_final confidence threshold means the key is absent.

`min_iptm_final: null` used to reach `float(None)` at settings.py:482 and die with a
TypeError; the documented workaround was to omit the key. These tests make the null
spelling mean the same thing as omitting it, so the shipped default applies.

Note the consequence, pinned below: because null now resolves to the default rather
than to "no gate", turning the final ipSAE gate off is spelled as a threshold of 0.0
on both halves of the pair, not as a null.
"""
import pytest

from bindcraft.settings import FINAL_CONFIDENCE_FILTERS, load_settings

PAIRS = sorted(FINAL_CONFIDENCE_FILTERS.items())
PAIR_IDS = [setting_name for setting_name, _metric in PAIRS]


@pytest.mark.parametrize('setting_name,metric', PAIRS, ids=PAIR_IDS)
def test_a_null_threshold_does_not_crash(setting_name, metric):
    """Regression: this raised TypeError: float() argument must be a string or a real number."""
    load_settings({setting_name: None})


@pytest.mark.parametrize('setting_name,metric', PAIRS, ids=PAIR_IDS)
def test_a_null_threshold_leaves_the_filter_at_its_default(setting_name, metric):
    assert load_settings({setting_name: None})['filters'][metric]['threshold'] == load_settings({})['filters'][metric]['threshold']


def test_a_null_threshold_restores_the_shipped_default_rather_than_sticking_as_none():
    """The crisp statement of 'unset': min_ipsae_final ships at 0.7, so nulling it gives 0.7,
    not None. Anything less would leave the pair disagreeing with filters.i_pSAE."""
    assert load_settings({'min_ipsae_final': None})['min_ipsae_final'] == pytest.approx(0.7)


def test_turning_the_final_ipsae_gate_off_is_spelled_as_zero_on_both_halves():
    """Since null means 'take the default', an always-passing gate is how a campaign opts out."""
    settings = load_settings({'min_ipsae_final': 0.0, 'filters': {'i_pSAE': {'threshold': 0.0}}})
    assert settings['filters']['i_pSAE']['threshold'] == pytest.approx(0.0)
