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


from bindcraft.preflight import undecided_avoidance


class _Settings:
    def __init__(self, settings, prepared_states=()):
        self.settings = settings
        self.prepared_states = prepared_states


def test_ipsae_ceiling_counts_as_deciding_an_off_target():
    decided = _Settings({'max_detarget_ipsae_harden': 0.3})
    assert undecided_avoidance(decided) == ''
