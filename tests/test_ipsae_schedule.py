from bindcraft.target_schedule import MultitargetSchedule


def _schedule(metric):
    return MultitargetSchedule(targets={}, target_objectives={}, iterations=10, confidence_metric=metric)


def test_schedule_defaults_to_iptm():
    assert MultitargetSchedule(targets={}, target_objectives={}, iterations=10).confidence_metric == 'iptm'


def test_schedule_accepts_ipsae():
    assert _schedule('ipsae').confidence_metric == 'ipsae'
