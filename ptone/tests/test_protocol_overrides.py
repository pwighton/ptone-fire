# Tests for ptone/protocol_overrides.py: settings by protocol name and sequence type

import json
import os

import pytest

from ptone.protocol_overrides import apply_protocol_overrides, describe_rule, get_rules, matching_rules

repoDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The example in ptone/protocol_overrides.py
RULES = [{'matchProtocolName': '*tse*', 'medianFilterWindowS': '2'},
         {'matchProtocolName': '*swi*', 'medianFilterWindowS': '5'},
         {'matchProtocolName': '*fl2d*', 'medianFilterWindowS': '3'},
         {'matchSequenceType': 'TurboSpinEcho', 'ptoneTxDB': '70'},
         {'matchSequenceType': 'Flash', 'ptoneTxDB': '70'}]

def config_with(rules, **params):
    return {'version': '0.0.1', 'parameters': dict(params, protocolOverrides=rules)}

@pytest.mark.parametrize('protocol, sequenceType, indices', [
    ('t2_tse_tra_dark-fluid--m-pt', 'TurboSpinEcho', [0, 3]),
    ('T2_TSE_TRA', 'turbospinecho', [0, 3]),          # Case doesn't matter
    ('TRA SWI--nm-pt', 'Flash', [1, 4]),             # As the scanner sends it, with the space
    ('t2_fl2d_tra_hemo--nm-pt', 'Flash', [2, 4]),
    ('t1_mprage--nm-pt', 'Flash', [4]),              # Only the sequence type matches
    ('t1_mprage--nm-pt', None, []),                  # No sequence type in the header
    (None, 'TurboSpinEcho', [3]),                    # No protocol name in the header
    ('ep2d_bold', 'EPI', []),
])
def test_matching_rules(protocol, sequenceType, indices):
    assert matching_rules(RULES, {'protocolName': protocol, 'sequenceType': sequenceType}) == indices

def test_patterns():
    scan = {'protocolName': 't2_tse', 'sequenceType': 'TurboSpinEcho'}
    assert matching_rules([{'matchProtocolName': 't?_tse*'}], scan) == [0]          # '?' is one character
    assert matching_rules([{'matchProtocolName': 'tse'}], scan) == []               # Patterns match the whole name
    assert matching_rules([{'matchSequenceType': 'Turbo*'}], scan) == [0]

def test_both_conditions_must_match():
    rule = {'matchSequenceType': 'Flash', 'matchProtocolName': '*mprage*', 'bulkMotionMethod': 'none'}
    assert matching_rules([rule], {'protocolName': 't1_mprage', 'sequenceType': 'Flash'}) == [0]
    assert matching_rules([rule], {'protocolName': 'TRA SWI', 'sequenceType': 'Flash'}) == []
    assert matching_rules([rule], {'protocolName': 't1_mprage', 'sequenceType': None}) == []

def test_apply_protocol_overrides():
    config = config_with(RULES, medianFilterWindowS='3', ptoneTxDB='60')
    newConfig, applied = apply_protocol_overrides(config, 'TRA SWI--nm-pt', 'Flash')
    assert newConfig['parameters']['medianFilterWindowS'] == '5' and newConfig['parameters']['ptoneTxDB'] == '70'
    assert config['parameters']['medianFilterWindowS'] == '3'          # The original is unchanged
    assert newConfig['version'] == '0.0.1'
    assert applied == {'protocolName': 'TRA SWI--nm-pt', 'sequenceType': 'Flash', 'protocolOverrideIndices': [1, 4],
                       'protocolOverrideSettings': {'medianFilterWindowS': '5', 'ptoneTxDB': '70'}}
    sameConfig, applied = apply_protocol_overrides(config, 'ep2d_bold', 'EPI')
    assert sameConfig == config and applied['protocolOverrideIndices'] == [] and applied['protocolOverrideSettings'] is None

def test_later_rules_win():
    rules = [{'matchSequenceType': 'Flash', 'medianFilterWindowS': '3', 'ptoneTxDB': '60'},
             {'matchProtocolName': '*swi*', 'medianFilterWindowS': '5'}]
    newConfig, applied = apply_protocol_overrides(config_with(rules), 'TRA SWI', 'Flash')
    assert applied['protocolOverrideSettings'] == {'medianFilterWindowS': '5', 'ptoneTxDB': '60'}
    newConfig, applied = apply_protocol_overrides(config_with(rules[::-1]), 'TRA SWI', 'Flash')
    assert applied['protocolOverrideSettings'] == {'medianFilterWindowS': '3', 'ptoneTxDB': '60'}

def test_describe_rule():
    assert describe_rule(1, RULES[1]) == "rule 1 (matchProtocolName '*swi*'), which sets medianFilterWindowS = 5"
    assert describe_rule(0, {'matchSequenceType': 'Flash', 'matchProtocolName': '*x*'}) == \
        "rule 0 (matchProtocolName '*x*', matchSequenceType 'Flash'), which sets nothing"

@pytest.mark.parametrize('config', [None, 'pilottone', {}, {'parameters': {}}, config_with([]), config_with('')])
def test_no_rules(config):
    assert get_rules(config) == []
    assert apply_protocol_overrides(config, 'x', 'Flash')[0] is config

def test_rules_as_json_text():
    # e.g. from the client's --set protocolOverrides=...
    assert get_rules(config_with(json.dumps(RULES))) == RULES

@pytest.mark.parametrize('rules, message', [
    ({'matchProtocolName': '*tse*'}, 'must be a list'),
    ('[{"matchProtocolName": "*tse*"', "isn't valid JSON"),
    (['*tse*'], 'must be an object'),
    ([{'medianFilterWindowS': '2'}], 'needs a condition'),
    ([{'match': '*tse*', 'medianFilterWindowS': '2'}], "now 'matchProtocolName'"),
    ([{'matchProtocol': '*tse*'}], 'unknown condition'),
    ([{'matchProtocolName': 2}], 'must be a pattern'),
    ([{'matchProtocolName': '*', 'protocolOverrides': []}], "can't set protocolOverrides"),
])
def test_malformed_rules(rules, message):
    with pytest.raises(ValueError, match=message):
        apply_protocol_overrides(config_with(rules), 'x', 'Flash')

@pytest.mark.parametrize('configFile', ['pilottone.json', 'pilottone_offline.json'])
@pytest.mark.parametrize('protocol, sequenceType, windowS', [
    ('t2_tse_tra_dark-fluid--m-pt', 'TurboSpinEcho', '2'),
    ('t2_tse_tra_dark-fluid ARIA--nm-pt', 'TurboSpinEcho', '2'),
    ('TRA SWI--nm-pt', 'Flash', '5'),
    ('t2_fl2d_tra_hemo--nm-pt', 'Flash', '3'),
    ('t2_fl2d_tra_hemo ARIA--m-pt', 'Flash', '3'),
    ('t1_mprage--nm-pt', 'Flash', '3'),
])
def test_repository_configs(configFile, protocol, sequenceType, windowS):
    # The rules in the repository's configs give the chosen window lengths for the test data's protocols
    with open(os.path.join(repoDir, configFile)) as f:
        config = json.load(f)
    assert apply_protocol_overrides(config, protocol, sequenceType)[0]['parameters']['medianFilterWindowS'] == windowS
