# Tests for ptone/protocol_overrides.py: settings by protocol name

import json
import os

import pytest

from ptone.protocol_overrides import apply_protocol_overrides, get_rules, match_rule

repoDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RULES = [{'match': '*tse*', 'medianFilterWindowS': '2'},
         {'match': '*swi*', 'medianFilterWindowS': '5', 'ptonePlot': 'false'},
         {'match': '*', 'medianFilterWindowS': '4'}]

def config_with(rules, **params):
    return {'version': '0.0.1', 'parameters': dict(params, protocolOverrides=rules)}

@pytest.mark.parametrize('protocol, index', [
    ('t2_tse_tra_dark-fluid--m-pt', 0),
    ('T2_TSE_TRA', 0),                          # Case doesn't matter
    ('TRA SWI--nm-pt', 1),                      # As the scanner sends it, with the space
    ('t2_fl2d_tra_hemo--nm-pt', 2),             # Only the catch-all matches
    ('tse_and_swi', 0),                         # Several match: the first wins
])
def test_match_rule(protocol, index):
    assert match_rule(RULES, protocol)[0] == index

def test_match_rule_no_match_or_no_name():
    rules = RULES[:2]
    assert match_rule(rules, 't1_mprage--nm-pt') == (None, None)
    assert match_rule(rules, None) == (None, None)
    assert match_rule(rules, '') == (None, None)
    assert match_rule([{'match': 't?_tse*'}], 't2_tse')[0] == 0          # '?' is one character
    assert match_rule([{'match': 'tse'}], 't2_tse')[0] is None           # Patterns match the whole name

def test_apply_protocol_overrides():
    config = config_with(RULES[:2], medianFilterWindowS='3', ptonePlot='true')
    newConfig, applied = apply_protocol_overrides(config, 'TRA SWI--nm-pt')
    assert newConfig['parameters']['medianFilterWindowS'] == '5' and newConfig['parameters']['ptonePlot'] == 'false'
    assert config['parameters']['medianFilterWindowS'] == '3'          # The original is unchanged
    assert newConfig['version'] == '0.0.1'
    assert applied == {'protocolName': 'TRA SWI--nm-pt', 'protocolOverrideIndex': 1, 'protocolOverrideMatch': '*swi*',
                       'protocolOverrideSettings': {'medianFilterWindowS': '5', 'ptonePlot': 'false'}}
    sameConfig, applied = apply_protocol_overrides(config, 't1_mprage')
    assert sameConfig == config and applied['protocolOverrideMatch'] is None and applied['protocolOverrideSettings'] is None

@pytest.mark.parametrize('config', [None, 'pilottone', {}, {'parameters': {}}, config_with([]), config_with('')])
def test_no_rules(config):
    assert get_rules(config) == []
    assert apply_protocol_overrides(config, 'x')[0] is config

def test_rules_as_json_text():
    # e.g. from the client's --set protocolOverrides=...
    assert get_rules(config_with(json.dumps(RULES))) == RULES

@pytest.mark.parametrize('rules, message', [
    ({'match': '*tse*'}, 'must be a list'),
    ('[{"match": "*tse*"', "isn't valid JSON"),
    ([{'medianFilterWindowS': '2'}], "'match' pattern"),
    (['*tse*'], "'match' pattern"),
    ([{'match': '*', 'protocolOverrides': []}], "can't set protocolOverrides"),
])
def test_malformed_rules(rules, message):
    with pytest.raises(ValueError, match=message):
        apply_protocol_overrides(config_with(rules), 'x')

@pytest.mark.parametrize('configFile', ['pilottone.json', 'pilottone_offline.json'])
@pytest.mark.parametrize('protocol, windowS', [
    ('t2_tse_tra_dark-fluid--m-pt', '2'),
    ('t2_tse_tra_dark-fluid ARIA--nm-pt', '2'),
    ('TRA SWI--nm-pt', '5'),
    ('t2_fl2d_tra_hemo--nm-pt', '3'),
    ('t2_fl2d_tra_hemo ARIA--m-pt', '3'),
    ('t1_mprage--nm-pt', '3'),
])
def test_repository_configs(configFile, protocol, windowS):
    # The rules in the repository's configs give the chosen window lengths for the test data's protocol names
    with open(os.path.join(repoDir, configFile)) as f:
        config = json.load(f)
    assert apply_protocol_overrides(config, protocol)[0]['parameters']['medianFilterWindowS'] == windowS
