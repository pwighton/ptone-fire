# Settings by protocol: the protocolOverrides setting in pilottone.json.
#
# protocolOverrides is a list of rules.  Each rule has one or more conditions on the scan, from its MRD
# header, and the settings to use when they all hold:
#   matchProtocolName  a pattern for the protocol name, as the scanner sends it (e.g. 'TRA SWI--nm-pt')
#   matchSequenceType  a pattern for the sequence type (sequenceParameters.sequence_type, e.g. 'TurboSpinEcho',
#                      'Flash'; FLASH, SWI and MPRAGE are all 'Flash', so it can't tell those apart)
# Patterns are shell-style wildcards ('*' any characters, '?' one character, '[abc]' one of a, b, c) matching
# the whole name, and case doesn't matter.  A scan without the name or type in its header matches no rule
# that needs it.
#
# Every rule that matches is applied, in list order, so a later rule's value for a setting replaces an
# earlier one's.  A rule can set any setting except protocolOverrides itself.  For example:
#   "protocolOverrides": [
#       {"matchProtocolName": "*tse*",  "medianFilterWindowS": "2"},
#       {"matchProtocolName": "*swi*",  "medianFilterWindowS": "5"},
#       {"matchProtocolName": "*fl2d*", "medianFilterWindowS": "3"},
#       {"matchSequenceType": "TurboSpinEcho", "ptoneTxDB": "70"},
#       {"matchSequenceType": "Flash",         "ptoneTxDB": "70"}
#   ]
# gives a TSE protocol 2 s windows and transmit gain 70 dB, and an SWI protocol (sequence type 'Flash') 5 s
# windows and 70 dB.  A rule with both conditions applies only to scans meeting both, e.g.
#   {"matchSequenceType": "Flash", "matchProtocolName": "*mprage*", "bulkMotionMethod": "none"}
#
# Used by pilottone.py (live and offline) and by bulk_motion_eval.py (--config), so the evaluation picks
# each scan's settings exactly as pilottone.py does.

import copy
import fnmatch
import json

# Rule conditions: the key in a rule, and the scan property it matches
CONDITIONS = {
    'matchProtocolName': 'protocolName',
    'matchSequenceType': 'sequenceType',
}

def get_rules(config):
    """
    The protocolOverrides rules in a config ({'parameters': {...}}), checked: a list of dicts, each with at
    least one condition (CONDITIONS) whose pattern is a string.  The list can also be given as JSON text
    (e.g. from a --set on the client command line).  Returns [] if there are none; raises ValueError if
    they're malformed.
    """
    if not isinstance(config, dict) or not isinstance(config.get('parameters'), dict):
        return []
    rules = config['parameters'].get('protocolOverrides', [])
    if isinstance(rules, str):
        if rules.strip() == '':
            return []
        try:
            rules = json.loads(rules)
        except ValueError as e:
            raise ValueError("protocolOverrides isn't valid JSON: %s" % e)
    if not isinstance(rules, list):
        raise ValueError("protocolOverrides must be a list of rules (got %s)" % type(rules).__name__)
    for i, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise ValueError("protocolOverrides rule %d must be an object (got %r)" % (i, rule))
        if 'match' in rule:
            raise ValueError("protocolOverrides rule %d uses 'match', which is now 'matchProtocolName'" % i)
        # Keys starting with 'match' are conditions, so a misspelt one is caught rather than taken as a setting
        unknown = [k for k in rule if k.startswith('match') and k not in CONDITIONS]
        if unknown:
            raise ValueError("protocolOverrides rule %d: unknown condition(s) %s (known: %s)"
                             % (i, ', '.join(unknown), ', '.join(CONDITIONS)))
        conditions = [k for k in rule if k in CONDITIONS]
        if not conditions:
            raise ValueError("protocolOverrides rule %d needs a condition (%s): %r" % (i, ' or '.join(CONDITIONS), rule))
        for k in conditions:
            if not isinstance(rule[k], str):
                raise ValueError("protocolOverrides rule %d: %s must be a pattern (got %r)" % (i, k, rule[k]))
        if 'protocolOverrides' in rule:
            raise ValueError("protocolOverrides rule %d can't set protocolOverrides" % i)
    return rules

def rule_matches(rule, scan):
    """True if every condition in the rule matches the scan ({'protocolName': ..., 'sequenceType': ...})."""
    for key, prop in CONDITIONS.items():
        if key in rule:
            value = scan.get(prop)
            if not value or not fnmatch.fnmatchcase(value.lower(), rule[key].lower()):
                return False
    return True

def matching_rules(rules, scan):
    """Indices of the rules matching the scan, in list order."""
    return [i for i, rule in enumerate(rules) if rule_matches(rule, scan)]

def rule_settings(rule):
    """The settings a rule sets (everything but its conditions)."""
    return {k: v for k, v in rule.items() if k not in CONDITIONS}

def describe_rule(index, rule):
    """e.g. "rule 0 (matchProtocolName '*tse*'), which sets medianFilterWindowS = 2", for log messages."""
    conditions = ', '.join("%s '%s'" % (k, rule[k]) for k in CONDITIONS if k in rule)
    settings = ', '.join('%s = %s' % kv for kv in rule_settings(rule).items())
    return "rule %d (%s), which sets %s" % (index, conditions, settings or 'nothing')

def apply_protocol_overrides(config, protocolName, sequenceType=None):
    """
    The config with the settings of every protocolOverrides rule matching the scan applied, in list order
    (a copy; config itself is unchanged), and a description of what was applied:
      {'protocolName': ..., 'sequenceType': ...,
       'protocolOverrideIndices': [indices of the matching rules],
       'protocolOverrideSettings': {setting: value} as set by them (later rules winning), or None if none}
    Raises ValueError if the rules are malformed.
    """
    rules = get_rules(config)
    scan = {'protocolName': protocolName, 'sequenceType': sequenceType}
    indices = matching_rules(rules, scan)
    applied = dict(scan, protocolOverrideIndices=indices, protocolOverrideSettings=None)
    if not indices:
        return config, applied
    settings = {}
    for i in indices:
        settings.update(rule_settings(rules[i]))
    newConfig = copy.deepcopy(config)
    newConfig['parameters'].update(settings)
    applied['protocolOverrideSettings'] = settings
    return newConfig, applied
