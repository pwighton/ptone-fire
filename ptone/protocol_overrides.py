# Settings by protocol name: the protocolOverrides setting in pilottone.json.
#
# protocolOverrides is an ordered list of rules.  Each has a 'match' pattern and the settings to use when the
# scan's protocol name (from the MRD header, as the scanner sends it) matches it, e.g.
#   "protocolOverrides": [
#       {"match": "*tse*", "medianFilterWindowS": "2"},
#       {"match": "*swi*", "medianFilterWindowS": "5"}
#   ]
# Patterns are shell-style wildcards ('*' any characters, '?' one character, '[abc]' one of a, b, c), and
# case doesn't matter.  The first rule that matches is used; if none does (or there's no protocol name), the
# settings are used as they are.  A rule can set any setting except protocolOverrides itself.
#
# Used by pilottone.py (live and offline) and by bulk_motion_eval.py (--config), so the evaluation picks
# each scan's settings exactly as pilottone.py does.

import copy
import fnmatch
import json

def get_rules(config):
    """
    The protocolOverrides rules in a config ({'parameters': {...}}), checked: a list of dicts, each with a
    'match' string.  The list can also be given as JSON text (e.g. from a --set on the client command line).
    Returns [] if there are none; raises ValueError if they're malformed.
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
        if not isinstance(rule, dict) or not isinstance(rule.get('match'), str):
            raise ValueError("protocolOverrides rule %d must be an object with a 'match' pattern (got %r)" % (i, rule))
        if 'protocolOverrides' in rule:
            raise ValueError("protocolOverrides rule %d ('%s') can't set protocolOverrides" % (i, rule['match']))
    return rules

def match_rule(rules, protocolName):
    """(index, rule) of the first rule whose pattern matches protocolName (ignoring case), or (None, None)."""
    if not protocolName:
        return None, None
    for i, rule in enumerate(rules):
        if fnmatch.fnmatchcase(protocolName.lower(), rule['match'].lower()):
            return i, rule
    return None, None

def apply_protocol_overrides(config, protocolName):
    """
    The config with the settings of the first protocolOverrides rule matching protocolName applied (a copy;
    config itself is unchanged), and a description of what was applied:
      {'protocolName': ..., 'protocolOverrideIndex': i or None, 'protocolOverrideMatch': pattern or None,
       'protocolOverrideSettings': {setting: value} or None}
    Raises ValueError if the rules are malformed.
    """
    rules = get_rules(config)
    index, rule = match_rule(rules, protocolName)
    applied = {'protocolName': protocolName, 'protocolOverrideIndex': index,
               'protocolOverrideMatch': rule['match'] if rule else None,
               'protocolOverrideSettings': None}
    if rule is None:
        return config, applied
    settings = {k: v for k, v in rule.items() if k != 'match'}
    newConfig = copy.deepcopy(config)
    newConfig['parameters'].update(settings)
    applied['protocolOverrideSettings'] = settings
    return newConfig, applied
