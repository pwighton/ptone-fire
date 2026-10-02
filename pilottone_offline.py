# Alias for pilottone.py, so that "-c pilottone_offline" uses pilottone_offline.json (settings for use away
# from the scanner, e.g. replaying data with client.py) instead of pilottone.json.  The server loads the
# config named by -c as a module, and the client sends <config name>.json with the data
from pilottone import process
