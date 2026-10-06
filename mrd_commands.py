# Command-line entry points, so the client and server can be run from any directory as 'mrd-client' and
# 'mrd-server', and the pilot tone scripts as 'ptone-offline-batch', 'ptone-plot' and
# 'ptone-bulk-motion-eval' (see pyproject.toml).  Each runs the script exactly as 'python client.py' or
# 'python main.py' would: as __main__, with this repository first on the module search path, so the server
# can still load configs such as pilottone.py.
#
# mrd-client looks for <config>.json in the current directory, then in this repository (see client.py).

import os
import runpy
import sys

repoDir = os.path.dirname(os.path.abspath(__file__))

def run_script(script):
    # Run a script from this repository as if invoked directly ('python <script> ...')
    sys.path.insert(0, repoDir)
    runpy.run_path(os.path.join(repoDir, script), run_name='__main__')

def client():
    run_script('client.py')

def server():
    run_script('main.py')

def ptone_offline_batch():
    run_script(os.path.join('ptone', 'ptone_offline_batch.py'))

def ptone_plot():
    run_script(os.path.join('ptone', 'ptone_plot.py'))

def ptone_bulk_motion_eval():
    run_script(os.path.join('ptone', 'bulk_motion_eval.py'))
