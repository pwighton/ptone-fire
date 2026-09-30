#!/usr/bin/python3
# Convert a raw MRD stream (e.g. bytes captured from a FIRE socket) to an MRD HDF5 file
# Usage: python stream2h5.py <input_stream_file> <output.h5>

import sys
import socket
import logging
from connection import Connection

class FileSocket:
    """Minimal socket stand-in so Connection can read from a file"""
    def __init__(self, f):
        self.f = f

    def recv(self, n, flags=0):
        if flags & socket.MSG_PEEK:
            pos  = self.f.tell()
            data = self.f.read(n)
            self.f.seek(pos)
            return data
        return self.f.read(n)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python stream2h5.py <input_stream_file> <output.h5>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format='%(message)s')

    with open(sys.argv[1], 'rb') as f:
        conn = Connection(FileSocket(f), savedata=True, savedataFile=sys.argv[2])
        for _ in conn:
            pass

        # Capture may end without an MRD_MESSAGE_CLOSE, which is what normally closes the file
        if conn.dset is not None and conn.dset._file.id.valid:
            conn.dset.close()
