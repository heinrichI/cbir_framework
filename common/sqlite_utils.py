import io
import numpy as np
import sqlite3

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def adapt_npint(npint):
    pythonint = int(npint)
    return pythonint


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

