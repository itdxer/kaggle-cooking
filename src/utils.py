import os
import time
import logging
from contextlib import contextmanager

import pandas as pd


logging.basicConfig(level=logging.INFO)

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'data'))


@contextmanager
def logtime(description):
    logging.info("Starting '{}'".format(description))
    start_time = time.time()
    yield
    end_time = time.time()
    time_delay = end_time - start_time
    logging.info("Finihed '{}' (it took {:.2f} seconds)"
                 "".format(description, time_delay))


def read_file_with_receipts(filepath):
    data = pd.read_json(filepath)
    data.ingredients = data.ingredients.apply(' '.join)
    return data
