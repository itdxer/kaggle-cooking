import os
import unittest

import six
import pandas as pd
import pandas.util.testing as pandas_testing

from src.utils import read_file_with_receipts


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
TEST_RECEIPT_FILE = os.path.join(CURRENT_DIR, 'files', 'fake_receipts.json')
EXPECTED_DATASET_FILE = os.path.join(CURRENT_DIR, 'files',
                                     'expected_receipts_dataset.csv')


class DataReadingTestCase(unittest.TestCase):
    def test_read_json_file(self):
        expected_dataset = pd.read_csv(EXPECTED_DATASET_FILE)
        actual_dataset = read_file_with_receipts(TEST_RECEIPT_FILE)
        pandas_testing.assert_frame_equal(expected_dataset, actual_dataset)

    def test_ingredients_column_data_type(self):
        dataset = read_file_with_receipts(TEST_RECEIPT_FILE)
        ingredients = dataset.ingredients[0]
        self.assertIsInstance(ingredients, six.string_types)
