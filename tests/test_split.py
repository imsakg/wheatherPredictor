import unittest
from weather_predictor.timeseries import chronological_split

class SplitTests(unittest.TestCase):
    def test_ordered_nonoverlapping_partitions(self):
        self.assertEqual(chronological_split(list(range(10)),.6,.2),([0,1,2,3,4,5],[6,7],[8,9]))
    def test_reject_empty_and_bad_fractions(self):
        for args in (([],.7,.2),([1],.7,.2),([1,2,3],.8,.3)):
            with self.assertRaises(ValueError): chronological_split(*args)
