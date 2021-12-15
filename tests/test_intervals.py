import unittest
from weather_predictor.intervals import residual_intervals

class IntervalTests(unittest.TestCase):
    def test_empirical_residual_bands(self):
        self.assertEqual(residual_intervals([10,11],[-1,0,2,-3],.8),[(7,13),(8,14)])
    def test_invalid_calibration(self):
        with self.assertRaises(ValueError): residual_intervals([1],[])
        with self.assertRaises(ValueError): residual_intervals([1],[1],1)
