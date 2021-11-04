import unittest
from weather_predictor.models import HoltTrend

class HoltTests(unittest.TestCase):
    def test_linear_series_and_damping(self):
        self.assertEqual(HoltTrend(1,1,1).fit([1,2,3]).predict(2),[4,5])
        self.assertEqual(HoltTrend(1,1,.5).fit([1,2,3]).predict(2),[3.5,3.75])
    def test_invalid_parameters_and_short_data(self):
        with self.assertRaises(ValueError): HoltTrend(phi=0)
        with self.assertRaises(ValueError): HoltTrend().fit([1])
