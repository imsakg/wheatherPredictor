import unittest
from weather_predictor.models import LinearTrend

class TrendTests(unittest.TestCase):
    def test_ordinary_least_squares(self):
        self.assertEqual(LinearTrend().fit([1,3,5,7]).predict(2),[9,11])
        self.assertEqual(LinearTrend().fit([4,4,4]).predict(1),[4])
    def test_short_series(self):
        with self.assertRaises(ValueError): LinearTrend().fit([1])
