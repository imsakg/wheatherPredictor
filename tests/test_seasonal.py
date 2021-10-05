import unittest
from weather_predictor.models import SeasonalNaive

class SeasonalTests(unittest.TestCase):
    def test_repeats_last_complete_season(self):
        self.assertEqual(SeasonalNaive(3).fit([0,1,2,3,4]).predict(5),[2,3,4,2,3])
    def test_bad_period_and_short_training(self):
        with self.assertRaises(ValueError): SeasonalNaive(0)
        with self.assertRaises(ValueError): SeasonalNaive(3).fit([1,2])
