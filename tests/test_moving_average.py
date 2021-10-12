import unittest
from weather_predictor.models import MovingAverage

class MovingAverageTests(unittest.TestCase):
    def test_recursive_local_mean(self):
        self.assertEqual(MovingAverage(2).fit([0,2,4]).predict(3),[3,3.5,3.25])
    def test_training_window_enforced(self):
        with self.assertRaises(ValueError): MovingAverage(4).fit([1,2])
