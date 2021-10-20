import unittest
from weather_predictor.models import ExponentialSmoothing

class SmoothingTests(unittest.TestCase):
    def test_level_update(self):
        self.assertEqual(ExponentialSmoothing(.5).fit([2,4,6]).predict(2),[4.5,4.5])
    def test_alpha_bounds(self):
        for alpha in (0,1.1,float('nan')):
            with self.assertRaises(ValueError): ExponentialSmoothing(alpha)
