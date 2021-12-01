import unittest
from weather_predictor.models import AutoRegression

class AutoregressionTests(unittest.TestCase):
    def test_recovers_an_ar_process(self):
        values=[1.0]
        for _ in range(30): values.append(2+.7*values[-1])
        forecast=AutoRegression(1,ridge=1e-8).fit(values).predict(3)
        expected=values[-1]
        for prediction in forecast:
            expected=2+.7*expected
            self.assertAlmostEqual(prediction,expected,places=5)
    def test_constant_series_and_bad_settings(self):
        self.assertEqual(AutoRegression(2).fit([4]*10).predict(2),[4,4])
        with self.assertRaises(ValueError): AutoRegression(0)
        with self.assertRaises(ValueError): AutoRegression(2,ridge=-1)
        with self.assertRaises(ValueError): AutoRegression(3).fit([1,2,3])
