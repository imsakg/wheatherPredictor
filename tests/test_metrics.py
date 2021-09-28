import unittest
from weather_predictor.metrics import score

class MetricTests(unittest.TestCase):
    def test_errors_and_zero_safe_smape(self):
        result=score([0,2],[0,4])
        self.assertEqual(result['mae'],1)
        self.assertAlmostEqual(result['rmse'],2**.5)
        self.assertEqual(result['bias'],1)
        self.assertAlmostEqual(result['smape'],100/3)
    def test_invalid_pairs(self):
        for actual,predicted in (([],[]),([1],[]),([float('nan')],[1])):
            with self.assertRaises(ValueError): score(actual,predicted)
