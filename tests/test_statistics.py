import unittest
from weather_predictor.statistics import autocorrelation, rolling_statistics

class StatisticsTests(unittest.TestCase):
    def test_autocorrelation_and_constant_convention(self):
        self.assertEqual(autocorrelation([1,1,1],2),[1,0,0])
        self.assertAlmostEqual(autocorrelation([1,2,3],1)[1],0)
        with self.assertRaises(ValueError): autocorrelation([1],1)
    def test_trailing_statistics(self):
        rows=rolling_statistics([1,3,5,7],2)
        self.assertEqual([row['mean'] for row in rows],[2,4,6])
        self.assertEqual([row['index'] for row in rows],[1,2,3])
        self.assertEqual(rows[0]['std'],1)
