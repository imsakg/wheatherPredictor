import unittest
from weather_predictor.preprocessing import interpolate_short_gaps

class InterpolationTests(unittest.TestCase):
    def test_short_internal_gaps_only(self):
        self.assertEqual(interpolate_short_gaps([None,2,None,6,None,None,9,None],1),[None,2,4,6,None,None,9,None])
    def test_invalid_limit(self):
        with self.assertRaises(ValueError): interpolate_short_gaps([1],-1)
