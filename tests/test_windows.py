import unittest
from weather_predictor.timeseries import lag_windows

class WindowTests(unittest.TestCase):
    def test_multi_step_alignment(self):
        x,y=lag_windows([0,1,2,3,4,5],3,2)
        self.assertEqual(x,[[0,1,2],[1,2,3]])
        self.assertEqual(y,[[3,4],[4,5]])
    def test_insufficient_and_invalid_width(self):
        with self.assertRaises(ValueError): lag_windows([1],2)
        with self.assertRaises(ValueError): lag_windows([1,2],0)
