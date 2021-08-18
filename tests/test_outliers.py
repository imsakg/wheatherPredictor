import unittest
from weather_predictor.quality import outlier_indices

class OutlierTests(unittest.TestCase):
    def test_flags_spike_without_replacing_it(self):
        values=[10,10,10,50,10,10,10]
        self.assertEqual(outlier_indices(values,window=2),[3])
        self.assertEqual(values[3],50)
    def test_smooth_series_and_invalid_window(self):
        self.assertEqual(outlier_indices(range(10)),[])
        with self.assertRaises(ValueError): outlier_indices([1],window=0)
