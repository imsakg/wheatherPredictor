import unittest
from weather_predictor.timeseries import StandardScaler

class ScalerTests(unittest.TestCase):
    def test_training_statistics_only_and_inverse(self):
        model=StandardScaler().fit([1,2,3])
        transformed=model.transform([100])
        self.assertGreater(transformed[0],100)
        self.assertEqual(model.inverse_transform(transformed),[100])
    def test_constant_series_and_unfitted(self):
        self.assertEqual(StandardScaler().fit([2,2]).transform([2]),[0])
        with self.assertRaises(ValueError): StandardScaler().transform([1])
        with self.assertRaises(ValueError): StandardScaler().fit([])
