import unittest
from weather_predictor.models import DifferencedAR

class DifferencedTests(unittest.TestCase):
    def test_integrates_predicted_changes(self):
        output=DifferencedAR(2).fit(range(20)).predict(3)
        self.assertEqual(output,[20,21,22])
    def test_minimum_history(self):
        with self.assertRaises(ValueError): DifferencedAR(2).fit([1,2,3,4])
