import unittest
from weather_predictor.models import Persistence

class PersistenceTests(unittest.TestCase):
    def test_repeats_last_observation(self):
        self.assertEqual(Persistence().fit([2,4]).predict(3),[4,4,4])
    def test_fit_and_horizon_guards(self):
        with self.assertRaises(ValueError): Persistence().predict(1)
        with self.assertRaises(ValueError): Persistence().fit([])
        for h in (0,-1,1.5):
            with self.assertRaises(ValueError): Persistence().fit([2]).predict(h)
