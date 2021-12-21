import unittest
from weather_predictor.models import WeightedEnsemble, Persistence, LinearTrend

class EnsembleTests(unittest.TestCase):
    def test_weighted_forecasts(self):
        model=WeightedEnsemble([Persistence,LinearTrend],[1,3]).fit([1,2,3])
        self.assertEqual(model.predict(2),[3.75,4.5])
    def test_bad_members_and_weights(self):
        for members,weights in (([],[]),([Persistence],[0]),([Persistence],[-1]),([Persistence],[1,1])):
            with self.assertRaises(ValueError): WeightedEnsemble(members,weights)
