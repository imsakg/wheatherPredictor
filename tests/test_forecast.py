import unittest
from datetime import datetime, timedelta, timezone
from weather_predictor.schema import Observation
from weather_predictor.forecast import forecast_observations, make_model

class ForecastTests(unittest.TestCase):
    def test_timestamps_and_model_selection(self):
        t=datetime(2021,1,1,tzinfo=timezone.utc)
        rows=[Observation(t+timedelta(hours=i),i) for i in range(10)]
        result=forecast_observations(rows,make_model('trend'),2)
        self.assertEqual([r['temperature'] for r in result],[10,11])
        self.assertEqual(result[0]['timestamp'],(t+timedelta(hours=10)).isoformat())
    def test_unknown_model(self):
        with self.assertRaises(ValueError): make_model('magic')
