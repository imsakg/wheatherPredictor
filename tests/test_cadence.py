import unittest
from datetime import datetime, timezone, timedelta
from weather_predictor.schema import Observation
from weather_predictor.forecast import forecast_observations, regular_observations
from weather_predictor.models import LinearTrend

class CadenceTests(unittest.TestCase):
    def test_infers_three_hour_forecast_cadence(self):
        t=datetime(2022,1,1,tzinfo=timezone.utc)
        rows=[Observation(t+timedelta(hours=3*i),3*i) for i in range(3)]
        self.assertEqual(forecast_observations(rows,LinearTrend(),1)[0]['timestamp'],(t+timedelta(hours=9)).isoformat())
    def test_rejects_duplicate_and_irregular_times(self):
        t=datetime(2022,1,1,tzinfo=timezone.utc)
        for offsets in ([0,0,1],[0,1,3]):
            rows=[Observation(t+timedelta(hours=i),i) for i in offsets]
            with self.assertRaises(ValueError): regular_observations(rows)
