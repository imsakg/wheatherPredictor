import unittest
from datetime import datetime, timezone, timedelta
from weather_predictor.schema import Observation
from weather_predictor.preprocessing import hourly_mean

class ResampleTests(unittest.TestCase):
    def test_hourly_means_circular_wind_and_precipitation(self):
        t = datetime(2021,1,1,tzinfo=timezone.utc)
        rows=[Observation(t,10,wind_direction=350,precipitation=1),Observation(t+timedelta(minutes=30),14,wind_direction=10,precipitation=2)]
        result=hourly_mean(rows)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0].temperature,12)
        self.assertAlmostEqual(result[0].wind_direction,0)
        self.assertEqual(result[0].precipitation,3)
    def test_no_empty_hour_imputation(self):
        self.assertEqual(hourly_mean([]),[])
