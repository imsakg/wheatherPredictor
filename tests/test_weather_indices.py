import unittest
from datetime import datetime, timezone
from weather_predictor.weather import dew_point, wind_chill, heat_index, threshold_alerts
from weather_predictor.schema import Observation

class WeatherIndexTests(unittest.TestCase):
    def test_derived_measurements(self):
        self.assertAlmostEqual(dew_point(20,100),20)
        self.assertAlmostEqual(dew_point(20,50),9.26,places=1)
        self.assertLess(wind_chill(0,10),0)
        self.assertEqual(wind_chill(20,5),20)
        self.assertGreater(heat_index(35,70),35)
        with self.assertRaises(ValueError): dew_point(20,0)
    def test_threshold_alerts(self):
        row=Observation(datetime(2022,1,1,tzinfo=timezone.utc),-2,wind_speed=20)
        self.assertEqual([v['hazard'] for v in threshold_alerts([row])],['cold','wind'])
