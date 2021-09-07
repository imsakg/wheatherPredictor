import unittest
from datetime import datetime, timezone, timedelta
from weather_predictor.features import calendar_features, wind_components

class FeatureTests(unittest.TestCase):
    def test_daily_cycle(self):
        t=datetime(2021,1,1,tzinfo=timezone.utc)
        a,b=calendar_features(t),calendar_features(t+timedelta(hours=6))
        self.assertAlmostEqual(a['hour_cos'],1)
        self.assertAlmostEqual(b['hour_sin'],1)
        self.assertAlmostEqual(b['year_sin']**2+b['year_cos']**2,1)
    def test_meteorological_wind(self):
        u,v=wind_components(4,0)
        self.assertAlmostEqual(u,0)
        self.assertAlmostEqual(v,-4)
        with self.assertRaises(ValueError): wind_components(-1,90)
