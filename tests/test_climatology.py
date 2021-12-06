import unittest
from datetime import datetime, timezone
from weather_predictor.climatology import MonthlyClimatology
from weather_predictor.schema import Observation

class ClimatologyTests(unittest.TestCase):
    def test_monthly_average_and_fallback(self):
        rows=[Observation(datetime(2021,m,1,tzinfo=timezone.utc),v) for m,v in [(1,2),(1,4),(2,8)]]
        model=MonthlyClimatology().fit(rows)
        self.assertEqual(model.predict([datetime(2022,1,1),datetime(2022,3,1)]),[3,14/3])
    def test_unfitted_and_empty(self):
        with self.assertRaises(ValueError): MonthlyClimatology().fit([])
        with self.assertRaises(ValueError): MonthlyClimatology().predict([])
