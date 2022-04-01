import unittest
from datetime import datetime, timezone, timedelta
from weather_predictor.schema import Observation
from weather_predictor.preprocessing import daily_summary

class DailyTests(unittest.TestCase):
    def test_local_day_and_partial_rain(self):
        t=datetime(2022,1,1,22,tzinfo=timezone.utc)
        rows=[Observation(t,2,precipitation=1),Observation(t+timedelta(hours=1),4)]
        result=daily_summary(rows,'Europe/Istanbul')
        self.assertEqual(result[0]['date'],'2022-01-02')
        self.assertEqual(result[0]['temperature_mean'],3)
        self.assertEqual(result[0]['precipitation_samples'],1)
        self.assertEqual(result[0]['precipitation_total'],1)
    def test_duplicates_do_not_double_count(self):
        row=Observation(datetime(2022,1,1,tzinfo=timezone.utc),1)
        self.assertEqual(daily_summary([row,row])[0]['count'],1)
