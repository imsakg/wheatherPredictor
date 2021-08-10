import unittest
from datetime import datetime, timezone, timedelta
from weather_predictor.schema import Observation
from weather_predictor.quality import quality_report

class QualityTests(unittest.TestCase):
    def test_gaps_duplicates_and_missing_fields(self):
        t = datetime(2021,1,1,tzinfo=timezone.utc)
        a = Observation(t,10); b = Observation(t+timedelta(hours=3),12,humidity=80)
        report = quality_report([a,b,a],timedelta(hours=1))
        self.assertEqual(report['duplicates'],1)
        self.assertEqual(report['missing_steps'],2)
        self.assertEqual(report['missing']['humidity'],2)
    def test_empty_and_invalid_frequency(self):
        self.assertEqual(quality_report([])['count'],0)
        with self.assertRaises(ValueError): quality_report([],timedelta(0))
