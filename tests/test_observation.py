import unittest
from datetime import datetime, timezone, timedelta
from weather_predictor.schema import Observation

class ObservationTests(unittest.TestCase):
    def test_utc_normalization_and_optional_measurements(self):
        item = Observation(datetime(2021, 1, 1, 3, tzinfo=timezone(timedelta(hours=3))), 8)
        self.assertEqual(item.timestamp.hour, 0)
        self.assertIsNone(item.humidity)
    def test_invalid_fields(self):
        for kwargs in ({'humidity':101}, {'wind_speed':-1}, {'pressure':0}, {'wind_direction':360}, {'precipitation':-1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Observation(datetime.now(timezone.utc), 10, **kwargs)
    def test_naive_and_infinite_rejected(self):
        with self.assertRaises(ValueError): Observation(datetime(2021,1,1), 0)
        with self.assertRaises(ValueError): Observation(datetime.now(timezone.utc), float('inf'))
