import io
import unittest
from datetime import datetime, timezone
from weather_predictor.schema import Observation
from weather_predictor.csvio import read_csv

class InputHardeningTests(unittest.TestCase):
    def test_temperature_is_not_optional(self):
        with self.assertRaises(ValueError): Observation(datetime.now(timezone.utc),None)
        with self.assertRaises(ValueError): Observation('yesterday',2)
    def test_duplicate_headers_and_short_rows(self):
        cases=['timestamp,temperature,temperature\n2022-01-01T00:00:00Z,2,99\n',
               'temperature,timestamp\n1\n','timestamp,temperature\n,1\n',
               'timestamp,temperature\n2022-01-01T00:00:00Z,1,2\n']
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError): read_csv(io.StringIO(text))
