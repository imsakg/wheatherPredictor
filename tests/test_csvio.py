import io
import unittest
from datetime import datetime, timezone
from weather_predictor.csvio import read_csv, write_csv
from weather_predictor.schema import Observation

class CsvTests(unittest.TestCase):
    def test_round_trip_and_order(self):
        items = [Observation(datetime(2021,1,1,tzinfo=timezone.utc), 1.25, humidity=100)]
        stream = io.StringIO(); write_csv(items, stream); stream.seek(0)
        self.assertEqual(read_csv(stream), items)
    def test_missing_temperature_has_line_number(self):
        with self.assertRaisesRegex(ValueError, 'line 2'):
            read_csv(io.StringIO('timestamp,temperature\n2021-01-01T00:00:00Z,\n'))
    def test_sort_and_missing_columns(self):
        stream = io.StringIO('timestamp,temperature\n2021-01-02T00:00:00Z,3\n2021-01-01T00:00:00Z,2\n')
        self.assertEqual(read_csv(stream)[0].temperature, 2)
        with self.assertRaises(ValueError): read_csv(io.StringIO('x,y\n1,2\n'))
