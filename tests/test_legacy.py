import io
import unittest
from datetime import date
from weather_predictor.legacy import read_daily, compass

class LegacyTests(unittest.TestCase):
    def test_midnight_noon_and_full_values(self):
        data = 'Time,Temperature,Dew Point,Humidity,Wind,Wind Speed,Pressure,Precip.\n12:20 AM,-4.5 °F,0,100 °%,CALM,0 °mph,29.10 °in,0.1 °in\n12:20 PM,100 °F,0,9 °%,N,12 °mph,30.10 °in,0 °in\n'
        rows = read_daily(io.StringIO(data), date(2021,1,1), 'Europe/Istanbul')
        self.assertEqual(rows[0].timestamp.hour,21)
        self.assertEqual(rows[1].timestamp.hour,9)
        self.assertEqual(rows[0].humidity,100)
        self.assertAlmostEqual(rows[1].temperature,37.7777778)
        self.assertIsNone(rows[0].wind_direction)
        self.assertAlmostEqual(rows[1].wind_speed,5.36448)
    def test_compass_points(self):
        self.assertEqual(compass('WSW'),247.5)
        self.assertIsNone(compass('VAR'))
        with self.assertRaises(ValueError): compass('nonsense')
