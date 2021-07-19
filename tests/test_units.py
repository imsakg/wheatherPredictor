import unittest
from weather_predictor import units

class UnitTests(unittest.TestCase):
    def test_signed_decimal_temperatures(self):
        self.assertAlmostEqual(units.fahrenheit_to_celsius(-4.5), -20.27777778)
        self.assertEqual(units.celsius_to_fahrenheit(0), 32)
    def test_wind_pressure_and_rain(self):
        self.assertAlmostEqual(units.mph_to_ms(10), 4.4704)
        self.assertAlmostEqual(units.inhg_to_hpa(30), 1015.9166, places=3)
        self.assertEqual(units.inches_to_mm(1), 25.4)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): units.mph_to_ms(float('nan'))
