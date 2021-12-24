import unittest
from weather_predictor.api import OpenWeatherClient, parse_current

class ApiTests(unittest.TestCase):
    def test_current_payload_and_metric_units(self):
        payload={'cod':200,'dt':1640995200,'main':{'temp':4,'humidity':100,'pressure':1010},'wind':{'speed':3,'deg':360},'rain':{'1h':.2}}
        calls=[]
        client=OpenWeatherClient('test-key',transport=lambda url:(calls.append(url) or payload))
        item=client.current(40,29)
        self.assertEqual(item.temperature,4)
        self.assertEqual(item.wind_direction,0)
        self.assertEqual(item.precipitation,.2)
        self.assertIn('units=metric',calls[0])
    def test_missing_key_coordinates_and_errors(self):
        with self.assertRaises(ValueError): OpenWeatherClient('')
        with self.assertRaises(ValueError): OpenWeatherClient('key',lambda _:{}).current(100,0)
        with self.assertRaises(ValueError): parse_current({'cod':401,'message':'private'})
