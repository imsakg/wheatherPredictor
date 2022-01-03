import unittest
from weather_predictor.api import OpenWeatherClient

class ApiForecastTests(unittest.TestCase):
    def test_provider_forecasts_stay_separate_from_observations(self):
        body={'cod':'200','list':[{'dt':1640995200,'main':{'temp':4},'rain':{'3h':3}}]}
        items=OpenWeatherClient('key',lambda _:body).forecast(40,29)
        self.assertEqual(items[0].precipitation,3)
        self.assertEqual(items[0].temperature,4)
        self.assertNotIn('1h',body['list'][0]['rain'])
    def test_bad_status_and_missing_list(self):
        for body in ({'cod':401},{'cod':200}):
            with self.assertRaises(ValueError): OpenWeatherClient('key',lambda _:body).forecast(0,0)
