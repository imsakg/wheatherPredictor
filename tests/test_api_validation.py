import unittest
from weather_predictor.api import parse_current, OpenWeatherClient

class ApiValidationTests(unittest.TestCase):
    def test_malformed_payloads_are_contextual(self):
        for payload in ([],None,{'main':[]},{'dt':1,'main':{'temp':None}}, {'dt':1,'main':{'temp':2},'wind':[]}):
            with self.subTest(payload=payload),self.assertRaises(ValueError): parse_current(payload)
    def test_rain_and_snow_liquid_equivalents(self):
        body={'dt':1,'main':{'temp':2},'rain':{'1h':1},'snow':{'1h':2}}
        self.assertEqual(parse_current(body).precipitation,3)
        body={'cod':'200','list':[{'dt':1,'main':{'temp':2},'snow':{'3h':2}}]}
        self.assertEqual(OpenWeatherClient('key',lambda _:body).forecast(0,0)[0].precipitation,2)
