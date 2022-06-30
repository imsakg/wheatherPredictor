import unittest
import xml.etree.ElementTree as ET
from weather_predictor.visualization import forecast_chart

class ForecastChartTests(unittest.TestCase):
    def test_history_prediction_and_bands(self):
        text=forecast_chart([1,2,3],[4,5],[(3,5),(4,6)])
        # Generated local SVG only; external XML is never loaded.
        ET.fromstring(text)
        self.assertIn('lower',text)
        self.assertIn('forecast',text)
    def test_band_alignment(self):
        with self.assertRaises(ValueError): forecast_chart([1],[2],[])
