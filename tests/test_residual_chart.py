import unittest
import xml.etree.ElementTree as ET
from weather_predictor.visualization import residual_chart

class ResidualChartTests(unittest.TestCase):
    def test_residual_zero_reference(self):
        text=residual_chart([1,3,2],[1,2,4])
        # Only locally generated SVG is parsed; no external XML is accepted.
        ET.fromstring(text)
        self.assertIn('error',text)
        self.assertIn('zero',text)
    def test_pair_length_validation(self):
        with self.assertRaises(ValueError): residual_chart([1],[])
