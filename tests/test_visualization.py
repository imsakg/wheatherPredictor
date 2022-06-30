import unittest
import xml.etree.ElementTree as ET
from weather_predictor.visualization import line_chart

class VisualizationTests(unittest.TestCase):
    def test_valid_svg_and_escaped_labels(self):
        text=line_chart({'A < B':[1,2,None,4],'constant':[3]*4},title='<script>alert(1)</script>')
        # Generated local SVG only; external XML is never loaded.
        root=ET.fromstring(text)
        self.assertTrue(root.tag.endswith('svg'))
        self.assertNotIn('<script>',text)
        self.assertIn('&lt;script&gt;',text)
    def test_constant_single_and_invalid_data(self):
        ET.fromstring(line_chart({'one':[1]}))
        for data in ({},{'x':[]},{'x':[float('nan')]}):
            with self.assertRaises(ValueError): line_chart(data)
