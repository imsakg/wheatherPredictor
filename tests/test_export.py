import io
import unittest
from weather_predictor.export import write_forecast

class ExportTests(unittest.TestCase):
    def test_prediction_columns_and_band_validation(self):
        stream=io.StringIO()
        write_forecast([{'timestamp':'2022-01-01T00:00:00Z','temperature':3}],stream,[(1,5)])
        self.assertIn('timestamp,temperature,lower,upper',stream.getvalue())
        self.assertIn(',3.0,1.0,5.0',stream.getvalue())
        with self.assertRaises(ValueError): write_forecast([{'timestamp':'x','temperature':3}],io.StringIO(),[])
