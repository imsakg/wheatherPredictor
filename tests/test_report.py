import unittest
from datetime import datetime, timezone, timedelta
from weather_predictor.report import forecast_report
from weather_predictor.schema import Observation

class ReportTests(unittest.TestCase):
    def test_self_contained_report(self):
        rows=[Observation(datetime(2022,1,1,tzinfo=timezone.utc)+timedelta(hours=i),i) for i in range(3)]
        forecast=[{'timestamp':'2022-01-01T03:00:00+00:00','temperature':3}]
        html=forecast_report(rows,forecast,title='<script>oops</script>')
        self.assertIn('<svg',html)
        self.assertIn('2022-01-01T03:00:00+00:00',html)
        self.assertNotIn('<script>',html)
        self.assertIn('Not a safety-critical forecast',html)
