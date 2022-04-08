import io
import json
import unittest
from weather_predictor.models import AutoRegression
from weather_predictor.model_io import load_model

class ModelLimitTests(unittest.TestCase):
    def test_bound_ar_workload_before_fitting(self):
        with self.assertRaises(ValueError): AutoRegression(700)
        text=json.dumps({'version':1,'model':'AutoRegression','parameters':{'lags':700,'ridge':.01},'training':list(range(1500))})
        with self.assertRaises(ValueError): load_model(io.StringIO(text))
    def test_document_size_and_type(self):
        with self.assertRaises(ValueError): load_model(io.StringIO(' '*2000001))
        with self.assertRaises(ValueError): load_model(io.StringIO('[]'))
