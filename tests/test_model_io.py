import io
import unittest
from weather_predictor.models import AutoRegression, DifferencedAR, HoltTrend
from weather_predictor.model_io import dump_model, load_model

class ModelIoTests(unittest.TestCase):
    def test_json_round_trip(self):
        for model in (AutoRegression(2),DifferencedAR(2),HoltTrend()):
            model.fit(range(20)); stream=io.StringIO(); dump_model(model,stream); stream.seek(0)
            self.assertEqual(load_model(stream).predict(3),model.predict(3))
    def test_no_arbitrary_type_loading(self):
        for text in ('{}','{"version":1,"model":"os.system","parameters":{},"training":[1]}'):
            with self.assertRaises(ValueError): load_model(io.StringIO(text))
