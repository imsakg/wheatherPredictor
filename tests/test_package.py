import unittest


class PackageTests(unittest.TestCase):
    def test_version(self):
        import weather_predictor
        self.assertEqual(weather_predictor.__version__, "0.2.0")
