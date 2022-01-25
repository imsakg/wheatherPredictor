import json
import subprocess
import sys
import unittest

class CliTests(unittest.TestCase):
    def test_help_and_inspect_error(self):
        result=subprocess.run([sys.executable,'-m','weather_predictor','--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('inspect',result.stdout)
        result=subprocess.run([sys.executable,'-m','weather_predictor','inspect','missing.csv'],capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertNotIn('Traceback',result.stderr)
