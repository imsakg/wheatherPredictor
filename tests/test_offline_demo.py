import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

class OfflineDemoTests(unittest.TestCase):
    def test_real_station_pipeline(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as folder:
            result=subprocess.run([sys.executable,'-m','examples.offline_demo','--output',folder],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            manifest=json.loads(result.stdout)
            self.assertEqual(manifest['observations'],24)
            self.assertGreater(manifest['folds'],0)
            for name in ('normalized.csv','forecast.csv','forecast.svg','residuals.svg','report.html','backtest.json'):
                self.assertGreater((Path(folder)/name).stat().st_size,0)
