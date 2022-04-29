import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from weather_predictor.cli import main

class LegacyEntrypointTests(unittest.TestCase):
    def test_original_scripts_have_no_import_side_effects(self):
        result=subprocess.run([sys.executable,'-c','import predictor, parser, scraper'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout,'')
    def test_import_original_daily_file(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as folder,redirect_stdout(io.StringIO()):
            target=Path(folder)/'normalized.csv'
            self.assertEqual(main(['import-daily','datas/2021-03-25.csv','--timezone','Europe/Istanbul','--hourly','--output',str(target)]),0)
            self.assertTrue(target.read_text().startswith('timestamp,temperature'))
