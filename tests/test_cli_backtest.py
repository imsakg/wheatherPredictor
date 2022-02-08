import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from weather_predictor.cli import main

class CliBacktestTests(unittest.TestCase):
    def test_json_backtest(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as folder:
            path=Path(folder)/'weather.csv'
            path.write_text('timestamp,temperature\n'+''.join('2021-01-01T{:02}:00:00Z,{}\n'.format(i,i) for i in range(12)))
            output=io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(['backtest',str(path),'--min-train','4','--horizon','2','--step','2']),0)
            self.assertEqual(json.loads(output.getvalue())['folds'],4)
