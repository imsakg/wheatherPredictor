import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timezone
from weather_predictor.cli import main
from weather_predictor.schema import Observation

class CliCollectTests(unittest.TestCase):
    def test_collect_uses_environment_key_and_updates_cache(self):
        item=Observation(datetime(2022,1,1,tzinfo=timezone.utc),5)
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as folder:
            path=Path(folder)/'station.csv'; output=io.StringIO()
            with patch('weather_predictor.api.OpenWeatherClient') as client, redirect_stdout(output):
                client.return_value.current.return_value=item
                self.assertEqual(main(['collect','--lat','40','--lon','29','--output',str(path)]),0)
                client.assert_called_once_with()
                client.return_value.current.assert_called_once_with(40,29)
            self.assertTrue(path.exists())
            self.assertEqual(json.loads(output.getvalue())['count'],1)
