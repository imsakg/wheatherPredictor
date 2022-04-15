import io
import os
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stderr
from weather_predictor.cli import main

class OutputSafetyTests(unittest.TestCase):
    def test_input_and_aliases_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as folder:
            source=Path(folder)/'station.csv'
            text='timestamp,temperature\n2022-01-01T00:00:00Z,1\n2022-01-01T01:00:00Z,2\n'
            source.write_text(text)
            alias=Path(folder)/'alias.csv'; alias.symlink_to(source)
            hard=Path(folder)/'hard.csv'; os.link(source,hard)
            for command in ('forecast','report'):
                for output in (source,alias,hard):
                    with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as caught:
                        main([command,str(source),'--output',str(output)])
                    self.assertEqual(caught.exception.code,2)
                    self.assertEqual(source.read_text(),text)
