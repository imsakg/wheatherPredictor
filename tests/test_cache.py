import os
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone
from weather_predictor.collection import save_csv_cache
from weather_predictor.csvio import read_csv
from weather_predictor.schema import Observation

class CacheTests(unittest.TestCase):
    def test_incremental_cache_and_atomic_conflicts(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as folder:
            path=Path(folder)/'cache.csv'; stamp=datetime(2021,1,1,tzinfo=timezone.utc)
            a,b=Observation(stamp,1),Observation(stamp,2)
            self.assertEqual(save_csv_cache(path,[a]),1)
            with self.assertRaises(ValueError): save_csv_cache(path,[b])
            with path.open() as stream: self.assertEqual(read_csv(stream),[a])
            self.assertEqual(save_csv_cache(path,[b],conflict='last'),1)
            self.assertEqual(list(Path(folder).glob('*.tmp')),[])
