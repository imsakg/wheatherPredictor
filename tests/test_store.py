import os
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from weather_predictor.store import ObservationStore
from weather_predictor.schema import Observation

class StoreTests(unittest.TestCase):
    def test_upsert_query_and_restart(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as folder:
            path=Path(folder)/'station.db'; t=datetime(2021,1,1,tzinfo=timezone.utc)
            a,b=Observation(t,1),Observation(t+timedelta(hours=1),2)
            with ObservationStore(path) as store:
                store.put([a,b],source='station')
                store.put([Observation(t,3)],source='corrected')
                self.assertEqual(store.query(end=t),[Observation(t,3)])
            with ObservationStore(path) as store:
                self.assertEqual(store.query(start=b.timestamp),[b])
