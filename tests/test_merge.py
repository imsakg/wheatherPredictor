import unittest
from datetime import datetime, timezone, timedelta
from weather_predictor.schema import Observation
from weather_predictor.collection import merge_observations

class MergeTests(unittest.TestCase):
    def test_sorted_deduplication(self):
        a = Observation(datetime(2021,1,1,tzinfo=timezone.utc), 2)
        b = Observation(a.timestamp + timedelta(hours=1),3)
        self.assertEqual(merge_observations([b,a],[a]),[a,b])
    def test_conflicts_are_explicit(self):
        a = Observation(datetime(2021,1,1,tzinfo=timezone.utc),2)
        b = Observation(a.timestamp,4)
        with self.assertRaises(ValueError): merge_observations([a],[b])
        self.assertEqual(merge_observations([a],[b],conflict='last'),[b])
        with self.assertRaises(ValueError): merge_observations([],conflict='magic')
