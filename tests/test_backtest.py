import unittest
from weather_predictor.backtest import rolling_backtest
from weather_predictor.models import Persistence, LinearTrend

class BacktestTests(unittest.TestCase):
    def test_expanding_origins_and_horizon_scores(self):
        report=rolling_backtest(range(10),Persistence,min_train=4,horizon=2,step=2)
        self.assertEqual(report['folds'],3)
        self.assertEqual(report['metrics']['mae'],1.5)
        self.assertEqual(report['by_horizon'][2]['mae'],2)
        self.assertEqual(report['records'][0]['origin'],4)
    def test_training_is_strictly_past(self):
        report=rolling_backtest([0,0,0,0,100,100],Persistence,4,1)
        self.assertEqual(report['records'][0]['predicted'],0)
        with self.assertRaises(ValueError): rolling_backtest([1],Persistence,4,1)
