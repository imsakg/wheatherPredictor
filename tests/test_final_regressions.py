import io
import json
import unittest
from datetime import date, datetime, timezone, timedelta
from weather_predictor.legacy import read_daily
from weather_predictor.weather import heat_index
from weather_predictor.model_io import load_model
from weather_predictor.forecast import forecast_observations
from weather_predictor.schema import Observation
from weather_predictor.climatology import MonthlyClimatology

class FinalRegressionTests(unittest.TestCase):
    def test_dst_ambiguous_and_nonexistent_local_clocks(self):
        for day,clock in ((date(2021,11,7),'1:30 AM'),(date(2021,3,14),'2:30 AM')):
            text='Time,Temperature,Humidity,Wind,Wind Speed,Pressure\n'+clock+',32 F,80 %,N,2 mph,30 in\n'
            with self.assertRaisesRegex(ValueError,'local clock'):
                read_daily(io.StringIO(text),day,'America/New_York')
    def test_noaa_heat_index_screen_and_adjustments(self):
        for temp,rh,expected in ((20,50,19.680555555555557),(27.77777777777778,90,33.32870991111106),(32.22222222222222,10,29.59942686788192)):
            self.assertAlmostEqual(heat_index(temp,rh),expected,places=7)
    def test_model_version_is_not_boolean(self):
        text=json.dumps({'version':True,'model':'Persistence','parameters':{},'training':[1]})
        with self.assertRaises(ValueError): load_model(io.StringIO(text))
    def test_model_forecast_shape_and_finiteness(self):
        class Broken:
            def fit(self,values): return self
            def predict(self,horizon): return self.output
        t=datetime(2022,1,1,tzinfo=timezone.utc)
        rows=[Observation(t,1),Observation(t+timedelta(hours=1),2)]
        for output in ([1],[1,float('nan')]):
            broken=Broken(); broken.output=output
            with self.assertRaises(ValueError): forecast_observations(rows,broken,2)
    def test_climatology_normalizes_query_timezone(self):
        rows=[Observation(datetime(2021,m,1,tzinfo=timezone.utc),v) for m,v in ((1,1),(2,2))]
        model=MonthlyClimatology().fit(rows)
        local=datetime(2022,1,31,23,tzinfo=timezone(timedelta(hours=-3)))
        self.assertEqual(model.predict([local]),[2])
        with self.assertRaises(ValueError): model.predict([datetime(2022,1,1)])
