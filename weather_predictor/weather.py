"""Derived weather indicators and local threshold diagnostics, not official warnings."""
import math
from .units import finite, celsius_to_fahrenheit, fahrenheit_to_celsius


def dew_point(temperature, humidity):
    temperature, humidity = finite(temperature), finite(humidity)
    if not 0 < humidity <= 100 or temperature <= -243.12:
        raise ValueError("Magnus dew point needs humidity in (0,100] and temperature > -243.12 C")
    gamma = math.log(humidity/100)+17.62*temperature/(243.12+temperature)
    return 243.12*gamma/(17.62-gamma)


def wind_chill(temperature, wind_speed):
    temperature, wind_speed = finite(temperature), finite(wind_speed)
    if wind_speed < 0:
        raise ValueError("wind speed must be nonnegative m/s")
    speed_kmh = wind_speed*3.6
    if temperature > 10 or speed_kmh <= 4.8:
        return temperature
    return 13.12+0.6215*temperature-11.37*speed_kmh**0.16+0.3965*temperature*speed_kmh**0.16


def heat_index(temperature, humidity):
    temperature, humidity = finite(temperature), finite(humidity)
    if not 0 <= humidity <= 100:
        raise ValueError("humidity must be in [0,100]")
    t, r = celsius_to_fahrenheit(temperature), humidity
    index = (0.5*(t+61+(t-68)*1.2+r*0.094)+t)/2
    if index >= 80:
        index = (-42.379+2.04901523*t+10.14333127*r-0.22475541*t*r-0.00683783*t*t
                 -0.05481717*r*r+0.00122874*t*t*r+0.00085282*t*r*r-0.00000199*t*t*r*r)
        if r < 13 and 80 <= t <= 112:
            index -= (13-r)/4*math.sqrt((17-abs(t-95))/17)
        elif r > 85 and 80 <= t <= 87:
            index += (r-85)/10*(87-t)/5
    return fahrenheit_to_celsius(index)


def threshold_alerts(observations, cold=0, heat=35, wind=15):
    cold, heat, wind = map(finite,(cold,heat,wind))
    if cold >= heat or wind < 0:
        raise ValueError("cold must be below heat and wind threshold nonnegative")
    output = []
    for row in observations:
        conditions = (("cold",row.temperature,cold,row.temperature<=cold),
                      ("heat",row.temperature,heat,row.temperature>=heat),
                      ("wind",row.wind_speed,wind,row.wind_speed is not None and row.wind_speed>=wind))
        for hazard,value,threshold,triggered in conditions:
            if triggered:
                output.append({"timestamp":row.timestamp.isoformat(),"hazard":hazard,"value":value,"threshold":threshold})
    return output
