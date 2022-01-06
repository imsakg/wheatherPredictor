"""OpenWeather 2.5 metric current-weather integration; transport is injectable."""
import os
from datetime import datetime, timezone
from urllib.parse import urlencode
from .schema import Observation
from .units import finite


def parse_current(payload):
    try:
        if str(payload.get("cod",200)) != "200":
            raise ValueError("weather provider rejected the request")
        main, wind = payload["main"], payload.get("wind",{})
        direction = wind.get("deg")
        if direction is not None:
            direction = finite(direction)
            if not 0 <= direction <= 360:
                raise ValueError("invalid wind direction")
            direction %= 360
        return Observation(datetime.fromtimestamp(payload["dt"],timezone.utc),main["temp"],
                           humidity=main.get("humidity"),pressure=main.get("pressure"),
                           wind_speed=wind.get("speed"),wind_direction=direction,
                           precipitation=payload.get("rain",{}).get("1h"))
    except (KeyError,TypeError,OverflowError,OSError) as exc:
        raise ValueError("invalid current-weather payload") from exc


class OpenWeatherClient:
    def __init__(self, api_key=None, transport=None):
        self.api_key = os.environ.get("OPENWEATHER_API_KEY","") if api_key is None else api_key
        if not self.api_key or not self.api_key.strip():
            raise ValueError("set OPENWEATHER_API_KEY or supply an API key")
        from .http import JsonTransport
        self.transport = transport if transport is not None else JsonTransport()

    def _request(self, endpoint, latitude, longitude):
        lat, lon = finite(latitude), finite(longitude)
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError("latitude/longitude out of range")
        if self.transport is None:
            raise ValueError("a JSON transport is required")
        params = urlencode({"lat":lat,"lon":lon,"units":"metric","appid":self.api_key})
        return self.transport("https://api.openweathermap.org/data/2.5/"+endpoint+"?"+params)

    def current(self, latitude, longitude):
        return parse_current(self._request("weather",latitude,longitude))


    def forecast(self, latitude, longitude):
        """Provider's 3-hour forecasts, not locally trained predictions or observed truth."""
        payload = self._request("forecast",latitude,longitude)
        if not isinstance(payload,dict) or str(payload.get("cod")) != "200" or not isinstance(payload.get("list"),list):
            raise ValueError("invalid provider forecast payload")
        result = []
        for item in payload["list"]:
            copied = dict(item)
            if "rain" in copied:
                copied["rain"] = {"1h":copied["rain"].get("3h")}
            result.append(parse_current(copied))
        return sorted(result,key=lambda row:row.timestamp)
