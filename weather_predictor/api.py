"""OpenWeather 2.5 metric integration. Provider forecasts remain separate from truth."""
import os
from datetime import datetime, timezone
from urllib.parse import urlencode
from .schema import Observation
from .units import finite


def _precipitation(payload):
    values = []
    for kind in ("rain","snow"):
        section = payload.get(kind,{})
        if not isinstance(section,dict):
            raise ValueError("invalid precipitation section")
        value = section.get("1h")
        if value is not None:
            value = finite(value)
            if value < 0:
                raise ValueError("precipitation cannot be negative")
            values.append(value)
    return sum(values) if values else None


def parse_current(payload):
    try:
        if not isinstance(payload,dict) or str(payload.get("cod",200)) != "200":
            raise ValueError("provider rejected request")
        main, wind = payload["main"], payload.get("wind",{})
        if not isinstance(main,dict) or not isinstance(wind,dict):
            raise ValueError("invalid measurement sections")
        direction = wind.get("deg")
        if direction is not None:
            direction = finite(direction)
            if not 0 <= direction <= 360:
                raise ValueError("invalid wind direction")
            direction %= 360
        return Observation(datetime.fromtimestamp(finite(payload["dt"]),timezone.utc),main["temp"],
                           humidity=main.get("humidity"),pressure=main.get("pressure"),
                           wind_speed=wind.get("speed"),wind_direction=direction,
                           precipitation=_precipitation(payload))
    except (KeyError,TypeError,ValueError,OverflowError,OSError):
        raise ValueError("invalid or rejected current-weather payload") from None


class OpenWeatherClient:
    def __init__(self, api_key=None, transport=None):
        self.api_key = os.environ.get("OPENWEATHER_API_KEY","") if api_key is None else api_key
        if not isinstance(self.api_key,str) or not self.api_key.strip():
            raise ValueError("set OPENWEATHER_API_KEY or supply an API key")
        from .http import JsonTransport
        self.transport = transport if transport is not None else JsonTransport()

    def _request(self, endpoint, latitude, longitude):
        lat, lon = finite(latitude), finite(longitude)
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError("latitude/longitude out of range")
        params = urlencode({"lat":lat,"lon":lon,"units":"metric","appid":self.api_key})
        return self.transport("https://api.openweathermap.org/data/2.5/"+endpoint+"?"+params)

    def current(self, latitude, longitude):
        return parse_current(self._request("weather",latitude,longitude))

    def forecast(self, latitude, longitude):
        """Three-hour forecasts; precipitation is the interval's liquid-equivalent mm."""
        payload = self._request("forecast",latitude,longitude)
        if not isinstance(payload,dict) or str(payload.get("cod")) != "200" or not isinstance(payload.get("list"),list) or not payload["list"]:
            raise ValueError("invalid provider forecast payload")
        result = []
        for item in payload["list"]:
            if not isinstance(item,dict):
                raise ValueError("invalid provider forecast entry")
            copied = dict(item)
            for kind in ("rain","snow"):
                if kind in copied:
                    section = copied[kind]
                    if not isinstance(section,dict):
                        raise ValueError("invalid precipitation section")
                    copied[kind] = {"1h":section.get("3h")}
            result.append(parse_current(copied))
        return sorted(result,key=lambda row:row.timestamp)
