# OpenWeather integration

The maintained client uses the documented OpenWeather 2.5 endpoints:

- HTTPS `/data/2.5/weather` for current conditions.
- HTTPS `/data/2.5/forecast` for the provider's three-hour forecast list.

Coordinates and `units=metric` are explicit. `dt` is interpreted as UTC Unix seconds. The parser reads `main.temp`, optional humidity/pressure and wind speed/direction. Direction 360 is normalized to north (0). Rain and snow liquid-equivalent amounts are combined when present; absent amounts remain unknown. Current and forecast precipitation intervals differ (one hour versus three hours); do not compare or sum them blindly.

References: [current weather API](https://openweathermap.org/api/current), [three-hour forecast API](https://openweathermap.org/api/forecast5). Consult provider documentation/terms for current quotas, account entitlements, retention and attribution. This project does not assert that a plan is free or that a key has been validated.

## Credentials and transport

`OpenWeatherClient()` reads `OPENWEATHER_API_KEY`. A Python caller may supply `api_key` explicitly, but should obtain it from secure runtime configuration. Do not commit secrets or enable URL/request logging: the provider requires the key in a query parameter. Public transport errors deliberately omit URLs, provider error bodies and keys. A custom injected transport is trusted code and must preserve these controls itself.

`JsonTransport` defaults to a 10-second timeout, three attempts and a 2 MB response limit. It retries transient connection failures plus HTTP 429/500/502/503/504 with bounded exponential backoff; other HTTP failures do not retry. It does not currently honor `Retry-After` or persist quota accounting. Redirects must stay on the original HTTPS origin. TLS validation remains enabled. Non-finite JSON values, oversized responses and malformed JSON are rejected.

## Offline contract tests

```python
from weather_predictor.api import OpenWeatherClient

# Explicitly a test double, not an observed API response.
payload = {"cod": 200, "dt": 1640995200, "main": {"temp": 5.0}}
client = OpenWeatherClient("test-only", transport=lambda url: payload)
assert client.current(40, 29).temperature == 5.0
```

Tests also inject openers that return local byte streams or fail with controlled HTTP/network errors. No credentialed live API call is required or claimed by the default suite. The separately exercised offline station demo uses actual tracked measurements, not fabricated provider output.

## Operational scope

`collect` merges one current sample into an explicitly selected CSV and prints count/latest/path metadata. Updates at an identical timestamp use last-write-wins deliberately. Use one cache/database per coordinate/station; station identity is not encoded in canonical CSV. Schedule repeated calls externally if desired, respecting provider limits. Local cache writes are atomic but require a single writer; SQLite operations are transactional. Forecasts are not observations and must never be inserted as observed training truth.

Missing intervals, station outages, calibration changes and precipitation accumulation semantics remain your responsibility. A weather-provider forecast is not the result of a local statistical model even though both can be represented by timestamped measurement-shaped values.
