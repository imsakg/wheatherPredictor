"""Canonical observations; timestamps are aware UTC, all values are SI-derived."""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from .units import finite


@dataclass(frozen=True)
class Observation:
    timestamp: datetime
    temperature: float
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[float] = None
    precipitation: Optional[float] = None

    def __post_init__(self):
        if not isinstance(self.timestamp,datetime):
            raise ValueError("timestamp must be a datetime")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))
        object.__setattr__(self,"temperature",finite(self.temperature))
        for name in ("humidity", "pressure", "wind_speed", "wind_direction", "precipitation"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, finite(value))
        if self.humidity is not None and not 0 <= self.humidity <= 100:
            raise ValueError("humidity must be in [0, 100]")
        if self.pressure is not None and self.pressure <= 0:
            raise ValueError("pressure must be positive")
        for name in ("wind_speed", "precipitation"):
            if getattr(self, name) is not None and getattr(self, name) < 0:
                raise ValueError(name + " must be nonnegative")
        if self.wind_direction is not None and not 0 <= self.wind_direction < 360:
            raise ValueError("wind_direction must be in [0, 360)")
