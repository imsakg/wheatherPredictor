"""Explicit conversions to Celsius, m/s, hPa and millimetres."""
import math


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("measurement must be finite")
    return value


def fahrenheit_to_celsius(value):
    return (finite(value) - 32) * 5 / 9


def celsius_to_fahrenheit(value):
    return finite(value) * 9 / 5 + 32


def mph_to_ms(value):
    return finite(value) * 0.44704


def inhg_to_hpa(value):
    return finite(value) * 33.8638866667


def inches_to_mm(value):
    return finite(value) * 25.4
