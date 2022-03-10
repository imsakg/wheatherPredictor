"""Standalone, offline HTML forecast summaries; no remote scripts or assets."""
from html import escape
from .visualization import forecast_chart
from .units import finite


def forecast_report(observations, predictions, title="Station weather forecast", intervals=None):
    rows, predictions = list(observations), list(predictions)
    chart = forecast_chart([row.temperature for row in rows],
                           [row["temperature"] for row in predictions],intervals)
    cells = ''.join('<tr><td>{}</td><td>{:.2f}</td></tr>'.format(escape(str(row["timestamp"])),finite(row["temperature"])) for row in predictions)
    title = escape(str(title))
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>'+title+'</title><style>body{font:16px system-ui;max-width:900px;margin:2rem auto;padding:1rem}svg{width:100%}td,th{padding:.4rem;text-align:left}</style>'
            '</head><body><h1>'+title+'</h1><p>Temperature in Celsius; timestamps UTC. '
            'Model output, not official meteorological guidance. Not a safety-critical forecast.</p>'+chart+
            '<table><thead><tr><th>Forecast time</th><th>Temperature (C)</th></tr></thead><tbody>'+cells+
            '</tbody></table><p>Sample-index chart: regularly spaced observations are required. '
            'Error bands, if present, are empirical and require independent calibration.</p></body></html>')
