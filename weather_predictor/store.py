"""SQLite persistence for one station per database; parameterized timestamp queries."""
import sqlite3
from datetime import datetime, timezone
from .schema import Observation
from .csvio import COLUMNS


class ObservationStore:
    def __init__(self, path):
        self.connection = sqlite3.connect(str(path))
        self.connection.execute("CREATE TABLE IF NOT EXISTS observations (timestamp TEXT PRIMARY KEY, temperature REAL NOT NULL, humidity REAL, pressure REAL, wind_speed REAL, wind_direction REAL, precipitation REAL, source TEXT NOT NULL)")
        self.connection.commit()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.connection.close()

    def put(self, observations, source="unknown"):
        rows = [tuple(getattr(row,name) if name!="timestamp" else row.timestamp.isoformat() for name in COLUMNS)+(source,) for row in observations]
        with self.connection:
            self.connection.executemany("INSERT OR REPLACE INTO observations VALUES (?,?,?,?,?,?,?,?)",rows)

    def query(self, start=None, end=None):
        where, params = [], []
        for stamp, operator in ((start,">="),(end,"<=")):
            if stamp is not None:
                if stamp.tzinfo is None or stamp.utcoffset() is None:
                    raise ValueError("query timestamps must be timezone-aware")
                where.append("timestamp " + operator + " ?")
                params.append(stamp.astimezone(timezone.utc).isoformat())
        sql = "SELECT " + ",".join(COLUMNS) + " FROM observations"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY timestamp"
        return [Observation(datetime.fromisoformat(row[0]),*row[1:]) for row in self.connection.execute(sql,params)]
