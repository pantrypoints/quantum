"""API fetchers (no keys): USGS earthquake catalogue, Open-Meteo rainfall archive."""
import datetime as dt
import numpy as np
import pandas as pd
import requests

USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"
OPENMETEO = "https://archive-api.open-meteo.com/v1/archive"


def usgs_events(lat, lon, radius_km, min_mag, years):
    end = dt.date.today(); start = end.replace(year=end.year - years)
    r = requests.get(USGS, params=dict(format="geojson", latitude=lat, longitude=lon, maxradiuskm=radius_km,
                                       minmagnitude=min_mag, starttime=start, endtime=end, limit=20000), timeout=30)
    r.raise_for_status()
    f = r.json()["features"]
    return pd.DataFrame({"id": [x["id"] for x in f], "mag": [x["properties"]["mag"] for x in f]})


def quake_rates(evA, evB, tiers, years):
    """tiers: 4 ascending magnitude thresholds -> quarter steps k=1..4. Shared events = same catalogue id in both radii."""
    def lam(ev):
        k = np.searchsorted(tiers, ev["mag"].values, side="right")
        return np.array([(k == i).sum() for i in range(1, 5)]) / years, k
    l1, k1 = lam(evA); l2, k2 = lam(evB)
    shared = set(evA["id"]) & set(evB["id"])
    c = evA[evA["id"].isin(shared) & (k1 >= 1)].shape[0] / years
    return np.maximum(l1, 1e-6), np.maximum(l2, 1e-6), c


def daily_rain(lat, lon, years):
    end = dt.date.today() - dt.timedelta(days=7); start = end.replace(year=end.year - years)
    r = requests.get(OPENMETEO, params=dict(latitude=lat, longitude=lon, start_date=str(start), end_date=str(end),
                                            daily="precipitation_sum", timezone="UTC"), timeout=60)
    r.raise_for_status()
    d = r.json()["daily"]
    return pd.Series(d["precipitation_sum"], index=pd.to_datetime(d["time"])).fillna(0)


def rain_trigger(sA, sB, window, thrA, thrB):
    """Annual trigger = max rolling-window rainfall >= threshold. Returns q1, q2, cr_share (solved from the model)."""
    ann = lambda s, th: (s.rolling(window).sum().groupby(s.index.year).max() >= th)
    a, b = ann(sA, thrA), ann(sB, thrB)
    a, b = a.align(b, join="inner")
    n = len(a); fl = .5 / n
    q1, q2 = max(a.mean(), fl), max(b.mean(), fl)
    m1, m2 = -np.log(1 - q1), -np.log(1 - q2)
    p00 = max(((~a) & (~b)).mean(), 1e-9)
    cr = np.clip(m1 + m2 + np.log(p00), 0, min(m1, m2))
    return q1, q2, float(cr / min(m1, m2)), n
