from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "staged"
RAW = ROOT / "raw"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)

ENERGY_URL = "https://raw.githubusercontent.com/LuvU3OOO/EneryConsumptionForecast/9a8a7792d3476428be68eb5936f2d7ec018c1ae4/energy_dataset.csv"
ECB_URL = "https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?startPeriod=2017-01-01&endPeriod=2017-12-31&format=csvdata"
GTFS = {
    "mbta": "https://cdn.mbta.com/MBTA_GTFS.zip",
    "trimet": "https://developer.trimet.org/schedule/gtfs.zip",
    "cta": "https://www.transitchicago.com/downloads/sch_data/google_transit.zip",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, path: Path) -> None:
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    path.write_bytes(r.content)


def stage_energy() -> dict:
    raw = RAW / "energy_dataset.csv"
    download(ENERGY_URL, raw)
    df = pd.read_csv(raw)
    ts = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.loc[ts.dt.year.eq(2017)].copy()
    df["timestamp_utc"] = ts.loc[df.index]
    keep = [
        "timestamp_utc",
        "generation wind onshore",
        "generation solar",
        "total load actual",
        "price day ahead",
        "price actual",
    ]
    df = df[keep].rename(columns={
        "generation wind onshore": "wind_onshore_mw",
        "generation solar": "solar_mw",
        "total load actual": "system_load_mw",
        "price day ahead": "price_day_ahead_eur_per_mwh",
        "price actual": "price_actual_eur_per_mwh",
    }).sort_values("timestamp_utc")
    df = df.drop_duplicates("timestamp_utc", keep="first")
    df["month"] = df["timestamp_utc"].dt.month
    outputs = []
    for month, g in df.groupby("month"):
        path = OUT / f"spain_energy_2017_{int(month):02d}.csv"
        g.drop(columns="month").to_csv(path, index=False)
        outputs.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "rows": len(g)})
    return {
        "source_url": ENERGY_URL,
        "raw_sha256": sha256(raw),
        "rows": len(df),
        "start": str(df["timestamp_utc"].min()),
        "end": str(df["timestamp_utc"].max()),
        "outputs": outputs,
    }


def stage_ecb() -> dict:
    raw = RAW / "ecb_usd_eur_2017.csv"
    download(ECB_URL, raw)
    df = pd.read_csv(raw)
    time_col = next(c for c in df.columns if c.upper() in {"TIME_PERIOD", "TIME_PERIOD_START"})
    value_col = next(c for c in df.columns if c.upper() == "OBS_VALUE")
    out = pd.DataFrame({
        "date": pd.to_datetime(df[time_col], errors="coerce").dt.date.astype(str),
        "usd_per_eur": pd.to_numeric(df[value_col], errors="coerce"),
    }).dropna().drop_duplicates("date").sort_values("date")
    path = OUT / "ecb_usd_per_eur_2017.csv"
    out.to_csv(path, index=False)
    return {
        "source_url": ECB_URL,
        "raw_sha256": sha256(raw),
        "rows": len(out),
        "mean_usd_per_eur": float(out["usd_per_eur"].mean()),
        "output": {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)},
    }


def parse_hms(value: object) -> float:
    if pd.isna(value):
        return np.nan
    parts = str(value).split(":")
    if len(parts) != 3:
        return np.nan
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(float(parts[2]))


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def read_zip_table(zf: zipfile.ZipFile, name: str, usecols=None) -> pd.DataFrame:
    with zf.open(name) as f:
        return pd.read_csv(f, low_memory=False, usecols=usecols)


def service_dates(zf: zipfile.ZipFile) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    names = set(zf.namelist())
    rows = []
    if "calendar.txt" in names:
        cal = read_zip_table(zf, "calendar.txt")
        for r in cal.itertuples(index=False):
            start = pd.Timestamp(str(int(r.start_date)))
            end = pd.Timestamp(str(int(r.end_date)))
            days = pd.date_range(start, end, freq="D")
            weekday_cols = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
            active = [bool(getattr(r, weekday_cols[d.weekday()])) for d in days]
            for d, a in zip(days, active):
                if a:
                    rows.append((str(r.service_id), d.normalize(), 1))
    base = pd.DataFrame(rows, columns=["service_id","date","active"])
    if "calendar_dates.txt" in names:
        ex = read_zip_table(zf, "calendar_dates.txt")
        ex["date"] = pd.to_datetime(ex["date"].astype(str))
        for r in ex.itertuples(index=False):
            key = (base["service_id"].astype(str).eq(str(r.service_id)) & base["date"].eq(r.date.normalize()))
            if int(r.exception_type) == 1 and not key.any():
                base.loc[len(base)] = [str(r.service_id), r.date.normalize(), 1]
            elif int(r.exception_type) == 2:
                base = base.loc[~key]
    base = base.drop_duplicates(["service_id","date"])
    dates = sorted(base["date"].unique())
    return base, dates


def shape_distances(zf: zipfile.ZipFile) -> dict[str, float]:
    shapes = read_zip_table(zf, "shapes.txt", usecols=["shape_id","shape_pt_lat","shape_pt_lon","shape_pt_sequence"])
    shapes = shapes.sort_values(["shape_id","shape_pt_sequence"])
    out = {}
    for sid, g in shapes.groupby("shape_id", sort=False):
        lat = pd.to_numeric(g["shape_pt_lat"], errors="coerce").to_numpy(float)
        lon = pd.to_numeric(g["shape_pt_lon"], errors="coerce").to_numpy(float)
        if len(g) < 2:
            out[str(sid)] = 0.0
            continue
        d = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
        out[str(sid)] = float(np.nansum(d))
    return out


def stage_gtfs(site: str, url: str) -> dict:
    raw = RAW / f"{site}_gtfs.zip"
    download(url, raw)
    with zipfile.ZipFile(raw) as zf:
        names = set(zf.namelist())
        routes = read_zip_table(zf, "routes.txt")
        bus_types = {3, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712}
        route_type = pd.to_numeric(routes.get("route_type"), errors="coerce")
        bus_route_ids = set(routes.loc[route_type.isin(bus_types), "route_id"].astype(str))
        trips = read_zip_table(zf, "trips.txt")
        trips["route_id"] = trips["route_id"].astype(str)
        trips = trips.loc[trips["route_id"].isin(bus_route_ids)].copy()
        if trips.empty:
            raise RuntimeError(f"{site}: no bus trips found")
        trips["service_id"] = trips["service_id"].astype(str)
        trips["trip_id"] = trips["trip_id"].astype(str)
        trips["shape_id"] = trips.get("shape_id", "").astype(str)
        trips["block_id"] = trips.get("block_id", trips["trip_id"]).fillna(trips["trip_id"]).astype(str)
        shape_km = shape_distances(zf)
        trips["trip_km"] = trips["shape_id"].map(shape_km).fillna(0.0)
        st = read_zip_table(zf, "stop_times.txt", usecols=["trip_id","arrival_time","departure_time","stop_sequence"])
        st["trip_id"] = st["trip_id"].astype(str)
        st = st.loc[st["trip_id"].isin(set(trips["trip_id"]))]
        st["arr_s"] = st["arrival_time"].map(parse_hms)
        st["dep_s"] = st["departure_time"].map(parse_hms)
        span = st.groupby("trip_id").agg(start_s=("dep_s","min"), end_s=("arr_s","max")).reset_index()
        trips = trips.merge(span, on="trip_id", how="left")
        # Missing/zero shape distances are inferred from scheduled duration at a conservative urban-bus speed.
        duration_h = np.maximum(0.1, (trips["end_s"] - trips["start_s"]) / 3600.0)
        trips.loc[trips["trip_km"].le(0), "trip_km"] = duration_h.loc[trips["trip_km"].le(0)] * 22.0
        active, dates = service_dates(zf)
    # Limit to the first 84 consecutive service dates to keep the derived release compact while retaining season-free chronology.
    all_dates = pd.date_range(min(dates), max(dates), freq="D")
    if len(all_dates) > 84:
        all_dates = all_dates[:84]
    active = active.loc[active["date"].isin(all_dates)].copy()
    trips_active = trips.merge(active[["service_id","date"]], on="service_id", how="inner")
    block = trips_active.groupby(["date","block_id"], as_index=False).agg(
        distance_km=("trip_km","sum"),
        return_seconds=("end_s","max"),
        first_seconds=("start_s","min"),
        trip_count=("trip_id","nunique"),
    )
    # Duty classes are based on fixed distance thresholds, not outcome-tuned quantiles.
    block["class"] = pd.cut(block["distance_km"], bins=[-np.inf, 120, 240, np.inf], labels=["LD","MD","HD"]).astype(str)
    kg_per_km = {"LD":0.075, "MD":0.090, "HD":0.105}
    block["hydrogen_kg"] = block["distance_km"] * block["class"].map(kg_per_km).astype(float)
    block["return_hour"] = (block["return_seconds"].fillna(0).astype(float) // 3600).astype(int) % 24
    timeline = pd.MultiIndex.from_product([all_dates, range(24)], names=["date","hour"]).to_frame(index=False)
    timeline["timestamp_local"] = pd.to_datetime(timeline["date"]) + pd.to_timedelta(timeline["hour"], unit="h")
    demand = block.pivot_table(index=["date","return_hour"], columns="class", values="hydrogen_kg", aggfunc="sum", fill_value=0).reset_index()
    demand = demand.rename(columns={"return_hour":"hour"})
    out = timeline.merge(demand, on=["date","hour"], how="left").fillna({"LD":0.0,"MD":0.0,"HD":0.0})
    for cls in ["LD","MD","HD"]:
        if cls not in out:
            out[cls] = 0.0
        out = out.rename(columns={cls:f"demand_{cls}_kg"})
    out["site"] = site
    path = OUT / f"{site}_schedule_derived_hydrogen_demand.csv"
    out[["site","timestamp_local","demand_LD_kg","demand_MD_kg","demand_HD_kg"]].to_csv(path, index=False)
    blocks_path = OUT / f"{site}_vehicle_blocks.csv"
    block.to_csv(blocks_path, index=False)
    return {
        "site": site,
        "source_url": url,
        "raw_sha256": sha256(raw),
        "service_start": str(all_dates.min().date()),
        "service_end": str(all_dates.max().date()),
        "bus_routes": len(bus_route_ids),
        "vehicle_blocks": len(block),
        "total_distance_km": float(block["distance_km"].sum()),
        "total_hydrogen_kg": float(block["hydrogen_kg"].sum()),
        "output": {"path":str(path.relative_to(ROOT)), "sha256":sha256(path), "rows":len(out)},
        "blocks": {"path":str(blocks_path.relative_to(ROOT)), "sha256":sha256(blocks_path), "rows":len(block)},
        "evidence_class":"OFFICIAL_GTFS_SCHEDULE_DERIVED_MODEL_CALIBRATED_HYDROGEN_DEMAND",
    }


def main() -> None:
    manifest = {
        "energy": stage_energy(),
        "ecb": stage_ecb(),
        "gtfs": [],
    }
    for site, url in GTFS.items():
        manifest["gtfs"].append(stage_gtfs(site, url))
    path = OUT / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
