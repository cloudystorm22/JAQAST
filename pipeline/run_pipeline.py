#!/usr/bin/env python3
"""
JAQAST pipeline — versi headless (bukan Colab).

Alur (dipindahkan dari JAQAST2_FIX.ipynb, logika inti tidak diubah):
  1. Tulis ~/.cdsapirc dari environment variable (secret), bukan hardcode.
  2. Download CAMS forecast (00Z, area DKI Jakarta + Kota Tangerang).
  3. Ekstrak NetCDF, sampling di centroid tiap wilayah -> timeseries hourly WIB.
  4. Hitung ISPU (rolling 24 jam progresif, ambil maks jam 09/15 WIB per hari).
  5. Hitung agregat meteorologi harian, gabungkan ke tabel ISPU.
  6. Prediksi kategori dengan model PyCaret (opsional, jika model tersedia).
  7. Gabungkan ke polygon wilayah -> tulis docs/data/latest.geojson + latest.json
     (dua file inilah yang dibaca website).
  8. Bersihkan file sementara (zip, nc) supaya repo tidak membengkak.

Didesain supaya AMAN dijalankan otomatis setiap hari lewat GitHub Actions:
  - Semua kredensial dari environment variable / GitHub Secrets.
  - Kalau download/proses gagal di tengah jalan, GeoJSON lama TIDAK ditimpa
    (fail-safe), jadi website tetap menampilkan data terakhir yang valid.
"""

import os
import sys
import glob
import json
import shutil
import zipfile
import traceback
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TZ = ZoneInfo("Asia/Jakarta")

# ---------------------------------------------------------------------------
# 0) KONFIGURASI (semua bisa dioverride lewat environment variable)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

ADS_URL = os.environ.get("ADS_URL", "https://ads.atmosphere.copernicus.eu/api")
ADS_KEY = os.environ.get("ADS_KEY")  # WAJIB diisi lewat GitHub Secret, jangan hardcode

RUN_MODE = os.environ.get("RUN_MODE", "yesterday")  # data CAMS run 00Z "yesterday" biasanya paling lengkap
AREA_BBOX = [-6.0, 106, -6.4, 107.1]  # [North, West, South, East]
AREA_TAG = "DKI"

GJSON_PATH = Path(os.environ.get("BOUND_GJSON", REPO_ROOT / "data" / "dki_tangerang_fix.geojson"))
MODEL_PATH = Path(os.environ.get("MODEL_PATH", REPO_ROOT / "pipeline" / "model" / "model_pycaret_uas"))
FEATURE_COLS_PATH = Path(os.environ.get("FEATURE_COLS_PATH", REPO_ROOT / "pipeline" / "model" / "model_feature_cols.json"))

WORK_DIR = Path(os.environ.get("WORK_DIR", "/tmp/jaqast_work"))
OUT_ROOT = WORK_DIR / "cams_extracted"

OUT_GEOJSON = Path(os.environ.get("OUT_GEOJSON", REPO_ROOT / "docs" / "data" / "latest.geojson"))
OUT_META = Path(os.environ.get("OUT_META", REPO_ROOT / "docs" / "data" / "latest.json"))
OUT_HISTORY_CSV = Path(os.environ.get("OUT_HISTORY_CSV", REPO_ROOT / "data" / "history.csv"))

R_d = 287.05  # J/(kg.K)

BP_IDN = {
    "pm10": [0, 50, 150, 350, 420, 500],
    "so2": [0, 52, 180, 400, 800, 1200],
    "co": [0, 4000, 8000, 15000, 30000, 45000],
    "o3": [0, 120, 235, 400, 800, 1000],
    "no2": [0, 80, 200, 1130, 2260, 3000],
}
I_BINS = [0, 50, 100, 200, 300, 500]

ISPU_CATEGORY_BREAKS = [
    (0, 51, "Baik"),
    (51, 101, "Sedang"),
    (101, 200, "Tidak Sehat"),
    (200, 300, "Sangat Tidak Sehat"),
    (300, 501, "Berbahaya"),
]


def log(msg):
    ts = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S WIB")
    print(f"[{ts}] {msg}", flush=True)


def ispu_category(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    for lo, hi, label in ISPU_CATEGORY_BREAKS:
        if lo <= v < hi:
            return label
    return "Berbahaya"


# ---------------------------------------------------------------------------
# 1) SETUP CDS/ADS CREDENTIALS
# ---------------------------------------------------------------------------
def setup_cdsapirc():
    if not ADS_KEY:
        raise RuntimeError(
            "ADS_KEY tidak ditemukan di environment. "
            "Set sebagai GitHub Secret bernama ADS_KEY (lihat README)."
        )
    cfg = f"url: {ADS_URL}\nkey: {ADS_KEY}\n"
    Path.home().joinpath(".cdsapirc").write_text(cfg)
    log("~/.cdsapirc ditulis dari secret (key tidak dicetak ke log).")


# ---------------------------------------------------------------------------
# 2) DOWNLOAD + EKSTRAK CAMS
# ---------------------------------------------------------------------------
def download_and_extract():
    import cdsapi

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    DATASET = "cams-global-atmospheric-composition-forecasts"
    BASE = {
        "variable": [
            "particulate_matter_10um", "surface_pressure", "carbon_monoxide",
            "nitrogen_dioxide", "ozone", "sulphur_dioxide", "temperature",
            "u_component_of_wind", "v_component_of_wind",
            "surface_net_solar_radiation", "total_precipitation", "relative_humidity",
        ],
        "pressure_level": ["1000"],
        "type": ["forecast"],
        "data_format": "netcdf_zip",
        "area": AREA_BBOX,
    }

    now_wib = datetime.now(TZ)
    if RUN_MODE == "today":
        target_local_date = now_wib.date()
    elif RUN_MODE == "yesterday":
        target_local_date = (now_wib - timedelta(days=1)).date()
    else:
        raise ValueError("RUN_MODE harus 'today' atau 'yesterday'")

    date_str = target_local_date.strftime("%Y-%m-%d")
    time_str = "00:00"
    log(f"Target run CAMS: {date_str} {time_str} UTC (RUN_MODE={RUN_MODE})")

    BASE_WITH_DT = dict(BASE)
    BASE_WITH_DT["date"] = [f"{date_str}/{date_str}"]
    BASE_WITH_DT["time"] = [time_str]

    def leads_for_day(k):
        a = 1 + 24 * k
        b = min(24 * (k + 1), 120)
        return a, b

    c = cdsapi.Client()
    zip_files = []
    for k in range(5):
        a, b = leads_for_day(k)
        req = dict(BASE_WITH_DT)
        req["leadtime_hour"] = [str(h) for h in range(a, b + 1)]
        fname = f"cams_fc_{target_local_date.strftime('%Y%m%d')}_{time_str.replace(':', '')}_{AREA_TAG}_UTCd{k}_{a:02d}-{b:02d}.zip"
        out_path = WORK_DIR / fname
        log(f"Download {fname} ...")
        c.retrieve(DATASET, req).download(str(out_path))
        zip_files.append(out_path)

    log(f"{len(zip_files)} file ZIP terunduh, mengekstrak...")
    for zp in zip_files:
        parts = zp.stem.split("_")
        uidx = parts.index([p for p in parts if p.startswith("UTCd")][0])
        run_date = parts[2]
        area = "_".join(parts[3:uidx])
        day_tag = parts[uidx]
        lead_tag = parts[uidx + 1]
        dest = OUT_ROOT / area / f"{run_date}_{day_tag}_{lead_tag}"
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp, "r") as z:
            z.extractall(dest)
    log("Ekstraksi selesai.")


# ---------------------------------------------------------------------------
# 3) SAMPLING + ISPU (logika sama seperti notebook, cell 19)
# ---------------------------------------------------------------------------
def ispu_idn(c_val, pol):
    if pd.isna(c_val):
        return np.nan
    bp = BP_IDN[pol]
    if c_val >= bp[-1]:
        return 500.0
    for i in range(len(bp) - 1):
        if bp[i] <= c_val <= bp[i + 1]:
            Cl, Ch = bp[i], bp[i + 1]
            Il, Ih = I_BINS[i], I_BINS[i + 1]
            return Il + (c_val - Cl) * (Ih - Il) / (Ch - Cl)
    Cl, Ch = bp[0], bp[1]
    Il, Ih = I_BINS[0], I_BINS[1]
    return Il + (c_val - Cl) * (Ih - Il) / (Ch - Cl)


def open_nc_robust(p):
    import xarray as xr

    errors = []
    for eng in ("h5netcdf", "netcdf4", "scipy"):
        try:
            ds = xr.open_dataset(p, engine=eng, decode_times=False, mask_and_scale=False)
            ds.load()
            return ds
        except Exception as e:
            errors.append(f"{eng}: {e}")

    # Semua engine gagal -> kemungkinan besar file bukan NetCDF valid
    # (download putus/kepotong, atau ADS balas halaman error, bukan data).
    size = Path(p).stat().st_size if Path(p).exists() else -1
    head = b""
    try:
        with open(p, "rb") as f:
            head = f.read(200)
    except Exception:
        pass
    diag = (
        f"Gagal buka NetCDF: {p}\n"
        f"  Ukuran file: {size} bytes\n"
        f"  200 byte pertama (untuk cek apakah ini benar NetCDF/HDF5, "
        f"tanda file NetCDF4 valid diawali b'\\x89HDF'): {head!r}\n"
        + "\n".join(f"  - {e}" for e in errors)
    )
    raise RuntimeError(diag)


def build_ds_with_time(nc_dir: Path):
    import xarray as xr
    import cftime
    files = list(nc_dir.glob("*.nc"))
    assert files, f"Tidak ada .nc di {nc_dir}"
    ds = xr.merge([open_nc_robust(str(p)) for p in files], compat="override", join="outer")

    frt, fp = ds["forecast_reference_time"], ds["forecast_period"]
    frt0 = cftime.num2date(
        frt.values,
        units=frt.attrs.get("units", "seconds since 1970-01-01 00:00:00"),
        calendar=frt.attrs.get("calendar", "proleptic_gregorian"),
    )
    frt0 = frt0[0] if isinstance(frt0, (list, tuple, np.ndarray)) else frt0
    base = pd.Timestamp(datetime(frt0.year, frt0.month, frt0.day, frt0.hour, frt0.minute, frt0.second, tzinfo=timezone.utc))
    valid = base + pd.to_timedelta(np.asarray(fp.values, float), unit="h")

    ds = ds.assign_coords(valid_time=("forecast_period", valid)).swap_dims({"forecast_period": "valid_time"})
    if "go3" in ds and "o3" not in ds:
        ds = ds.rename({"go3": "o3"})
    if "r" in ds and "rh" not in ds:
        ds = ds.rename({"r": "rh"})
    keep = [v for v in ["pm10", "so2", "co", "o3", "no2", "u", "v", "t", "rh", "sp", "ssr", "tp"] if v in ds]
    return ds[keep]


def hourly_at_centroid(ds, gdf):
    keep = [v for v in ["pm10", "so2", "co", "o3", "no2", "u", "v", "t", "rh", "sp", "ssr", "tp"] if v in ds]
    ds = ds[keep]
    rows = []
    for _, r in gdf.iterrows():
        name = str(r["name"])
        ctr = r.geometry.centroid
        pt = ds.sel(latitude=float(ctr.y), longitude=float(ctr.x), method="nearest")
        df = pt.to_dataframe().reset_index().rename(columns={"valid_time": "time_utc"})
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
        df["time_wib"] = df["time_utc"].dt.tz_convert("Asia/Jakarta")
        df["wilayah"] = name
        rows.append(df)
    return pd.concat(rows, ignore_index=True).sort_values(["wilayah", "time_wib"])


def interpolate_hourly(df, time_col="time_wib"):
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col]).dt.tz_convert("Asia/Jakarta").dt.floor("h")
    out = out.sort_values(time_col).drop_duplicates(time_col, keep="last").set_index(time_col)
    full_idx = pd.date_range(out.index.min(), out.index.max(), freq="h", tz="Asia/Jakarta")
    out = out.reindex(full_idx)
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.interpolate(method="time").ffill().bfill()
    return out.reset_index().rename(columns={"index": time_col})


def ispu_rolling_24h_reports(df_one: pd.DataFrame) -> pd.DataFrame:
    need_cols = ["sp", "t", "pm10", "so2", "co", "o3", "no2", "u", "v", "ssr", "tp"]
    cols_exist = [c for c in need_cols if c in df_one.columns]
    df = df_one[["time_wib"] + cols_exist].copy()
    df = interpolate_hourly(df, time_col="time_wib")
    df = df.set_index("time_wib").infer_objects(copy=False).apply(pd.to_numeric, errors="coerce")

    rho = df["sp"] / (R_d * df["t"])
    df["pm10_ugm3"] = df["pm10"] * 1e9
    for g in ["so2", "co", "o3", "no2"]:
        if g in df.columns:
            df[f"{g}_ugm3"] = df[g] * rho * 1e9

    cols_ug = [c for c in ["pm10_ugm3", "so2_ugm3", "co_ugm3", "o3_ugm3", "no2_ugm3"] if c in df.columns]
    roll = df[cols_ug].rolling(window=24, min_periods=1).mean()
    cnt = df[cols_ug].rolling(window=24, min_periods=1).count()
    samples_24h = cnt.min(axis=1)

    sel = roll[roll.index.hour.isin([9, 15])].copy()
    if sel.empty:
        return pd.DataFrame(columns=["time_wib", "ispu_pm10", "ispu_so2", "ispu_co", "ispu_o3", "ispu_no2", "ispu_total", "samples_24h", "provisional"])
    sel_samples = samples_24h.loc[sel.index]
    provisional_flag = sel_samples < 24

    out = []
    for ts, r in sel.iterrows():
        m = {}
        for p, src in {"pm10": "pm10_ugm3", "so2": "so2_ugm3", "co": "co_ugm3", "o3": "o3_ugm3", "no2": "no2_ugm3"}.items():
            m[p] = float(r[src]) if src in sel.columns else np.nan
        sub = {
            "ispu_pm10": ispu_idn(m["pm10"], "pm10") if not np.isnan(m["pm10"]) else np.nan,
            "ispu_so2": ispu_idn(m["so2"], "so2") if not np.isnan(m["so2"]) else np.nan,
            "ispu_co": ispu_idn(m["co"], "co") if not np.isnan(m["co"]) else np.nan,
            "ispu_o3": ispu_idn(m["o3"], "o3") if not np.isnan(m["o3"]) else np.nan,
            "ispu_no2": ispu_idn(m["no2"], "no2") if not np.isnan(m["no2"]) else np.nan,
        }
        vals = [v for v in sub.values() if not np.isnan(v)]
        sub["ispu_total"] = float(np.nanmax(vals)) if vals else np.nan
        out.append({"time_wib": ts, **sub, "samples_24h": int(sel_samples.loc[ts]), "provisional": bool(provisional_flag.loc[ts])})
    return pd.DataFrame(out).sort_values("time_wib").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4) METEOROLOGI HARIAN (logika sama seperti notebook, cell 21)
# ---------------------------------------------------------------------------
def daily_meteo_24h(df_one: pd.DataFrame) -> pd.DataFrame:
    need = ["time_wib", "u", "v", "t", "rh", "sp", "tp", "ssr"]
    cols = [c for c in need if c in df_one.columns]
    if len(cols) < 2:
        return pd.DataFrame(columns=["date_wib", "rr", "ws_avg", "ws_max", "wd_avg", "tt_air_max", "tt_air_avg", "tt_air_min", "rh_avg", "pp_air", "sr_avg", "sr_max"])

    df = df_one[cols].copy()
    df["time_wib"] = pd.to_datetime(df["time_wib"]).dt.tz_convert("Asia/Jakarta").dt.floor("h")
    df = df.sort_values("time_wib").drop_duplicates("time_wib", keep="last").set_index("time_wib")

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="Asia/Jakarta")
    missing_ts = len(full_idx.difference(df.index)) > 0
    num_df = df.apply(pd.to_numeric, errors="coerce")
    has_nan = num_df.isna().any().any()

    if missing_ts or has_nan:
        tmp = df.reset_index()
        tmp = interpolate_hourly(tmp, time_col="time_wib")
        df = tmp.set_index("time_wib").infer_objects(copy=False).apply(pd.to_numeric, errors="coerce")
    else:
        df = num_df

    ws = np.hypot(df.get("u", np.nan), df.get("v", np.nan))
    wd = (180 / np.pi) * np.arctan2(-df.get("u", np.nan), -df.get("v", np.nan)) % 360
    tC = df.get("t", np.nan) - 273.15
    pp_hPa = df.get("sp", np.nan) / 100.0

    if "tp" in df.columns:
        d_tp = df["tp"].diff()
        d_tp[d_tp < 0] = np.nan
        rr_mm = d_tp.fillna(0.0) * 1000.0
    else:
        rr_mm = pd.Series(index=df.index, dtype=float)

    if "ssr" in df.columns:
        d_ssr = df["ssr"].diff()
        is_acc = (d_ssr.dropna() >= 0).mean() > 0.7
        sr_Wm2 = (d_ssr.fillna(0.0) / 3600.0) if is_acc else df["ssr"]
    else:
        sr_Wm2 = pd.Series(index=df.index, dtype=float)

    rows = []
    for day, g in df.groupby(df.index.normalize()):
        ws_g, wd_g, tC_g = ws.loc[g.index], wd.loc[g.index], tC.loc[g.index]
        rh_g = df.get("rh", pd.Series(index=g.index, dtype=float)).loc[g.index]
        pp_g, rr_g, sr_g = pp_hPa.loc[g.index], rr_mm.loc[g.index], sr_Wm2.loc[g.index]
        th = np.deg2rad(wd_g.dropna().values)
        wd_avg = float((np.rad2deg(np.arctan2(np.mean(np.sin(th)), np.mean(np.cos(th)))) % 360)) if th.size > 0 else np.nan
        rows.append({
            "date_wib": day.date(), "rr": float(rr_g.sum(skipna=True)),
            "ws_avg": float(ws_g.mean(skipna=True)), "ws_max": float(ws_g.max(skipna=True)), "wd_avg": wd_avg,
            "tt_air_max": float(tC_g.max(skipna=True)), "tt_air_avg": float(tC_g.mean(skipna=True)), "tt_air_min": float(tC_g.min(skipna=True)),
            "rh_avg": float(rh_g.mean(skipna=True)), "pp_air": float(pp_g.mean(skipna=True)),
            "sr_avg": float(sr_g.mean(skipna=True)), "sr_max": float(sr_g.max(skipna=True)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5) PREDIKSI PYCARET (opsional)
# ---------------------------------------------------------------------------
def run_prediction(hasil_final: pd.DataFrame) -> pd.DataFrame:
    if not MODEL_PATH.with_suffix(".pkl").exists() and not Path(str(MODEL_PATH) + ".pkl").exists():
        log("Model PyCaret tidak ditemukan — lewati tahap prediksi (hanya ISPU + meteo yang dipakai).")
        return hasil_final

    try:
        from pycaret.classification import load_model, predict_model
    except ModuleNotFoundError:
        log(
            "⚠️ File model ada, tapi paket 'pycaret' belum terpasang — tahap prediksi DILEWATI, "
            "bukan dianggap gagal. Untuk mengaktifkan prediksi, buka pipeline/requirements.txt, "
            "hapus tanda # di baris pycaret/scikit-learn/numba, lalu commit ulang."
        )
        return hasil_final

    df_in = hasil_final.rename(columns={
        "ispu_pm10": "pm10", "ispu_so2": "so2", "ispu_co": "co", "ispu_o3": "o3", "ispu_no2": "no2",
    }).copy()

    model = load_model(str(MODEL_PATH))
    log(f"Model dimuat: {MODEL_PATH}")

    feature_cols = None
    if FEATURE_COLS_PATH.exists():
        feature_cols = json.loads(FEATURE_COLS_PATH.read_text())

    id_cols = [c for c in ["wilayah", "date_wib"] if c in df_in.columns]
    if feature_cols is None:
        candidate_cols = [c for c in df_in.columns if c not in id_cols]
        feature_cols = df_in[candidate_cols].select_dtypes(include=[np.number]).columns.tolist()

    X = df_in.copy()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feature_cols]

    pred = predict_model(model, data=X)
    out = pd.concat([df_in[id_cols].reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
    return hasil_final.merge(out.drop(columns=[c for c in id_cols if c not in ("wilayah", "date_wib")], errors="ignore"),
                              on=[c for c in id_cols if c in hasil_final.columns], how="left", suffixes=("", "_pred"))


# ---------------------------------------------------------------------------
# 6) TULIS OUTPUT UNTUK WEBSITE
# ---------------------------------------------------------------------------
def _json_safe(v):
    """Bikin nilai aman untuk json.dumps (NaN/Inf -> null, numpy types -> python native)."""
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.strftime("%Y-%m-%d")
    return v


def write_outputs(gdf, hasil_final: pd.DataFrame):
    """
    Menulis docs/data/latest.geojson berisi, untuk tiap wilayah, SATU polygon
    dengan properti "series": daftar data harian (ISPU + meteo) selama beberapa
    hari ke depan (hasil forecast CAMS). Frontend memakai "series" ini untuk
    date picker, jadi tidak perlu fetch ulang / duplikasi geometry per tanggal.
    """
    hasil_final = hasil_final.copy()
    hasil_final["date_wib"] = pd.to_datetime(hasil_final["date_wib"]).dt.strftime("%Y-%m-%d")
    hasil_final["ispu_kategori"] = hasil_final["ispu_total"].apply(ispu_category)

    series_cols = [c for c in [
        "date_wib", "ispu_total", "ispu_kategori",
        "ispu_pm10", "ispu_so2", "ispu_co", "ispu_o3", "ispu_no2",
        "tt_air_avg", "rh_avg", "rr", "ws_avg", "provisional",
    ] if c in hasil_final.columns]

    series_by_wilayah = {}
    all_dates = set()
    for wilayah, sub in hasil_final.sort_values("date_wib").groupby("wilayah"):
        recs = [{k: _json_safe(v) for k, v in rec.items()} for rec in sub[series_cols].to_dict(orient="records")]
        series_by_wilayah[wilayah] = recs
        all_dates.update(sub["date_wib"].tolist())

    features = []
    n_dengan_data = 0
    for _, row in gdf.iterrows():
        name = str(row["name"])
        series = series_by_wilayah.get(name, [])
        latest = series[-1] if series else {}
        if series:
            n_dengan_data += 1
        props = {"wilayah": name, "series": series}
        for k in series_cols:
            props[k] = latest.get(k)
        features.append({
            "type": "Feature",
            "geometry": json.loads(json.dumps(row.geometry.__geo_interface__)),
            "properties": props,
        })

    fc = {"type": "FeatureCollection", "features": features}

    OUT_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = OUT_GEOJSON.with_suffix(".tmp.geojson")
    tmp_path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(OUT_GEOJSON)
    log(f"GeoJSON ditulis: {OUT_GEOJSON} ({len(features)} wilayah, {len(all_dates)} tanggal tersedia)")

    meta = {
        "generated_at_wib": datetime.now(TZ).isoformat(),
        "n_wilayah": int(len(features)),
        "n_wilayah_dengan_data": int(n_dengan_data),
        "available_dates": sorted(all_dates),
        "sumber": "CAMS (Copernicus Atmosphere Monitoring Service)",
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"Metadata ditulis: {OUT_META}")

    OUT_HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    hist_new = hasil_final[["wilayah"] + series_cols]
    if OUT_HISTORY_CSV.exists():
        hist_old = pd.read_csv(OUT_HISTORY_CSV)
        hist_all = pd.concat([hist_old, hist_new], ignore_index=True)
        hist_all = hist_all.drop_duplicates(subset=["wilayah", "date_wib"], keep="last")
    else:
        hist_all = hist_new
    hist_all.to_csv(OUT_HISTORY_CSV, index=False)
    log(f"History diperbarui: {OUT_HISTORY_CSV} ({len(hist_all)} baris total)")


def cleanup():
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR, ignore_errors=True)
        log(f"File sementara dibersihkan: {WORK_DIR}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    import geopandas as gpd

    if not GJSON_PATH.exists():
        raise FileNotFoundError(
            f"Boundary GeoJSON tidak ditemukan di {GJSON_PATH}. "
            "Commit dulu file hasil cell 2-3 notebook (dki_tangerang_fix.geojson) ke data/."
        )

    setup_cdsapirc()
    download_and_extract()

    log("Membaca boundary wilayah...")
    gdf = gpd.read_file(GJSON_PATH).to_crs("EPSG:4326")

    day_dirs = sorted({p.parent for p in OUT_ROOT.rglob("*.nc")}, key=lambda p: p.stat().st_mtime)
    assert day_dirs, f"Tidak ada .nc hasil ekstrak di {OUT_ROOT}"

    log("Sampling nilai CAMS di centroid tiap wilayah...")
    hourlies = [hourly_at_centroid(build_ds_with_time(d), gdf) for d in day_dirs]
    df_hourly_all = pd.concat(hourlies, ignore_index=True)

    log("Menghitung ISPU rolling 24 jam...")
    rows = []
    for wilayah, sub in df_hourly_all.groupby("wilayah"):
        rep = ispu_rolling_24h_reports(sub)
        if rep.empty:
            continue
        rep["date_wib"] = rep["time_wib"].dt.date
        rep["hour"] = rep["time_wib"].dt.hour
        rep = (rep.sort_values(["date_wib", "ispu_total", "hour"], ascending=[True, False, False])
               .groupby("date_wib", as_index=False).head(1).drop(columns=["hour"]))
        rep.insert(0, "wilayah", wilayah)
        rep = rep.sort_values("date_wib").tail(5)
        rows.append(rep)
    laporan_max = pd.concat(rows, ignore_index=True).sort_values(["wilayah", "date_wib"]).reset_index(drop=True)

    log("Menghitung agregat meteorologi harian...")
    met_rows = []
    for wilayah, sub in df_hourly_all.groupby("wilayah"):
        met = daily_meteo_24h(sub)
        if met.empty:
            continue
        met.insert(0, "wilayah", wilayah)
        met_rows.append(met)
    meteo_daily = pd.concat(met_rows, ignore_index=True).sort_values(["wilayah", "date_wib"]).reset_index(drop=True)

    hasil_final = laporan_max.merge(meteo_daily, on=["wilayah", "date_wib"], how="left").sort_values(["wilayah", "date_wib"]).reset_index(drop=True)

    log("Menjalankan prediksi (jika model tersedia)...")
    hasil_final = run_prediction(hasil_final)

    write_outputs(gdf, hasil_final)
    log("Pipeline selesai ✅")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("❌ Pipeline GAGAL — GeoJSON lama TIDAK diubah, website tetap tampilkan data terakhir yang valid.")
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup()
