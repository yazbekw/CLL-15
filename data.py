"""Data loader — graceful handling of unavailable symbols."""
import os
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests

from config import (
    BASE_URL, REQUEST_TIMEOUT, SYMBOLS, CACHE_DIR, PERIOD,
    FUNDING_LOOKBACK_DAYS,
)

_session = requests.Session()
_session.headers.update({"User-Agent": "TFv5-research/1.0"})


def _get_json(path, params, retries=4):
    last_err = None
    for k in range(retries):
        try:
            r = _session.get(BASE_URL + path, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            obj = r.json()
            if obj.get("code") != 0:
                raise RuntimeError(obj.get("message", "CoinEx API error"))
            return obj.get("data", [])
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (k + 1))
    raise last_err


def _fetch_klines_range(market, start_ms, end_ms):
    rows = []
    cursor_end = end_ms
    while cursor_end > start_ms:
        chunk = _get_json("/futures/kline", {
            "market": market, "period": PERIOD, "limit": 1000,
            "start_time": start_ms, "end_time": cursor_end,
        })
        if not chunk:
            break
        rows.extend(chunk)
        ts = [int(x["created_at"]) for x in chunk]
        earliest = min(ts)
        if earliest <= start_ms or len(chunk) < 1000:
            break
        cursor_end = earliest - 1
        time.sleep(0.05)
    return rows


def _fetch_funding_range(market, start_ms, end_ms):
    rows = []
    page = 1
    while True:
        chunk = _get_json("/futures/funding-rate-history", {
            "market": market, "start_time": start_ms, "end_time": end_ms,
            "page": page, "limit": 1000,
        })
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < 1000:
            break
        page += 1
        if page > 30:
            break
        time.sleep(0.05)
    if not rows:
        return pd.DataFrame(columns=["funding_rate"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["funding_time"].astype("int64"),
                              unit="ms", utc=True).dt.tz_localize(None)
    df["funding_rate"] = pd.to_numeric(df["actual_funding_rate"], errors="coerce").fillna(0.0)
    return df[["ts", "funding_rate"]].drop_duplicates("ts").sort_values("ts").set_index("ts")


def _rows_to_df(rows):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["created_at"].astype("int64"),
                              unit="ms", utc=True).dt.tz_localize(None)
    for c in ["open", "close", "high", "low", "volume", "value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().drop_duplicates("ts").sort_values("ts").set_index("ts")
    return df[["open", "close", "high", "low", "volume", "value"]]


def _kline_path(market):
    return os.path.join(CACHE_DIR, f"{market}_{PERIOD}.parquet")


def _funding_path(market):
    return os.path.join(CACHE_DIR, f"{market}_funding.parquet")


def _last_ms(index):
    if len(index) == 0:
        return 0
    return int(index.max().value // 10**6)


def load_klines(market, days, force_refresh=False):
    path = _kline_path(market)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000

    df_cached = None
    if os.path.exists(path) and not force_refresh:
        try:
            df_cached = pd.read_parquet(path)
        except Exception:
            df_cached = None

    if df_cached is not None and not df_cached.empty:
        last = _last_ms(df_cached.index)
        if last < end_ms - 4 * 3600 * 1000:
            try:
                new_rows = _fetch_klines_range(market, last + 1, end_ms)
                if new_rows:
                    new_df = _rows_to_df(new_rows)
                    df_cached = pd.concat([df_cached, new_df]).drop_duplicates().sort_index()
            except Exception as e:
                print(f"      [warn] delta failed {market}: {e}")
        df_cached = df_cached[df_cached.index >= pd.to_datetime(start_ms, unit="ms")]
        if not df_cached.empty:
            df_cached.to_parquet(path)
            return df_cached

    rows = _fetch_klines_range(market, start_ms, end_ms)
    if not rows:
        raise RuntimeError(f"No klines for {market}")
    df = _rows_to_df(rows)
    df.to_parquet(path)
    return df


def load_funding(market, days=FUNDING_LOOKBACK_DAYS, force_refresh=False):
    path = _funding_path(market)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000

    df_cached = None
    if os.path.exists(path) and not force_refresh:
        try:
            df_cached = pd.read_parquet(path)
        except Exception:
            df_cached = None

    if df_cached is not None and not df_cached.empty:
        last = _last_ms(df_cached.index)
        if last < end_ms - 8 * 3600 * 1000:
            try:
                new = _fetch_funding_range(market, last + 1, end_ms)
                if not new.empty:
                    df_cached = pd.concat([df_cached, new]).drop_duplicates().sort_index()
            except Exception as e:
                print(f"      [warn] funding delta failed {market}: {e}")
        df_cached = df_cached[df_cached.index >= pd.to_datetime(start_ms, unit="ms")]
        if not df_cached.empty:
            df_cached.to_parquet(path)
            return df_cached

    df = _fetch_funding_range(market, start_ms, end_ms)
    df.to_parquet(path)
    return df


def load_all(days, with_funding=True, force_refresh=False):
    """Load all symbols. Skip those that fail (unavailable on CoinEx)."""
    data, funding = {}, {}
    failed = []
    for s in SYMBOLS:
        try:
            print(f"    loading 4H {s}...")
            d = load_klines(s, days, force_refresh)
            if len(d) < 100:
                failed.append(s)
                print(f"      [skip] not enough data for {s}")
                continue
            data[s] = d
            if with_funding:
                print(f"    loading funding {s}...")
                funding[s] = load_funding(s, FUNDING_LOOKBACK_DAYS, force_refresh)
        except Exception as e:
            failed.append(s)
            print(f"      [skip] {s}: {e}")
    if failed:
        print(f"\n  ⚠ skipped symbols: {failed}")
        print(f"  running with {len(data)} coins\n")
    if not data:
        raise RuntimeError("No symbols loaded successfully.")
    return data, funding


def common_index(data):
    idx = None
    for df in data.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    return idx.sort_values()
