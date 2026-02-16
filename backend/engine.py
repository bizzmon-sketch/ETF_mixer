from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import os
import sqlite3
import threading
import time
from typing import Dict, Iterable, List, Tuple
import hashlib
import random

import numpy as np
import pandas as pd
import FinanceDataReader as fdr

CONFIG = {
  "etf_list_path": "data/etf_list.csv",
  "price_db_path": "data/prices.sqlite",
  "months": 6,
  "min_observations": 90,
  "trading_days_month": 21,
  "window_days": 120,
  "cache_ttl_sec": 600,
  "refresh_buffer_days": 3,
  "refresh_interval_sec": 600,
  "refresh_missing_batch_size": 80,
  "portfolio_candidate_limit": 20,
  "portfolio_max_holdings": 10,
  "portfolio_min_weight": 5,
  "portfolio_max_weight": 30,
  "portfolio_weight_step": 5,
  "fdr_source": None,
}

RISK_BUCKETS: List[Tuple[float, float, str]] = [
  (0.0, 3.0, "0-3%"),
  (3.0, 6.0, "3-6%"),
  (6.0, 9.0, "6-9%"),
  (9.0, 12.0, "9-12%"),
  (12.0, 15.0, "12-15%"),
]
RISK_BUCKET_LABELS = [label for _, _, label in RISK_BUCKETS] + ["15%+"]
ASSET_CLASSES = ["Equity", "Bond", "Alt", "CashLike"]
PORTFOLIO_VERSION = "sampled-v1"
PORTFOLIO_SEED_SALT = f"ETF_mixer|{PORTFOLIO_VERSION}"

_CACHE: Dict[str, object] = {
  "timestamp": None,
  "metrics": None,
  "recommendations": None,
  "delta3m": None,
  "returns_tail": None,
  "refresh_mode": None,
  "cached_at": None,
  "data_asof": None,
}

logger = logging.getLogger(__name__)

last_refresh_ts: float | None = None
refresh_lock = threading.Lock()
refresh_in_progress = False


def load_etf_list(path: str = CONFIG["etf_list_path"]) -> pd.DataFrame:
  df = pd.read_csv(path, encoding="utf-8-sig")
  df = df[["Code", "Name"]].dropna().drop_duplicates().reset_index(drop=True)
  return df


def fetch_close_prices(codes: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
  frames = []
  source = CONFIG.get("fdr_source")
  for code in codes:
    try:
      if source:
        data = fdr.DataReader(code, start, end, data_source=source)
      else:
        data = fdr.DataReader(code, start, end)
    except Exception:
      continue
    if "Close" not in data.columns:
      continue
    series = data["Close"].rename(code)
    frames.append(series)
  if not frames:
    return pd.DataFrame()
  close = pd.concat(frames, axis=1).sort_index()
  return close


def _ensure_db_dir(db_path: str) -> None:
  folder = os.path.dirname(db_path)
  if folder and not os.path.isdir(folder):
    os.makedirs(folder, exist_ok=True)


def _connect_db() -> sqlite3.Connection:
  db_path = CONFIG["price_db_path"]
  _ensure_db_dir(db_path)
  return sqlite3.connect(db_path)


def _init_db(conn: sqlite3.Connection) -> None:
  conn.execute(
    """
    CREATE TABLE IF NOT EXISTS prices (
      code TEXT NOT NULL,
      date TEXT NOT NULL,
      close REAL NOT NULL,
      PRIMARY KEY(code, date)
    )
    """
  )
  conn.commit()


def _chunked(values: Iterable[str], size: int) -> Iterable[List[str]]:
  batch = []
  for value in values:
    batch.append(value)
    if len(batch) >= size:
      yield batch
      batch = []
  if batch:
    yield batch


def _fetch_last_dates(conn: sqlite3.Connection, codes: List[str]) -> Dict[str, str]:
  if not codes:
    return {}
  last_dates: Dict[str, str] = {}
  for chunk in _chunked(codes, 900):
    placeholders = ",".join("?" for _ in chunk)
    query = f"""
      SELECT code, MAX(date) AS max_date
      FROM prices
      WHERE code IN ({placeholders})
      GROUP BY code
    """
    for code, max_date in conn.execute(query, chunk):
      if max_date:
        last_dates[code] = max_date
  return last_dates


def _upsert_prices(conn: sqlite3.Connection, rows: List[Tuple[str, str, float]]) -> None:
  if not rows:
    return
  conn.executemany(
    """
    INSERT INTO prices (code, date, close)
    VALUES (?, ?, ?)
    ON CONFLICT(code, date) DO UPDATE SET close=excluded.close
    """,
    rows,
  )


def _fetch_series_from_reader(code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
  source = CONFIG.get("fdr_source")
  try:
    if source:
      data = fdr.DataReader(code, start, end, data_source=source)
    else:
      data = fdr.DataReader(code, start, end)
  except Exception:
    return None
  if "Close" not in data.columns:
    return None
  series = data["Close"].dropna()
  if series.empty:
    return None
  return series


def update_prices_incremental(codes: List[str], buffer_days: int = 10) -> None:
  if not codes:
    return
  end = pd.Timestamp.today().normalize()
  default_start = end - pd.DateOffset(months=CONFIG["months"])
  buffer_days = min(int(buffer_days), 10)
  with _connect_db() as conn:
    _init_db(conn)
    last_dates = _fetch_last_dates(conn, codes)
    for code in codes:
      try:
        last_date_str = last_dates.get(code)
        if last_date_str:
          try:
            last_date = pd.Timestamp(last_date_str)
          except Exception:
            last_date = None
        else:
          last_date = None
        start = default_start
        if last_date is not None:
          start = last_date - pd.Timedelta(days=buffer_days)
        series = _fetch_series_from_reader(code, start, end)
        if series is None:
          continue
        rows = [
          (code, idx.strftime("%Y-%m-%d"), float(value))
          for idx, value in series.items()
        ]
        _upsert_prices(conn, rows)
        conn.commit()
      except Exception as exc:
        logger.warning("price update failed for %s: %s", code, exc)
        continue


def load_close_prices(codes: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
  if not codes:
    return pd.DataFrame()
  start_str = start.strftime("%Y-%m-%d")
  end_str = end.strftime("%Y-%m-%d")
  frames = []
  with _connect_db() as conn:
    _init_db(conn)
    for chunk in _chunked(codes, 900):
      placeholders = ",".join("?" for _ in chunk)
      query = f"""
        SELECT code, date, close
        FROM prices
        WHERE code IN ({placeholders})
          AND date BETWEEN ? AND ?
      """
      params = list(chunk) + [start_str, end_str]
      df = pd.read_sql_query(query, conn, params=params)
      if df.empty:
        continue
      frames.append(df)
  if not frames:
    return pd.DataFrame()
  all_rows = pd.concat(frames, ignore_index=True)
  all_rows["date"] = pd.to_datetime(all_rows["date"])
  close = all_rows.pivot_table(index="date", columns="code", values="close", aggfunc="last")
  return close.sort_index()


def load_recent_prices(code: str, days: int = 120) -> List[Dict[str, object]]:
  if not code:
    return []
  try:
    days = int(days)
  except Exception:
    days = 120
  if days <= 0:
    days = 120
  rows: List[Dict[str, object]] = []
  with _connect_db() as conn:
    _init_db(conn)
    query = """
      SELECT date, close
      FROM prices
      WHERE code = ?
      ORDER BY date DESC
      LIMIT ?
    """
    for date_value, close in conn.execute(query, (code, days)):
      rows.append({
        "date": date_value,
        "close": float(close),
      })
  rows.reverse()
  return rows


def classify_risk(risk_pct: float) -> str | None:
  if np.isnan(risk_pct):
    return None
  for low, high, label in RISK_BUCKETS:
    if low <= risk_pct < high:
      return label
  if risk_pct >= RISK_BUCKETS[-1][1]:
    return "15%+"
  return None


def _compute_returns_tail(close: pd.DataFrame, window_days: int) -> pd.DataFrame:
  if close.empty:
    return pd.DataFrame()
  returns = close.pct_change().dropna(how="all")
  if returns.empty:
    return returns
  window_days = int(window_days)
  if window_days > 0:
    return returns.tail(window_days)
  return returns


def compute_metrics(
  etf_df: pd.DataFrame,
  close: pd.DataFrame,
  returns_tail: pd.DataFrame | None = None,
) -> pd.DataFrame:
  if close.empty:
    return pd.DataFrame(columns=[
      "Code", "Name", "return_6m", "return_120d", "sharpe_120d", "risk_6m", "risk_pct", "risk_bucket",
    ])

  counts = close.count()
  eligible = counts[counts >= CONFIG["min_observations"]].index.tolist()
  close = close[eligible].dropna(how="all")
  if close.empty:
    return pd.DataFrame(columns=[
      "Code", "Name", "return_6m", "return_120d", "sharpe_120d", "risk_6m", "risk_pct", "risk_bucket",
    ])

  returns = close.pct_change().dropna(how="all")
  if returns_tail is None:
    returns_tail = _compute_returns_tail(close, CONFIG["window_days"])
  if not returns_tail.empty:
    returns_tail = returns_tail.reindex(columns=eligible)
  return_6m = (close.iloc[-1] / close.iloc[0]) - 1
  if returns_tail.empty:
    return_120d = pd.Series(index=return_6m.index, dtype=float)
    sharpe_120d = pd.Series(index=return_6m.index, dtype=float)
  else:
    return_120d = (1 + returns_tail).prod() - 1
    returns_mean = returns_tail.mean()
    returns_std = returns_tail.std()
    sharpe_120d = (returns_mean / returns_std) * np.sqrt(252)
    sharpe_120d = sharpe_120d.replace([np.inf, -np.inf], np.nan)
    sharpe_120d = sharpe_120d.where(returns_std > 0)
  vol_month = returns.std() * np.sqrt(CONFIG["trading_days_month"])
  risk_pct = vol_month * 100

  metrics = pd.DataFrame({
    "Code": return_6m.index,
    "return_6m": return_6m.values,
    "return_120d": return_120d.values,
    "sharpe_120d": sharpe_120d.values,
    "risk_6m": vol_month.values,
    "risk_pct": risk_pct.values,
  })
  metrics["risk_bucket"] = metrics["risk_pct"].apply(classify_risk)
  metrics = metrics.merge(etf_df, on="Code", how="left")
  metrics = metrics[["Code", "Name", "return_6m", "return_120d", "sharpe_120d", "risk_6m", "risk_pct", "risk_bucket"]]
  return metrics


def compute_delta3m(close: pd.DataFrame) -> pd.DataFrame:
  if close.empty:
    return pd.DataFrame(columns=["Code", "return_prev3m", "return_recent3m", "delta_3m"])

  end = close.index.max()
  start = end - pd.DateOffset(months=CONFIG["months"])
  mid = end - pd.DateOffset(months=3)
  close_6m = close.loc[close.index >= start]
  if close_6m.empty:
    return pd.DataFrame(columns=["Code", "return_prev3m", "return_recent3m", "delta_3m"])

  records = []
  for code in close_6m.columns:
    series = close_6m[code].dropna()
    if series.empty:
      continue
    series_prev = series.loc[series.index <= mid]
    series_recent = series.loc[series.index >= mid]
    if series_prev.empty or series_recent.empty:
      continue
    start_price = series_prev.iloc[0]
    mid_price = series_prev.iloc[-1]
    end_price = series_recent.iloc[-1]
    if start_price == 0 or mid_price == 0:
      continue
    return_prev = (mid_price / start_price) - 1
    return_recent = (end_price / mid_price) - 1
    records.append({
      "Code": code,
      "return_prev3m": return_prev,
      "return_recent3m": return_recent,
      "delta_3m": return_recent - return_prev,
    })

  return pd.DataFrame.from_records(records)


def select_best_by_bucket(metrics: pd.DataFrame) -> pd.DataFrame:
  if metrics.empty:
    return pd.DataFrame(columns=["Code", "Name", "return_6m", "risk_6m", "risk_pct", "risk_bucket"])
  filtered = metrics.dropna(subset=["risk_bucket"])
  if filtered.empty:
    return pd.DataFrame(columns=["Code", "Name", "return_6m", "risk_6m", "risk_pct", "risk_bucket"])
  best = filtered.sort_values(["risk_bucket", "return_6m"], ascending=[True, False])
  best = best.groupby("risk_bucket", as_index=False).head(1)
  return best.reset_index(drop=True)


def classify_asset_class(name: str) -> str:
  if not name:
    return "Equity"
  raw = str(name)
  lowered = raw.lower()
  cashlike_lower = ["mmf", "cdbond", "kofr", "koribor", "cash"]
  cashlike_raw = ["현금", "단기", "초단기", "머니", "콜", "통안", "단기채"]
  alt_lower = ["reit", "commodity", "commodities"]
  alt_raw = ["리츠", "금", "은", "원유", "원자재", "희토류"]
  bond_lower = ["bond", "credit"]
  bond_raw = ["채권", "국채", "회사채", "국고", "공채", "크레딧", "듀레이션"]

  if any(keyword in lowered for keyword in cashlike_lower):
    return "CashLike"
  if any(keyword in raw for keyword in cashlike_raw):
    return "CashLike"
  if any(keyword in lowered for keyword in alt_lower):
    return "Alt"
  if any(keyword in raw for keyword in alt_raw):
    return "Alt"
  if any(keyword in lowered for keyword in bond_lower):
    return "Bond"
  if any(keyword in raw for keyword in bond_raw):
    return "Bond"
  return "Equity"


def _sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
  return df.sort_values(["return_6m", "risk_pct", "Code"], ascending=[False, True, True])


def _fallback_candidate_sort(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  df["fallback_score"] = df["return_6m"] / df["risk_pct"].clip(lower=1e-6)
  return df.sort_values(
    ["fallback_score", "return_6m", "risk_pct", "Code"],
    ascending=[False, False, True, True],
  )


def _allocate_portfolio_weights(
  holdings: List[Dict[str, object]],
  bucket_label: str,
) -> Dict[str, int]:
  min_w = int(CONFIG["portfolio_min_weight"])
  max_w = int(CONFIG["portfolio_max_weight"])
  step = int(CONFIG["portfolio_weight_step"])
  weights: Dict[str, int] = {h["Code"]: min_w for h in holdings}
  remaining = 100 - (min_w * len(holdings))

  bucket_bias = 0.0
  if bucket_label in ("12-15%", "15%+"):
    bucket_bias = 1.0
  elif bucket_label in ("9-12%", "6-9%"):
    bucket_bias = 0.5
  else:
    bucket_bias = -0.5

  def score(holding: Dict[str, object]) -> float:
    base = float(holding["return_6m"])
    asset_class = holding["asset_class"]
    if asset_class in ("Equity", "Alt"):
      return base + bucket_bias
    if asset_class in ("Bond", "CashLike"):
      return base - bucket_bias
    return base

  non_cash = [h for h in holdings if h["asset_class"] != "CashLike"]
  non_cash = sorted(non_cash, key=lambda h: (-score(h), h["Code"]))
  cashlike = [h for h in holdings if h["asset_class"] == "CashLike"]
  if cashlike:
    cashlike = sorted(cashlike, key=lambda h: (-score(h), h["Code"]))
  else:
    cashlike = []

  while remaining > 0:
    progressed = False
    for holding in non_cash:
      code = holding["Code"]
      if weights[code] >= max_w:
        continue
      if remaining < step:
        break
      weights[code] += step
      remaining -= step
      progressed = True
      if remaining == 0:
        break
    if not progressed:
      break

  for holding in cashlike:
    code = holding["Code"]
    while remaining > 0 and weights[code] < max_w:
      if remaining < step:
        break
      weights[code] += step
      remaining -= step

  if remaining > 0:
    logger.warning("portfolio allocation leftover: %s", remaining)
  return weights


def _build_portfolio_for_bucket(metrics: pd.DataFrame, bucket_label: str) -> Dict[str, object] | None:
  bucket_df = metrics.loc[metrics["risk_bucket"] == bucket_label].copy()
  if bucket_df.empty:
    return None
  bucket_df = bucket_df.dropna(subset=["Code", "Name", "return_6m", "risk_pct"])
  bucket_df["asset_class"] = bucket_df["Name"].apply(classify_asset_class)

  candidates_by_class: Dict[str, pd.DataFrame] = {}
  limit = int(CONFIG["portfolio_candidate_limit"])
  for asset_class in ASSET_CLASSES:
    class_df = bucket_df[bucket_df["asset_class"] == asset_class]
    if class_df.empty:
      return None
    candidates_by_class[asset_class] = _sort_candidates(class_df).head(limit)

  holdings_rows: List[Dict[str, object]] = []
  used_codes: set[str] = set()
  for asset_class in ASSET_CLASSES:
    top = candidates_by_class[asset_class].iloc[0]
    holdings_rows.append({
      "Code": top["Code"],
      "Name": top["Name"],
      "return_6m": float(top["return_6m"]),
      "risk_pct": float(top["risk_pct"]),
      "asset_class": asset_class,
    })
    used_codes.add(top["Code"])

  extra_pool = pd.concat(
    [candidates_by_class["Equity"], candidates_by_class["Bond"], candidates_by_class["Alt"]],
    ignore_index=True,
  )
  extra_pool = _sort_candidates(extra_pool)
  max_holdings = int(CONFIG["portfolio_max_holdings"])
  for _, row in extra_pool.iterrows():
    if len(holdings_rows) >= max_holdings:
      break
    if row["Code"] in used_codes:
      continue
    holdings_rows.append({
      "Code": row["Code"],
      "Name": row["Name"],
      "return_6m": float(row["return_6m"]),
      "risk_pct": float(row["risk_pct"]),
      "asset_class": row["asset_class"],
    })
    used_codes.add(row["Code"])

  weights = _allocate_portfolio_weights(holdings_rows, bucket_label)

  holdings_output = []
  for holding in holdings_rows:
    code = holding["Code"]
    weight = weights.get(code, 0)
    if weight <= 0:
      continue
    holdings_output.append({
      "Code": code,
      "Name": holding["Name"],
      "weight": weight,
      "asset_class": holding["asset_class"],
      "return_6m": holding["return_6m"],
      "risk_pct": holding["risk_pct"],
    })

  holdings_output = sorted(
    holdings_output,
    key=lambda h: (-h["weight"], h["Code"]),
  )

  total_weight = sum(h["weight"] for h in holdings_output)
  if total_weight != 100:
    logger.warning("portfolio weights sum %s for bucket %s", total_weight, bucket_label)

  total_return = sum((h["weight"] / 100) * h["return_6m"] for h in holdings_output)
  total_risk = sum((h["weight"] / 100) * h["risk_pct"] for h in holdings_output)

  return {
    "risk_bucket": bucket_label,
    "return_6m": total_return,
    "risk_pct": total_risk,
    "holdings": [
      {
        "Code": h["Code"],
        "Name": h["Name"],
        "weight": h["weight"],
        "asset_class": h["asset_class"],
      }
      for h in holdings_output
    ],
  }


def _get_data_asof(close: pd.DataFrame) -> str | None:
  if close.empty:
    return None
  last_dates = []
  for code in close.columns:
    series = close[code].dropna()
    if series.empty:
      continue
    last_dates.append(series.index.max())
  if not last_dates:
    return None
  ts = min(last_dates)
  if isinstance(ts, pd.Timestamp):
    return ts.strftime("%Y-%m-%d")
  return None


def _now_kst() -> datetime:
  return datetime.now(timezone(timedelta(hours=9)))


def _maybe_start_background_refresh(codes: List[str]) -> str:
  global last_refresh_ts, refresh_in_progress
  now = time.time()
  with refresh_lock:
    if refresh_in_progress:
      return "inflight"
    if last_refresh_ts is not None:
      if now - last_refresh_ts < CONFIG["refresh_interval_sec"]:
        return "skipped"
    refresh_in_progress = True
    last_refresh_ts = now

  thread = threading.Thread(
    target=_background_refresh,
    args=(codes,),
    daemon=True,
  )
  thread.start()
  return "started"


def _background_refresh(codes: List[str]) -> None:
  global refresh_in_progress
  try:
    refresh_buffer = min(max(int(CONFIG["refresh_buffer_days"]), 3), 10)
    missing_codes: List[str] = []
    codes_with_last: List[str] = []
    with _connect_db() as conn:
      _init_db(conn)
      last_dates = _fetch_last_dates(conn, codes)
    for code in codes:
      if code in last_dates:
        codes_with_last.append(code)
      else:
        missing_codes.append(code)
    if codes_with_last:
      update_prices_incremental(codes_with_last, buffer_days=refresh_buffer)
    if missing_codes:
      batch_size = int(CONFIG["refresh_missing_batch_size"])
      for batch in _chunked(missing_codes, batch_size):
        update_prices_incremental(batch, buffer_days=refresh_buffer)
  finally:
    with refresh_lock:
      refresh_in_progress = False
    _CACHE["timestamp"] = None


def _refresh_cache_from_db() -> None:
  etf_df = load_etf_list()
  codes = etf_df["Code"].tolist()
  end = pd.Timestamp.today().normalize()
  start = end - pd.DateOffset(months=CONFIG["months"])
  close = load_close_prices(codes, start, end)

  returns_tail = _compute_returns_tail(close, CONFIG["window_days"])
  metrics = compute_metrics(etf_df, close, returns_tail=returns_tail)
  recommendations = select_best_by_bucket(metrics)
  delta3m = compute_delta3m(close)

  cached_at = _now_kst()
  _CACHE["timestamp"] = time.time()
  _CACHE["cached_at"] = cached_at.isoformat(timespec="seconds")
  _CACHE["data_asof"] = _get_data_asof(close)
  _CACHE["metrics"] = metrics
  _CACHE["recommendations"] = recommendations
  _CACHE["delta3m"] = delta3m
  _CACHE["returns_tail"] = returns_tail


def _get_cache() -> None:
  ts = _CACHE["timestamp"]
  if ts is None:
    _refresh_cache_from_db()
    refresh_status = _maybe_start_background_refresh(load_etf_list()["Code"].tolist())
    if refresh_status == "started":
      _CACHE["refresh_mode"] = "bg_refresh_started"
    elif refresh_status == "inflight":
      _CACHE["refresh_mode"] = "bg_refresh_inflight"
    else:
      _CACHE["refresh_mode"] = "db_immediate"
    return
  age = time.time() - ts
  if age > CONFIG["cache_ttl_sec"]:
    _refresh_cache_from_db()
    refresh_status = _maybe_start_background_refresh(load_etf_list()["Code"].tolist())
    if refresh_status == "started":
      _CACHE["refresh_mode"] = "bg_refresh_started"
    elif refresh_status == "inflight":
      _CACHE["refresh_mode"] = "bg_refresh_inflight"
    else:
      _CACHE["refresh_mode"] = "db_immediate"
    return
  refresh_status = _maybe_start_background_refresh(load_etf_list()["Code"].tolist())
  if refresh_status == "started":
    _CACHE["refresh_mode"] = "bg_refresh_started"
  elif refresh_status == "inflight":
    _CACHE["refresh_mode"] = "bg_refresh_inflight"
  else:
    _CACHE["refresh_mode"] = "cache_hit"


def _df_to_records(df: pd.DataFrame) -> List[Dict[str, object]]:
  records = df.to_dict(orient="records")
  cleaned = []
  for rec in records:
    cleaned_rec = {}
    for key, value in rec.items():
      if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        cleaned_rec[key] = None
      else:
        cleaned_rec[key] = value
    cleaned.append(cleaned_rec)
  return cleaned


def get_scatter_data() -> List[Dict[str, object]]:
  _get_cache()
  metrics = _CACHE["metrics"]
  return _df_to_records(metrics)


def get_scatter_meta() -> Dict[str, object]:
  if _CACHE["timestamp"] is None:
    _get_cache()
  if last_refresh_ts is None:
    last_refresh_value = None
  else:
    last_refresh_value = datetime.fromtimestamp(
      last_refresh_ts,
      timezone(timedelta(hours=9)),
    ).isoformat(timespec="seconds")
  return {
    "refresh_mode": _CACHE.get("refresh_mode"),
    "cached_at": _CACHE.get("cached_at"),
    "data_asof": _CACHE.get("data_asof"),
    "last_refresh_ts": last_refresh_value,
  }


def get_recommendations() -> List[Dict[str, object]]:
  _get_cache()
  recommendations = _CACHE["recommendations"]
  return _df_to_records(recommendations)


def get_delta3m() -> List[Dict[str, object]]:
  _get_cache()
  delta3m = _CACHE["delta3m"]
  return _df_to_records(delta3m)


def _bucket_bounds() -> List[Dict[str, object]]:
  bounds = []
  for low, high, label in RISK_BUCKETS:
    bounds.append({"label": label, "min": low, "max": high})
  bounds.append({"label": "15%+", "min": 15.0, "max": None})
  return bounds


def _risk_in_bucket(risk_pct: float, bucket: Dict[str, object]) -> bool:
  if risk_pct is None:
    return False
  if bucket["max"] is None:
    return risk_pct >= float(bucket["min"])
  return float(bucket["min"]) <= risk_pct < float(bucket["max"])


def _bucket_target(bucket: Dict[str, object]) -> float:
  if bucket["max"] is None:
    return float(bucket["min"])
  return (float(bucket["min"]) + float(bucket["max"])) / 2


def _build_candidate_pools(metrics: pd.DataFrame, config: Dict[str, object]) -> Dict[str, List[Dict[str, object]]]:
  df = metrics.dropna(subset=["Code", "Name", "return_6m", "risk_pct"]).copy()
  if df.empty:
    return {}
  df["asset_class"] = df["Name"].apply(classify_asset_class)
  df = _sort_candidates(df)
  topn_config = config.get("topN_by_class", 20)
  pools: Dict[str, List[Dict[str, object]]] = {}
  for asset_class in ASSET_CLASSES:
    if isinstance(topn_config, dict):
      topn_default = int(topn_config.get("default", 20))
      topn = int(topn_config.get(asset_class, topn_default))
    else:
      topn = int(topn_config)
    class_df = df[df["asset_class"] == asset_class]
    if class_df.empty:
      pools[asset_class] = []
      continue
    sharpe_df = class_df.dropna(subset=["sharpe_120d"]).copy()
    sharpe_df = sharpe_df.sort_values(["sharpe_120d", "return_6m", "risk_pct", "Code"], ascending=[False, False, True, True])
    if sharpe_df.empty:
      selected = _fallback_candidate_sort(class_df).head(topn)
    else:
      selected = sharpe_df.head(topn)
    pools[asset_class] = [
      {
        "Code": row["Code"],
        "Name": row["Name"],
        "return_6m": float(row["return_6m"]),
        "risk_pct": float(row["risk_pct"]),
        "asset_class": asset_class,
      }
      for _, row in selected.iterrows()
    ]
  return pools


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
  weights = np.array(weights, dtype=float)
  total = float(np.sum(weights))
  if total <= 0:
    return weights
  return weights / total


def _format_weight_percentages(weights: np.ndarray, decimals: int = 2) -> List[float]:
  weights = _normalize_weights(weights)
  percents = np.round(weights * 100, decimals=decimals)
  diff = 100.0 - float(np.sum(percents))
  if abs(diff) > 1e-6:
    idx = int(np.argmax(percents))
    percents[idx] = round(float(percents[idx] + diff), decimals)
  return [float(value) for value in percents]


def _qp_solve_weights(
  mu: np.ndarray,
  sigma: np.ndarray,
  gamma: float,
  solver: str | None = None,
) -> np.ndarray | None:
  try:
    import cvxpy as cp
  except Exception:
    return None
  n = len(mu)
  if n == 0:
    return None
  w = cp.Variable(n, nonneg=True)
  objective = cp.Maximize(mu @ w - gamma * cp.quad_form(w, sigma))
  constraints = [cp.sum(w) == 1]
  prob = cp.Problem(objective, constraints)
  try:
    if solver:
      prob.solve(solver=solver, warm_start=True)
    else:
      prob.solve(warm_start=True)
  except Exception:
    return None
  if w.value is None:
    return None
  return np.array(w.value, dtype=float).flatten()


def _portfolio_risk_pct(weights: np.ndarray, sigma: np.ndarray) -> float | None:
  if weights.size == 0:
    return None
  variance = float(weights.T @ sigma @ weights)
  if variance < 0:
    return None
  daily_vol = np.sqrt(variance)
  monthly_vol = daily_vol * np.sqrt(CONFIG["trading_days_month"]) * 100
  return float(monthly_vol)


def _generate_portfolios_qp(
  metrics: pd.DataFrame,
  config: Dict[str, object],
  returns_tail: pd.DataFrame | None,
  score_mode: str,
  debug: bool = False,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
  try:
    import cvxpy  # noqa: F401
    solver_available = True
  except Exception:
    solver_available = False

  meta = {
    "version": "qp-v1",
    "topN_by_class": config.get("topN_by_class"),
    "window_days": int(CONFIG["window_days"]),
    "solver_available": solver_available,
    "score_mode": score_mode,
  }
  qp_audit = None
  if debug:
    stage0_df = metrics.dropna(subset=["Code", "Name"]).copy()
    qp_audit = {
      "S0_universe": int(len(stage0_df)),
      "buckets": {},
    }
    for bucket in _bucket_bounds():
      label = bucket["label"]
      s1_df = stage0_df[stage0_df["risk_bucket"] == label].copy()
      s2_df = s1_df.dropna(subset=["return_6m", "risk_pct", "sharpe_120d"]).copy()
      class_counts = {asset_class: 0 for asset_class in ASSET_CLASSES}
      unknown_count = 0
      if not s2_df.empty:
        s2_df["asset_class"] = s2_df["Name"].apply(classify_asset_class)
        known_mask = s2_df["asset_class"].isin(ASSET_CLASSES)
        unknown_count = int((~known_mask).sum())
        s3_df = s2_df.loc[known_mask]
        class_counts = {
          asset_class: int((s3_df["asset_class"] == asset_class).sum())
          for asset_class in ASSET_CLASSES
        }
      feasible = all(class_counts[asset_class] >= 1 for asset_class in ASSET_CLASSES)
      qp_audit["buckets"][label] = {
        "S1_bucket": int(len(s1_df)),
        "S2_metrics_non_nan": int(len(s2_df)),
        "S3_asset_class": {
          "known": int(sum(class_counts.values())),
          "unknown": int(unknown_count),
          "by_class": class_counts,
        },
        "S4_class_min_feasible": int(1 if feasible else 0),
        "S5_portfolios_produced": 0,
      }

  pools = _build_candidate_pools(metrics, config)
  meta["universe_counts"] = {key: len(value) for key, value in pools.items()}
  if not pools or any(len(pools.get(cls, [])) == 0 for cls in ASSET_CLASSES):
    items = []
    for bucket in _bucket_bounds():
      items.append({
        "risk_bucket": bucket["label"],
        "risk_pct": None,
        "return_6m": None,
        "holdings": [],
        "error": "insufficient_candidates",
      })
    if qp_audit is not None:
      meta["qp_audit"] = qp_audit
    return items, meta

  if not solver_available:
    items = []
    for bucket in _bucket_bounds():
      items.append({
        "risk_bucket": bucket["label"],
        "risk_pct": None,
        "return_6m": None,
        "holdings": [],
        "error": "missing_solver",
      })
    if qp_audit is not None:
      meta["qp_audit"] = qp_audit
    return items, meta

  if returns_tail is None or returns_tail.empty:
    items = []
    for bucket in _bucket_bounds():
      items.append({
        "risk_bucket": bucket["label"],
        "risk_pct": None,
        "return_6m": None,
        "holdings": [],
        "error": "missing_covariance",
      })
    if qp_audit is not None:
      meta["qp_audit"] = qp_audit
    return items, meta

  items: List[Dict[str, object]] = []
  max_holdings = int(CONFIG["portfolio_max_holdings"])
  if max_holdings < 4:
    max_holdings = 4

  for bucket in _bucket_bounds():
    holdings_rows: List[Dict[str, object]] = []
    used_codes: set[str] = set()
    for asset_class in ASSET_CLASSES:
      pool = pools.get(asset_class, [])
      if not pool:
        holdings_rows = []
        break
      top = pool[0]
      holdings_rows.append(dict(top))
      used_codes.add(top["Code"])

    if not holdings_rows:
      items.append({
        "risk_bucket": bucket["label"],
        "risk_pct": None,
        "return_6m": None,
        "holdings": [],
        "error": "insufficient_candidates",
      })
      continue

    extra_pool = (pools.get("Equity", []) + pools.get("Alt", []))
    for row in extra_pool:
      if len(holdings_rows) >= max_holdings:
        break
      if row["Code"] in used_codes:
        continue
      holdings_rows.append(dict(row))
      used_codes.add(row["Code"])

    codes = [h["Code"] for h in holdings_rows]
    returns_slice = returns_tail.reindex(columns=codes).dropna(how="any")
    if returns_slice.empty:
      items.append({
        "risk_bucket": bucket["label"],
        "risk_pct": None,
        "return_6m": None,
        "holdings": [],
        "error": "missing_covariance",
      })
      continue

    sigma = returns_slice.cov().values
    if np.isnan(sigma).any():
      items.append({
        "risk_bucket": bucket["label"],
        "risk_pct": None,
        "return_6m": None,
        "holdings": [],
        "error": "missing_covariance",
      })
      continue
    sigma = (sigma + sigma.T) / 2
    sigma = sigma + np.eye(len(codes)) * 1e-8

    mu = np.array([float(h["return_6m"]) for h in holdings_rows], dtype=float)

    gamma = 1.0
    best_weights = None
    best_risk = None
    best_gamma = None
    gamma_low = 1e-4
    gamma_high = 1e4

    for _ in range(12):
      weights = _qp_solve_weights(mu, sigma, gamma)
      if weights is None:
        break
      weights = _normalize_weights(weights)
      risk_pct = _portfolio_risk_pct(weights, sigma)
      best_weights = weights
      best_risk = risk_pct
      best_gamma = gamma
      if risk_pct is None:
        break
      if _risk_in_bucket(risk_pct, bucket):
        break
      if bucket["max"] is None:
        if risk_pct < float(bucket["min"]):
          gamma_high = gamma
          gamma = (gamma_low + gamma) / 2
        else:
          gamma_low = gamma
          gamma = (gamma + gamma_high) / 2
      else:
        if risk_pct < float(bucket["min"]):
          gamma_high = gamma
          gamma = (gamma_low + gamma) / 2
        else:
          gamma_low = gamma
          gamma = (gamma + gamma_high) / 2

    if best_weights is None:
      items.append({
        "risk_bucket": bucket["label"],
        "risk_pct": None,
        "return_6m": None,
        "holdings": [],
        "error": "solver_failed",
      })
      continue

    weights_pct = _format_weight_percentages(best_weights, decimals=2)
    holdings_output = []
    total_return = 0.0
    for holding, weight_pct in zip(holdings_rows, weights_pct):
      total_return += (weight_pct / 100) * float(holding["return_6m"])
      holdings_output.append({
        "Code": holding["Code"],
        "Name": holding["Name"],
        "weight": float(weight_pct),
        "asset_class": holding["asset_class"],
      })

    holdings_output = sorted(holdings_output, key=lambda h: (-h["weight"], h["Code"]))

    items.append({
      "risk_bucket": bucket["label"],
      "risk_pct": best_risk,
      "return_6m": total_return,
      "holdings": holdings_output,
      "meta": {
        "gamma": best_gamma,
        "within_bucket": _risk_in_bucket(best_risk, bucket) if best_risk is not None else False,
        "lo": bucket["min"],
        "hi": bucket["max"],
      },
    })

  if qp_audit is not None:
    for item in items:
      label = item.get("risk_bucket")
      if label in qp_audit["buckets"]:
        qp_audit["buckets"][label]["S5_portfolios_produced"] = int(0 if item.get("error") else 1)
    meta["qp_audit"] = qp_audit

  return items, meta


def _sample_holding(rng: random.Random, pool: List[Dict[str, object]], used: set[str]) -> Dict[str, object] | None:
  if not pool:
    return None
  candidates = [item for item in pool if item["Code"] not in used]
  if not candidates:
    return None
  idx = rng.randrange(len(candidates))
  return candidates[idx]


def _sample_portfolio_holdings(
  rng: random.Random,
  pools: Dict[str, List[Dict[str, object]]],
  max_holdings: int,
) -> List[Dict[str, object]] | None:
  holdings: List[Dict[str, object]] = []
  used: set[str] = set()

  for asset_class in ASSET_CLASSES:
    picked = _sample_holding(rng, pools.get(asset_class, []), used)
    if not picked:
      return None
    holdings.append(picked)
    used.add(picked["Code"])

  extra_slots = max_holdings - len(holdings)
  if extra_slots <= 0:
    return holdings

  equity_bias = 0.7
  for _ in range(extra_slots):
    if rng.random() < equity_bias:
      first, second = "Equity", "Alt"
    else:
      first, second = "Alt", "Equity"
    picked = _sample_holding(rng, pools.get(first, []), used)
    if not picked:
      picked = _sample_holding(rng, pools.get(second, []), used)
    if not picked:
      break
    holdings.append(picked)
    used.add(picked["Code"])

  return holdings


def _compute_portfolio_metrics(
  holdings: List[Dict[str, object]],
  weights: Dict[str, int],
) -> Tuple[float, float, List[Dict[str, object]]]:
  total_return = 0.0
  total_risk = 0.0
  output_holdings = []
  for holding in holdings:
    code = holding["Code"]
    weight = weights.get(code, 0)
    if weight <= 0:
      continue
    total_return += (weight / 100) * float(holding["return_6m"])
    total_risk += (weight / 100) * float(holding["risk_pct"])
    output_holdings.append({
      "Code": holding["Code"],
      "Name": holding["Name"],
      "weight": weight,
      "asset_class": holding["asset_class"],
      "risk_pct": float(holding["risk_pct"]),
      "return_6m": float(holding["return_6m"]),
    })
  output_holdings = sorted(output_holdings, key=lambda h: (-h["weight"], h["Code"]))
  return total_return, total_risk, output_holdings


def _compute_portfolio_120d_metrics(
  holdings: List[Dict[str, object]],
  weights: Dict[str, int],
  returns_tail: pd.DataFrame | None,
) -> Tuple[float | None, float | None]:
  if returns_tail is None or returns_tail.empty:
    return None, None
  codes = [
    h["Code"] for h in holdings
    if weights.get(h["Code"], 0) > 0 and h["Code"] in returns_tail.columns
  ]
  if not codes:
    return None, None
  returns_slice = returns_tail[codes].dropna(how="any")
  if returns_slice.empty:
    return None, None
  weight_values = np.array([weights[code] / 100 for code in codes], dtype=float)
  rp = returns_slice.mul(weight_values, axis=1).sum(axis=1)
  if rp.empty:
    return None, None
  return_120d = (1 + rp).prod() - 1
  std = rp.std()
  if std is None or np.isnan(std) or std == 0:
    sharpe_120d = None
  else:
    sharpe_120d = (rp.mean() / std) * np.sqrt(252)
  return float(return_120d), None if sharpe_120d is None else float(sharpe_120d)


def _tune_weights_to_target(
  holdings: List[Dict[str, object]],
  weights: Dict[str, int],
  bucket: Dict[str, object],
  max_iters: int = 20,
  max_moves: int = 40,
  max_moves_per_iter: int = 6,
) -> Tuple[Dict[str, int], float, bool, int, int]:
  min_w = int(CONFIG["portfolio_min_weight"])
  max_w = int(CONFIG["portfolio_max_weight"])
  step = int(CONFIG["portfolio_weight_step"])
  holding_map = {h["Code"]: h for h in holdings}
  moves = 0
  lo = float(bucket["min"]) if bucket.get("min") is not None else 0.0
  hi = bucket.get("max")
  hi_value = float(hi) if hi is not None else None

  def portfolio_risk() -> float:
    total = 0.0
    for code, weight in weights.items():
      total += (weight / 100) * float(holding_map[code]["risk_pct"])
    return total

  def select_pair(
    receiver_pool: List[Dict[str, object]],
    donor_pool: List[Dict[str, object]],
  ) -> Tuple[str, str] | None:
    for receiver in receiver_pool:
      receiver_code = receiver["Code"]
      if weights.get(receiver_code, 0) + step > max_w:
        continue
      for donor in donor_pool:
        donor_code = donor["Code"]
        if weights.get(donor_code, 0) - step < min_w:
          continue
        return receiver_code, donor_code
    return None

  tune_iters = 0
  for iteration in range(1, max_iters + 1):
    tune_iters = iteration
    moved_this_iter = False
    for _ in range(max_moves_per_iter):
      if moves >= max_moves:
        break
      current_risk = portfolio_risk()
      within_bucket = _risk_in_bucket(current_risk, bucket)
      if within_bucket:
        return weights, current_risk, True, tune_iters, moves

      if hi_value is None:
        should_lower = False
        should_raise = current_risk < lo
      else:
        should_raise = current_risk < lo
        should_lower = current_risk >= hi_value

      if should_raise:
        receiver_pool = [
          h for h in holdings
          if h["asset_class"] in ("Equity", "Alt") and weights.get(h["Code"], 0) <= (max_w - step)
        ]
        receiver_pool = sorted(
          receiver_pool,
          key=lambda h: (-h["risk_pct"], -h["return_6m"], h["Code"]),
        )
        donor_pool = [
          h for h in holdings
          if h["asset_class"] in ("CashLike", "Bond") and weights.get(h["Code"], 0) >= (min_w + step)
        ]
        donor_pool = sorted(
          donor_pool,
          key=lambda h: (h["risk_pct"], h["return_6m"], h["Code"]),
        )
      elif should_lower:
        receiver_pool = [
          h for h in holdings
          if h["asset_class"] in ("CashLike", "Bond") and weights.get(h["Code"], 0) <= (max_w - step)
        ]
        receiver_pool = sorted(
          receiver_pool,
          key=lambda h: (h["risk_pct"], -h["return_6m"], h["Code"]),
        )
        donor_pool = [
          h for h in holdings
          if h["asset_class"] in ("Equity", "Alt") and weights.get(h["Code"], 0) >= (min_w + step)
        ]
        donor_pool = sorted(
          donor_pool,
          key=lambda h: (-h["risk_pct"], h["return_6m"], h["Code"]),
        )
      else:
        return weights, current_risk, True, tune_iters, moves

      pair = select_pair(receiver_pool, donor_pool)
      if not pair:
        break
      receiver_code, donor_code = pair
      weights[receiver_code] += step
      weights[donor_code] -= step
      moves += 1
      moved_this_iter = True

    if not moved_this_iter:
      break

  final_risk = portfolio_risk()
  return weights, final_risk, _risk_in_bucket(final_risk, bucket), tune_iters, moves


def _generate_portfolios_sampled(
  metrics: pd.DataFrame,
  config: Dict[str, object],
  returns_tail: pd.DataFrame | None,
  score_mode: str,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
  sample_count_default = int(config["sample_count"])
  sample_count_overrides = config.get("sample_count_by_bucket") or {}
  meta = {
    "version": PORTFOLIO_VERSION,
    "sample_count": sample_count_default,
    "sample_count_by_bucket": {
      "default": sample_count_default,
      **{key: int(value) for key, value in sample_count_overrides.items()},
    },
    "portfolio_size_range": [4, int(CONFIG["portfolio_max_holdings"])],
    "topN_by_class": 20,
    "seed_salt": PORTFOLIO_SEED_SALT,
    "universe_mode": "top20_by_asset_class",
    "window_days": int(CONFIG["window_days"]),
  }
  pools = _build_candidate_pools(metrics, config)
  meta["universe_counts"] = {key: len(value) for key, value in pools.items()}
  if not pools or any(len(pools.get(cls, [])) == 0 for cls in ASSET_CLASSES):
    items = []
    for bucket in _bucket_bounds():
      items.append({
        "risk_bucket": bucket["label"],
        "risk_pct": None,
        "return_6m": None,
        "holdings": [],
        "error": "insufficient_candidates",
        "meta": {
          "target_risk": None,
          "within_bucket": False,
          "chosen_risk": None,
          "distance_to_target": None,
          "tune_iters": 0,
          "moves": 0,
          "lo": bucket["min"],
          "hi": bucket["max"],
        },
      })
    return items, meta

  target_risk_map = {
    "0-3%": 1.5,
    "3-6%": 4.5,
    "6-9%": 7.5,
    "9-12%": 10.5,
    "12-15%": 13.5,
    "15%+": 15.6,
  }

  items: List[Dict[str, object]] = []
  max_holdings = int(CONFIG["portfolio_max_holdings"])
  min_holdings = 4
  if max_holdings < min_holdings:
    max_holdings = min_holdings
  score_mode = (score_mode or "sharpe").lower()
  for bucket in _bucket_bounds():
    local_sample_count = int(sample_count_overrides.get(bucket["label"], sample_count_default))
    target_risk = float(target_risk_map.get(bucket["label"], _bucket_target(bucket)))
    seed_source = f"{PORTFOLIO_SEED_SALT}|{bucket['label']}"
    seed_int = int(hashlib.md5(seed_source.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed_int)
    candidates: List[Dict[str, object]] = []
    for candidate_idx in range(local_sample_count):
      size_seed = f"{PORTFOLIO_SEED_SALT}|{bucket['label']}|{candidate_idx}|size"
      size_seed_int = int(hashlib.md5(size_seed.encode()).hexdigest()[:8], 16)
      size_rng = random.Random(size_seed_int)
      n_assets = size_rng.randint(min_holdings, max_holdings)
      holdings = _sample_portfolio_holdings(rng, pools, n_assets)
      if not holdings:
        continue
      weights = _allocate_portfolio_weights(holdings, bucket["label"])
      weights, tuned_risk, within_bucket, tune_iters, moves = _tune_weights_to_target(
        holdings,
        weights,
        bucket,
      )
      total_weight = sum(weights.values())
      if total_weight != 100:
        continue
      total_return, total_risk, output_holdings = _compute_portfolio_metrics(holdings, weights)
      return_120d, sharpe_120d = _compute_portfolio_120d_metrics(holdings, weights, returns_tail)
      candidates.append({
        "risk_bucket": bucket["label"],
        "risk_pct": total_risk,
        "return_6m": total_return,
        "return_120d": return_120d,
        "sharpe_120d": sharpe_120d,
        "score_mode": score_mode,
        "holdings": output_holdings,
        "meta": {
          "target_risk": target_risk,
          "within_bucket": within_bucket,
          "chosen_risk": total_risk,
          "distance_to_target": abs(total_risk - target_risk) if total_risk is not None else None,
          "tune_iters": tune_iters,
          "moves": moves,
          "lo": bucket["min"],
          "hi": bucket["max"],
        },
      })

    if not candidates:
      items.append({
        "risk_bucket": bucket["label"],
        "risk_pct": None,
        "return_6m": None,
        "holdings": [],
        "error": "no_candidates",
      })
      continue

    in_bucket = [c for c in candidates if _risk_in_bucket(c["risk_pct"], bucket)]
    if in_bucket:
      if score_mode == "return":
        def score_value(candidate: Dict[str, object]) -> float:
          return float(candidate.get("return_120d") or candidate.get("return_6m") or float("-inf"))
      else:
        def score_value(candidate: Dict[str, object]) -> float:
          value = candidate.get("sharpe_120d")
          if value is None or (isinstance(value, float) and np.isnan(value)):
            return float("-inf")
          return float(value)
      best = max(
        in_bucket,
        key=lambda c: (score_value(c), c.get("return_120d") or c.get("return_6m") or 0.0, -c["risk_pct"]),
      )
      items.append(best)
      continue

    best = min(
      candidates,
      key=lambda c: (abs(c["risk_pct"] - target_risk), -c["return_6m"]),
    )
    best["error"] = "bucket_fallback"
    if "meta" not in best:
      best["meta"] = {
        "target_risk": target_risk,
        "within_bucket": False,
        "chosen_risk": best.get("risk_pct"),
      }
    best["meta"]["within_bucket"] = False
    best["meta"]["chosen_risk"] = best.get("risk_pct")
    best["meta"]["error_reason"] = "no_within_candidates"
    if best.get("risk_pct") is not None:
      best["meta"]["distance_to_target"] = abs(best.get("risk_pct") - target_risk)
    else:
      best["meta"]["distance_to_target"] = None
    if "tune_iters" not in best["meta"]:
      best["meta"]["tune_iters"] = 0
    if "moves" not in best["meta"]:
      best["meta"]["moves"] = 0
    best["meta"]["lo"] = bucket["min"]
    best["meta"]["hi"] = bucket["max"]
    items.append(best)

  return items, meta


def _generate_portfolios_grid(
  metrics: pd.DataFrame,
  config: Dict[str, object],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
  meta = {
    "version": "grid-stub",
  }
  items = []
  for bucket in _bucket_bounds():
    items.append({
      "risk_bucket": bucket["label"],
      "risk_pct": None,
      "return_6m": None,
      "return_120d": None,
      "sharpe_120d": None,
      "score_mode": "return",
      "holdings": [],
      "error": "strategy_not_implemented",
    })
  return items, meta


def _generate_portfolios_bucket_fallback(
  metrics: pd.DataFrame,
  config: Dict[str, object],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
  meta = {
    "version": "bucket-fallback-stub",
  }
  items = []
  for bucket in _bucket_bounds():
    items.append({
      "risk_bucket": bucket["label"],
      "risk_pct": None,
      "return_6m": None,
      "return_120d": None,
      "sharpe_120d": None,
      "score_mode": "return",
      "holdings": [],
      "error": "strategy_not_implemented",
    })
  return items, meta


def generate_portfolios(
  items: pd.DataFrame | List[Dict[str, object]],
  strategy: str = "sampled",
  config: Dict[str, object] | None = None,
  returns_tail: pd.DataFrame | None = None,
  score_mode: str = "sharpe",
  debug: int | bool = False,
  debug_code: str | None = None,
) -> Dict[str, object]:
  if isinstance(items, list):
    metrics = pd.DataFrame.from_records(items)
  else:
    metrics = items.copy()

  default_config = {
    "sample_count": 1000,
    "sample_count_by_bucket": {},
    "topN_by_class": {
      "Equity": 25,
      "Alt": 20,
      "Bond": 20,
      "CashLike": 10,
    },
  }
  if config:
    default_config.update(config)
  score_mode = (score_mode or "sharpe").lower()
  if score_mode not in ("sharpe", "return"):
    score_mode = "sharpe"

  if metrics.empty:
    empty_items = []
    for bucket in _bucket_bounds():
      empty_items.append({
        "risk_bucket": bucket["label"],
        "risk_pct": None,
        "return_6m": None,
        "return_120d": None,
        "sharpe_120d": None,
        "score_mode": score_mode,
        "holdings": [],
        "error": "no_data",
      })
    return format_portfolios(
      empty_items,
      strategy,
      {"version": "empty"},
    )

  if strategy == "sampled":
    items, meta = _generate_portfolios_sampled(metrics, default_config, returns_tail, score_mode)
  elif strategy == "qp":
    items, meta = _generate_portfolios_qp(
      metrics,
      default_config,
      returns_tail,
      score_mode,
      debug=bool(debug),
    )
  elif strategy == "grid":
    items, meta = _generate_portfolios_grid(metrics, default_config)
  elif strategy == "bucket_fallback":
    items, meta = _generate_portfolios_bucket_fallback(metrics, default_config)
  else:
    items, meta = _generate_portfolios_sampled(metrics, default_config)
    meta["strategy_warning"] = "unknown_strategy"

  units = {
    "risk_unit": "monthly_vol_pct",
    "return_6m_unit": "cumulative",
    "return_120d_unit": "cumulative",
    "sharpe_120d_unit": "annualized",
    "window_days": int(CONFIG["window_days"]),
    "score_mode": score_mode,
  }
  if meta is None:
    meta = {}
  meta["units"] = units
  meta["score_mode"] = units["score_mode"]
  return format_portfolios(items, strategy, meta)


def format_portfolios(
  items: List[Dict[str, object]],
  strategy: str,
  meta: Dict[str, object] | None = None,
) -> Dict[str, object]:
  payload = {
    "count": len(items),
    "items": items,
    "strategy": strategy,
  }
  if meta is not None:
    payload["meta"] = meta
  return payload


def get_portfolios(
  strategy: str = "sampled",
  score: str = "sharpe",
  debug: int | bool = False,
  debug_code: str | None = None,
) -> Dict[str, object]:
  _get_cache()
  metrics = _CACHE["metrics"]
  returns_tail = _CACHE.get("returns_tail")
  if metrics is None:
    metrics = pd.DataFrame()
  debug_enabled = bool(debug)
  t0 = time.perf_counter() if debug_enabled else 0.0
  payload = generate_portfolios(
    metrics,
    strategy=strategy,
    returns_tail=returns_tail,
    score_mode=score,
    debug=debug_enabled,
    debug_code=debug_code,
  )
  if not debug_enabled:
    return payload

  counts: Dict[str, object] = {}
  if strategy == "qp":
    audit = (payload.get("meta") or {}).get("qp_audit") or {}
    buckets = audit.get("buckets") or {}
    counts = {
      "S0_universe": int(audit.get("S0_universe", 0)),
      "S1_bucket": {label: int(data.get("S1_bucket", 0)) for label, data in buckets.items()},
      "S2_metrics_non_nan": {label: int(data.get("S2_metrics_non_nan", 0)) for label, data in buckets.items()},
      "S3_asset_class": {
        label: {
          "known": int((data.get("S3_asset_class") or {}).get("known", 0)),
          "unknown": int((data.get("S3_asset_class") or {}).get("unknown", 0)),
        }
        for label, data in buckets.items()
      },
      "S4_class_min_feasible": {label: int(data.get("S4_class_min_feasible", 0)) for label, data in buckets.items()},
      "S5_portfolios_produced": {label: int(data.get("S5_portfolios_produced", 0)) for label, data in buckets.items()},
    }

  payload["debug"] = {
    "debug_code": debug_code,
    "note": "B0.5 audit enabled",
    # Stage meanings: S0 universe -> S1 bucket split -> S2 metric-valid rows -> S3 class-known rows -> S4 class-min feasible -> S5 portfolio produced.
    "counts": counts,
    "timing_ms": {
      "total": round((time.perf_counter() - t0) * 1000, 2),
    },
  }
  return payload


def get_price_series(code: str, days: int = 120) -> Dict[str, object]:
  name = None
  try:
    etf_df = load_etf_list()
    matched = etf_df[etf_df["Code"] == code]
    if not matched.empty:
      name = str(matched.iloc[0]["Name"])
  except Exception:
    name = None
  items = load_recent_prices(code, days)
  return {
    "code": code,
    "name": name,
    "days": days,
    "items": items,
  }


if __name__ == "__main__":
  _refresh_cache_from_db()
  print("scatter rows:", len(get_scatter_data()))
  print("recommendations:", len(get_recommendations()))
  print("delta3m rows:", len(get_delta3m()))
