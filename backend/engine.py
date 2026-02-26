from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Tuple
import hashlib
import random

import numpy as np
import pandas as pd
import FinanceDataReader as fdr

CONFIG = {
  "etf_list_path": "data/etf_list.csv",
  "price_db_path": "data/prices.sqlite",
  "months": 24,
  "min_observations": 90,
  "trading_days_month": 21,
  "window_days": 260,
  "weekly_window": 52,
  "weekly_min_observations": 30,
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

# [P0-2] 주봉 단위 규약 전역 상수
_WEEKLY_PERIODS_PER_YEAR = 52
_WEEKLY_SCALE_TO_MONTHLY = float(np.sqrt(52.0 / 12.0))
_RF_ANNUAL = 0.03
_RF_WEEKLY = _RF_ANNUAL / 52.0

RISK_BUCKETS: List[Tuple[float, float, str]] = [
  (0.0, 3.0, "0-3%"),
  (3.0, 6.0, "3-6%"),
  (6.0, 9.0, "6-9%"),
]
ASSET_CLASSES = ["Equity", "Bond", "Alt", "CashLike"]
PORTFOLIO_VERSION = "sampled-v1"
PORTFOLIO_SEED_SALT = f"ETF_mixer|{PORTFOLIO_VERSION}"

_CACHE: Dict[str, object] = {
  "timestamp": None,
  "metrics": None,
  "recommendations": None,
  "delta3m": None,
  "sharpe_13w": None,
  "returns_tail": None,
  "refresh_mode": None,
  "cached_at": None,
  "data_asof": None,
  "is_week_partial": False,
  "weekly_excluded": [],
  "weekly_last_label": None,
  "weekly_last_observed": None,
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
  return None


def build_weekly_price_matrix(
  price_df_daily: pd.DataFrame,
  window_weeks: int = 52,
) -> Dict[str, object]:
  """
  [P0-2] 일봉 close -> 주봉 리샘플링 -> 로그 수익률 52주 윈도우.
  당주 처리: 방식 B (당주 포함, partial 플래그 + 관측일/라벨 분리 기록).
  """
  empty_result = {
    "prices": pd.DataFrame(),
    "log_returns": pd.DataFrame(),
    "data_as_of": None,
    "weekly_last_label": None,
    "weekly_last_observed": None,
    "is_current_week_partial": False,
    "excluded": [],
    "n_weeks": 0,
  }
  if price_df_daily.empty:
    return empty_result

  try:
    last_observed = price_df_daily.dropna(how="all").index.max()
  except Exception:
    last_observed = None

  price_weekly = price_df_daily.resample("W-FRI", label="right", closed="right").last()
  if price_weekly.empty:
    return empty_result
  weekly_last_label = price_weekly.index.max()

  is_partial = False
  if last_observed is not None and weekly_last_label is not None:
    is_partial = bool(last_observed < weekly_last_label)

  if len(price_weekly) > window_weeks + 1:
    price_weekly = price_weekly.iloc[-(window_weeks + 1):]

  price_weekly = price_weekly.ffill(limit=1)

  min_obs = int(CONFIG.get("weekly_min_observations", 30))
  valid_counts = price_weekly.count()
  excluded = valid_counts[valid_counts < min_obs].index.tolist()
  valid_cols = valid_counts[valid_counts >= min_obs].index.tolist()
  price_weekly = price_weekly[valid_cols] if valid_cols else pd.DataFrame()
  if price_weekly.empty:
    return empty_result

  log_price = np.log(price_weekly.replace(0, np.nan))
  log_returns = log_price.diff().dropna(how="all")
  if len(log_returns) > window_weeks:
    log_returns = log_returns.iloc[-window_weeks:]

  def _fmt_dt(x: object) -> str | None:
    return x.strftime("%Y-%m-%d") if hasattr(x, "strftime") else (str(x)[:10] if x is not None else None)

  weekly_last_label_s = _fmt_dt(weekly_last_label)
  weekly_last_observed_s = _fmt_dt(last_observed)
  data_as_of = weekly_last_observed_s or weekly_last_label_s

  return {
    "prices": price_weekly,
    "log_returns": log_returns,
    "data_as_of": data_as_of,
    "weekly_last_label": weekly_last_label_s,
    "weekly_last_observed": weekly_last_observed_s,
    "is_current_week_partial": bool(is_partial),
    "excluded": excluded,
    "n_weeks": int(len(log_returns)),
  }


def compute_weekly_metrics_series(log_returns_series: pd.Series) -> Dict[str, float] | None:
  """
  [P0-2] 단일 ETF 주봉 로그 수익률 -> 지표
  내부 계산: log
  UI 표시: return_52w는 exp(sum(log))-1 (simple 변환)
  """
  r = log_returns_series.dropna()
  if len(r) < 30:
    return None

  std = float(r.std(ddof=1))
  if std < 1e-9:
    return None

  log_cumsum = float(r.sum())
  return_52w = float(np.exp(log_cumsum) - 1.0)

  risk_monthly_pct = std * _WEEKLY_SCALE_TO_MONTHLY * 100.0
  risk_annual_pct = std * np.sqrt(52.0) * 100.0

  mean_r = float(r.mean())
  sharpe_52w = (mean_r - _RF_WEEKLY) / std * np.sqrt(52.0)

  mean_log_r_weekly = mean_r
  mean_log_r_ann = mean_r * 52.0

  return {
    "return_52w": round(return_52w, 6),
    "risk_weekly_std": round(std, 8),
    "risk_pct": round(risk_monthly_pct, 4),
    "risk_annual_pct": round(risk_annual_pct, 4),
    "sharpe_52w_ann": round(sharpe_52w, 4),
    "mean_log_r_weekly": round(mean_log_r_weekly, 10),
    "mean_log_r_ann": round(mean_log_r_ann, 10),
  }


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
  returns_tail: pd.DataFrame | None = None,  # backward-compat, unused
) -> pd.DataFrame:
  """
  [P0-2] 주봉 52주 기반 메트릭 (내부 log, UI simple).
  하위 호환 필드 유지:
    return_6m=return_52w, sharpe_120d=sharpe_52w_ann
  """
  _ = returns_tail
  empty_cols = [
    "Code", "Name",
    "return_52w",
    "return_6m",
    "sharpe_52w_ann",
    "sharpe_120d",
    "risk_6m",
    "risk_pct",
    "risk_bucket",
    "mean_log_r_ann",
    "mean_log_r_weekly",
  ]

  if close.empty:
    return pd.DataFrame(columns=empty_cols)

  weekly_data = build_weekly_price_matrix(
    close,
    window_weeks=int(CONFIG.get("weekly_window", 52)),
  )
  log_returns = weekly_data["log_returns"]
  excluded = weekly_data["excluded"]
  if log_returns.empty:
    return pd.DataFrame(columns=empty_cols)

  valid_cols = [c for c in log_returns.columns if c not in excluded]
  records = []
  for code in valid_cols:
    m = compute_weekly_metrics_series(log_returns[code])
    if m is None:
      continue
    records.append({
      "Code": code,
      "return_52w": m["return_52w"],
      "return_6m": m["return_52w"],
      "sharpe_52w_ann": m["sharpe_52w_ann"],
      "sharpe_120d": m["sharpe_52w_ann"],
      "risk_6m": m["risk_weekly_std"],
      "risk_pct": m["risk_pct"],
      "mean_log_r_ann": m["mean_log_r_ann"],
      "mean_log_r_weekly": m["mean_log_r_weekly"],
    })

  if not records:
    return pd.DataFrame(columns=empty_cols)

  metrics = pd.DataFrame.from_records(records)
  metrics["risk_bucket"] = metrics["risk_pct"].apply(classify_risk)
  metrics = metrics.merge(etf_df, on="Code", how="left")
  out_cols = [c for c in empty_cols if c in metrics.columns]
  return metrics[out_cols].reset_index(drop=True)


def compute_delta3m(close: pd.DataFrame) -> pd.DataFrame:
  if close.empty:
    return pd.DataFrame(columns=["Code", "return_prev3m", "return_recent3m", "delta_3m"])

  end = close.index.max()
  start = end - pd.DateOffset(months=6)
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


def compute_sharpe_13w_ago(
  close: pd.DataFrame,
  weekly_window: int = 52,
) -> pd.Series:
  """
  13주 전 기준으로 과거 52주 sharpe를 계산한다.
  returns: Series(index=Code, value=sharpe_13w_ago)
  """
  if close.empty:
    return pd.Series(dtype=float)
  end_old = close.index.max() - pd.DateOffset(weeks=13)
  start_old = end_old - pd.DateOffset(weeks=weekly_window)
  close_old = close.loc[
    (close.index >= start_old) & (close.index <= end_old)
  ]
  if len(close_old) < 20:
    return pd.Series(dtype=float)

  weekly_old = close_old.resample("W-FRI").last().dropna(how="all")
  log_ret_old = np.log(weekly_old / weekly_old.shift(1)).dropna(how="all")

  scale = np.sqrt(52 / 12)
  result = {}
  for code in log_ret_old.columns:
    s = log_ret_old[code].dropna()
    if len(s) < 20:
      continue
    std = float(s.std())
    if std < 1e-9:
      continue
    mu = float(s.mean()) * 52
    sharpe = mu / (std * scale)
    result[code] = sharpe
  return pd.Series(result)


def select_best_by_bucket(metrics: pd.DataFrame) -> pd.DataFrame:
  if metrics.empty:
    return pd.DataFrame(columns=["Code", "Name", "return_6m", "risk_6m", "risk_pct", "risk_bucket"])
  filtered = metrics.dropna(subset=["risk_bucket"])
  if filtered.empty:
    return pd.DataFrame(columns=["Code", "Name", "return_6m", "risk_6m", "risk_pct", "risk_bucket"])
  best = filtered.sort_values(["risk_bucket", "return_6m"], ascending=[True, False])
  best = best.groupby("risk_bucket", as_index=False).head(1)
  return best.reset_index(drop=True)


def classify_asset_class(name: str) -> str:  # noqa: PLR0911
  """ETF 종목명으로 자산 클래스를 분류한다.

  우선순위:
    1. 커버드콜 전용 (기초자산이 채권이면 Bond, 아니면 Equity)
    2. CashLike  - 단기 현금성 (머니마켓, KOFR, CD금리, 단기채권 등)
    3. Alt       - 실물자산리츠상품선물 (골드선물, 리츠, 원유선물 등)
    4. Bond      - 채권국채회사채
    5. Equity    - 나머지 (주식)

  기존 단음절 키워드("콜","금","은","단기") 제거로 오분류 60건+ 수정:
    - 커버드콜 ETF 28개: CashLike  Equity/Bond
    - 금융주은행주  6개: Alt  Equity
    - SOFRCD금리  11개: Alt  CashLike
    - 구리농산물선물 등 8개: Equity  Alt
  """
  if not name:
    return "Equity"
  raw = str(name)
  lowered = raw.lower()

  #  1순위: 커버드콜 전용 규칙
  if "커버드콜" in raw:
    _bond_kw_lower = ["bond", "credit"]
    _bond_kw_raw   = ["채권", "국채", "회사채"]
    if any(k in lowered for k in _bond_kw_lower) or any(k in raw for k in _bond_kw_raw):
      return "Bond"
    return "Equity"

  #  2순위: CashLike
  cashlike_lower = ["mmf", "kofr", "koribor", "sofr"]
  cashlike_raw   = [
    "머니마켓", "단기통안채", "단기자금", "CD금리", "CD1년금리",
    "단기변동금리", "초단기채권", "초단기국채", "통안채",
  ]
  if any(k in lowered for k in cashlike_lower):
    return "CashLike"
  if any(k in raw for k in cashlike_raw):
    return "CashLike"
  if "단기채권" in raw and not any(x in raw for x in ["달러", "미국", "해외", "선진국"]):
    return "CashLike"

  #  3순위: Alt
  alt_lower = ["reit", "commodity", "commodities", "gold", "silver"]
  alt_raw   = [
    "리츠", "골드선물", "금선물", "금현물", "실물금", "금채굴", "금은선물",
    "은선물", "실물은", "원유선물", "원유에너지", "구리선물", "구리실물",
    "농산물선물", "3대농산물", "탄소배출권", "달러선물", "엔선물",
    "희토류", "원자재",
  ]
  if any(k in lowered for k in alt_lower):
    return "Alt"
  if any(k in raw for k in alt_raw):
    return "Alt"

  #  4순위: Bond
  bond_lower = ["bond", "credit"]
  bond_raw   = ["채권", "국채", "회사채", "국고채", "공채", "크레딧", "듀레이션"]
  if any(k in lowered for k in bond_lower):
    return "Bond"
  if any(k in raw for k in bond_raw):
    return "Bond"

  #  5순위: Equity
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
  logger.info("[P0-1] price range: %s ~ %s, codes=%s", start.date(), end.date(), len(codes))
  close = load_close_prices(codes, start, end)

  weekly_data_for_cache = build_weekly_price_matrix(
    close,
    window_weeks=int(CONFIG.get("weekly_window", 52)),
  )
  returns_tail = weekly_data_for_cache["log_returns"]
  metrics = compute_metrics(etf_df, close, returns_tail=returns_tail)
  recommendations = select_best_by_bucket(metrics)
  delta3m = compute_delta3m(close)
  sharpe_13w = compute_sharpe_13w_ago(close)

  cached_at = _now_kst()
  _CACHE["timestamp"] = time.time()
  _CACHE["cached_at"] = cached_at.isoformat(timespec="seconds")
  _CACHE["data_asof"] = weekly_data_for_cache["data_as_of"]
  _CACHE["is_week_partial"] = weekly_data_for_cache["is_current_week_partial"]
  _CACHE["weekly_excluded"] = weekly_data_for_cache["excluded"]
  _CACHE["weekly_last_label"] = weekly_data_for_cache.get("weekly_last_label")
  _CACHE["weekly_last_observed"] = weekly_data_for_cache.get("weekly_last_observed")
  _CACHE["metrics"] = metrics
  _CACHE["recommendations"] = recommendations
  _CACHE["delta3m"] = delta3m
  _CACHE["sharpe_13w"] = sharpe_13w
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
    "freq": "weekly",
    "return_unit": "simple_52w",
    "data_asof": _CACHE.get("data_asof"),
    "is_current_week_partial": _CACHE.get("is_week_partial", False),
    "weekly_last_label": _CACHE.get("weekly_last_label"),
    "weekly_last_observed": _CACHE.get("weekly_last_observed"),
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


def _infer_returns_frequency(returns_tail: pd.DataFrame | None) -> Tuple[str, int]:
  if returns_tail is None or returns_tail.empty:
    return "daily", 252
  index = returns_tail.index
  if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
    return "daily", 252
  deltas = index.to_series().diff().dropna()
  if deltas.empty:
    return "daily", 252
  try:
    median_days = float(deltas.dt.total_seconds().median() / 86400.0)
  except Exception:
    return "daily", 252
  if median_days >= 5.0:
    return "weekly", 52
  return "daily", 252


def _risk_scaling_policy(freq: str, periods_per_year: int) -> Dict[str, object]:
  freq_value = str(freq or "daily").lower()
  ppy = int(periods_per_year) if periods_per_year is not None else 252
  if ppy <= 0:
    ppy = 252
  if freq_value not in ("daily", "weekly"):
    freq_value = "weekly" if ppy <= 60 else "daily"
  window_days = int(CONFIG["window_days"])
  window_weeks = max(1, int(round(window_days / 5)))
  window_periods = window_days if freq_value == "daily" else window_weeks
  scale_to_monthly = float(np.sqrt(ppy / 12.0))
  return {
    "freq": freq_value,
    "periods_per_year": int(ppy),
    "window_periods": int(window_periods),
    "window_days": int(window_days),
    "window_weeks": int(window_weeks),
    "scale_to_monthly": float(scale_to_monthly),
  }


def _align_returns_intersection(
  returns_tail: pd.DataFrame | None,
  codes: List[str],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
  alignment_meta: Dict[str, object] = {
    "alignment_policy": "intersection",
    "alignment_n_dates": 0,
    "alignment_start": None,
    "alignment_end": None,
    "alignment_drop_pct": None,
  }
  if returns_tail is None or returns_tail.empty or not codes:
    return pd.DataFrame(), alignment_meta

  aligned = returns_tail.reindex(columns=codes)
  total_dates = int(len(aligned))
  aligned = aligned.dropna(how="any")
  if not aligned.empty:
    aligned = aligned[~aligned.index.duplicated(keep="last")].sort_index()

  n_dates = int(len(aligned))
  alignment_meta["alignment_n_dates"] = n_dates
  if total_dates > 0:
    dropped = max(total_dates - n_dates, 0)
    alignment_meta["alignment_drop_pct"] = float((dropped / total_dates) * 100.0)
  if n_dates > 0:
    start = aligned.index.min()
    end = aligned.index.max()
    alignment_meta["alignment_start"] = start.strftime("%Y-%m-%d") if hasattr(start, "strftime") else str(start)
    alignment_meta["alignment_end"] = end.strftime("%Y-%m-%d") if hasattr(end, "strftime") else str(end)
  return aligned, alignment_meta


def _estimate_feasible_min_risk_est(
  holdings_rows: List[Dict[str, object]],
  required_classes: List[str],
  min_weight: float,
  max_holdings: int,
) -> float | None:
  if not holdings_rows:
    return None
  risk_map: Dict[str, float] = {}
  class_rows: Dict[str, List[Dict[str, object]]] = {asset_class: [] for asset_class in ASSET_CLASSES}
  for row in holdings_rows:
    code = row.get("Code")
    asset_class = row.get("asset_class")
    risk_pct = _to_json_float(row.get("risk_pct"))
    if code is None or risk_pct is None or asset_class not in class_rows:
      continue
    risk_map[code] = float(risk_pct)
    class_rows[asset_class].append(row)
  if not risk_map:
    return None

  chosen_codes: List[str] = []
  for asset_class in required_classes:
    pool = class_rows.get(asset_class) or []
    if not pool:
      return None
    best = min(pool, key=lambda item: (_to_json_float(item.get("risk_pct")) or float("inf"), item.get("Code")))
    code = str(best["Code"])
    if code not in chosen_codes:
      chosen_codes.append(code)

  for row in sorted(holdings_rows, key=lambda item: (_to_json_float(item.get("risk_pct")) or float("inf"), item.get("Code"))):
    code = row.get("Code")
    if code is None or code in chosen_codes:
      continue
    if len(chosen_codes) >= int(max_holdings):
      break
    chosen_codes.append(str(code))

  if not chosen_codes:
    return None
  weights = {code: float(min_weight) for code in chosen_codes}
  remaining = 1.0 - float(min_weight) * len(chosen_codes)
  if remaining < -1e-9:
    return None

  while remaining > 1e-10:
    progressed = False
    for code in sorted(chosen_codes, key=lambda c: (risk_map.get(c, float("inf")), c)):
      row = next((item for item in holdings_rows if str(item.get("Code")) == code), None)
      if row is None:
        continue
      cap = 0.4
      room = cap - weights[code]
      if room <= 1e-10:
        continue
      add = min(room, remaining)
      weights[code] += add
      remaining -= add
      progressed = True
      if remaining <= 1e-10:
        break
    if not progressed:
      break

  total = sum(weights.values())
  if total <= 0:
    return None
  return float(sum((w / total) * risk_map.get(code, 0.0) for code, w in weights.items()))


def _build_candidate_pools(metrics: pd.DataFrame, config: Dict[str, object]) -> Dict[str, List[Dict[str, object]]]:
  df = metrics.dropna(subset=["Code", "Name", "return_6m", "risk_pct", "sharpe_120d"]).copy()
  if df.empty:
    return {}
  df["asset_class"] = df["Name"].apply(classify_asset_class)
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
    selected = class_df.sort_values(
      ["sharpe_120d", "return_6m", "risk_pct", "Code"],
      ascending=[False, False, True, True],
    ).head(topn)
    pools[asset_class] = [
      {
        "Code": row["Code"],
        "Name": row["Name"],
        "return_52w": float(row.get("return_52w") or row["return_6m"]),
        "return_6m": float(row["return_6m"]),
        "risk_pct": float(row["risk_pct"]),
        "sharpe_window": float(row["sharpe_120d"]),
        "mean_log_r_ann": float(row.get("mean_log_r_ann") or 0.0),
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


def _quantize_5pct(
  w_cont: np.ndarray,
  ub_arr: np.ndarray,
  step: float = 0.05,
) -> np.ndarray | None:
  """
  연속 비중 -> 5% 단위 이산화 (Largest Remainder Method).

  규칙:
    - 각 비중은 {0, 0.05, 0.10, ..., ub_arr[i]} 중 하나
    - 비중 > 0이면 반드시 >= 0.05 (최소 5% 룰)
    - sum(w_q) == 1.0 정확히
    - ub_arr[i] 초과 불가

  실패 시 None 반환 (합이 1이 되지 않는 경우).
  """
  w = np.array(w_cont, dtype=float)
  ub = np.array(ub_arr, dtype=float)
  n = len(w)
  if n == 0:
    return None

  # Step 1: w < 0.05인 자산은 0으로 처리 (최소 룰)
  w = np.where(w < step - 1e-9, 0.0, w)

  # Step 2: floor to step
  floor_w = np.floor(w / step) * step
  floor_w = np.minimum(floor_w, np.floor(ub / step) * step)

  # Step 3: 나머지 계산
  remainder = w - floor_w

  # Step 4: 분배할 슬롯 수
  total_floor = float(np.sum(floor_w))
  slots = int(round((1.0 - total_floor) / step))
  if slots < 0:
    return None

  # Step 5: remainder 내림차순으로 슬롯 배분
  # 동률은 index 오름차순 (결정론적)
  order = sorted(range(n), key=lambda i: (-remainder[i], i))
  q = floor_w.copy()
  assigned = 0
  for idx in order:
    if assigned >= slots:
      break
    if q[idx] + step <= ub[idx] + 1e-9:
      q[idx] += step
      assigned += 1

  # 슬롯 미소진 시 (상한 제약으로): 다음 후보 재탐색
  if assigned < slots:
    for idx in order:
      if assigned >= slots:
        break
      if q[idx] + step <= ub[idx] + 1e-9:
        q[idx] += step
        assigned += 1

  # Step 6: 최소 룰 재검사 (0 < q[i] < 0.05 제거)
  q = np.where((q > 1e-9) & (q < step - 1e-9), 0.0, q)

  # Step 7: 합 검증
  total = float(np.sum(q))
  if abs(total - 1.0) > 1e-6:
    return None

  return q


def _swap_repair_band(
  w_q: np.ndarray,
  sigma: np.ndarray,
  ub_arr: np.ndarray,
  lo_pct: float,
  hi_pct: float,
  scale_to_monthly: float,
  max_steps: int = 400,
  step: float = 0.05,
) -> Tuple[np.ndarray, Dict[str, object]]:
  """
  이산 비중 w_q를 수정하여 lo_pct <= risk_pct <= hi_pct 달성.

  스왑: w_q[i] -= step, w_q[j] += step (sum=1 유지)
  방향: risk < lo -> 저분산->고분산 이동 / risk > hi -> 반대
  결정론: sigma 대각값(sigma[i,i]) 기준 정렬

  반환: (w_final, diag)
  diag: {repair_steps, repair_success, achieved_risk_pct}
  """
  n = len(w_q)
  w = w_q.copy()

  def _risk(ww):
    var = float(ww @ sigma @ ww)
    return float(np.sqrt(max(var, 0.0)) * scale_to_monthly * 100.0)

  def _valid(ww, i, j):
    # i에서 빼고 j에 더한 후 제약 확인
    if ww[i] - step < -1e-9:
      return False
    # 0 < ww[i]-step < 0.05 이면 최소 룰 위반
    new_i = ww[i] - step
    if new_i > 1e-9 and new_i < step - 1e-9:
      return False
    new_j = ww[j] + step
    if new_j > ub_arr[j] + 1e-9:
      return False
    return True

  diag_steps = 0
  for step_idx in range(max_steps):
    current_risk = _risk(w)
    if lo_pct <= current_risk <= hi_pct:
      break
    diag_steps = step_idx + 1

    # 리스크를 높여야 하는 경우: 저분산 자산(i)->고분산 자산(j)
    if current_risk < lo_pct:
      # i: sigma[i,i] 오름차순, j: sigma[j,j] 내림차순
      i_order = sorted(range(n), key=lambda k: (sigma[k, k], k))
      j_order = sorted(range(n), key=lambda k: (-sigma[k, k], k))
    else:
      # 리스크를 낮춰야 하는 경우: 고분산->저분산
      i_order = sorted(range(n), key=lambda k: (-sigma[k, k], k))
      j_order = sorted(range(n), key=lambda k: (sigma[k, k], k))

    moved = False
    for i in i_order:
      for j in j_order:
        if i == j:
          continue
        if _valid(w, i, j):
          w[i] -= step
          w[j] += step
          moved = True
          break
      if moved:
        break
    if not moved:
      break

  final_risk = _risk(w)
  return w, {
    "repair_steps": diag_steps,
    "repair_success": bool(lo_pct <= final_risk <= hi_pct),
    "achieved_risk_pct": float(final_risk),
  }


def _compute_final_hash(w_q: np.ndarray, asset_codes: List[str]) -> str:
  """
  포트폴리오 고유 해시 (버킷 간 중복 감지용).
  holdings code:weight 쌍을 정렬 후 MD5.
  """
  parts = sorted(
    f"{code}:{round(float(weight), 4):.4f}"
    for code, weight in zip(asset_codes, w_q)
    if float(weight) > 1e-9
  )
  return hashlib.md5("|".join(parts).encode()).hexdigest()


def _to_json_float(value: object) -> float | None:
  try:
    v = float(value)
  except Exception:
    return None
  if not np.isfinite(v):
    return None
  return float(v)


def _to_json_bool(value: object) -> bool:
  try:
    return bool(value)
  except Exception:
    return False


def _fmt_float(value: object, digits: int = 6) -> str:
  v = _to_json_float(value)
  if v is None:
    return "na"
  return f"{v:.{digits}f}"


def _qp_fail_reason_from_status(status: str | None) -> str:
  if status is None:
    return "no_solution"
  status_value = str(status).lower()
  if status_value in ("infeasible", "infeasible_inaccurate"):
    return "infeasible"
  if status_value in ("unbounded", "unbounded_inaccurate"):
    return "unbounded"
  if status_value in ("optimal", "optimal_inaccurate"):
    return "no_solution"
  return "unknown_status"


def _qp_is_dcp_error(message: str) -> bool:
  msg = (message or "").lower()
  return "dcp" in msg or "not dcp" in msg


def _qp_is_psd_issue(message: str) -> bool:
  msg = (message or "").lower()
  return "psd" in msg or "positive semidefinite" in msg or "quad_form" in msg


def _qp_default_solvers(cp_module: Any) -> List[str]:
  order = ["ECOS", "OSQP", "SCS", "CLARABEL"]
  try:
    installed = set(cp_module.installed_solvers())
  except Exception:
    installed = set()
  if not installed:
    return ["ECOS", "OSQP", "SCS"]
  return [solver_name for solver_name in order if solver_name in installed]


def _qp_solve_weights(
  mu: np.ndarray,
  sigma: np.ndarray,
  gamma: float,
  lower_bounds: np.ndarray | None = None,
  upper_bounds: np.ndarray | None = None,
  solver: str | None = None,
  risk_cap_pct: float | None = None,
  scale_to_monthly: float | None = None,
) -> Tuple[np.ndarray | None, Dict[str, object]]:
  diag: Dict[str, object] = {
    "solver_requested": solver,
    "solver_mode": "explicit" if solver else "default_fallback",
    "chosen_solver": None,
    "prob.status": None,
    "prob.value": None,
    "exception": None,
    "solver_attempts": [],
    "fail_reason": None,
  }
  try:
    import cvxpy as cp
  except Exception as exc:
    diag["exception"] = str(exc)
    diag["fail_reason"] = "exception"
    return None, diag

  mu_arr = np.array(mu, dtype=float).flatten()
  sigma_arr = np.array(sigma, dtype=float)
  n = int(len(mu_arr))
  diag["n"] = n
  diag["gamma"] = _to_json_float(gamma)
  diag["has_upper_bounds"] = _to_json_bool(upper_bounds is not None)
  diag["has_lower_bounds"] = _to_json_bool(lower_bounds is not None)
  if lower_bounds is not None:
    lower_arr = np.array(lower_bounds, dtype=float).flatten()
  else:
    lower_arr = None
  if upper_bounds is not None:
    upper_arr = np.array(upper_bounds, dtype=float).flatten()
  else:
    upper_arr = None
  diag["sum_lower_bounds"] = _to_json_float(np.sum(lower_arr)) if lower_arr is not None and lower_arr.size else None
  diag["sum_upper_bounds"] = _to_json_float(np.sum(upper_arr)) if upper_arr is not None and upper_arr.size else None
  diag["risk_cap_pct"] = _to_json_float(risk_cap_pct)
  diag["scale_to_monthly"] = _to_json_float(scale_to_monthly)
  diag["mu_min"] = _to_json_float(np.min(mu_arr)) if mu_arr.size else None
  diag["mu_max"] = _to_json_float(np.max(mu_arr)) if mu_arr.size else None
  sigma_diag = np.diag(sigma_arr) if sigma_arr.ndim == 2 and sigma_arr.shape[0] > 0 else np.array([], dtype=float)
  diag["sigma_diag_min"] = _to_json_float(np.min(sigma_diag)) if sigma_diag.size else None
  diag["sigma_diag_max"] = _to_json_float(np.max(sigma_diag)) if sigma_diag.size else None
  diag["mu_has_nan"] = bool(np.isnan(mu_arr).any()) if mu_arr.size else False
  diag["mu_has_inf"] = bool(np.isinf(mu_arr).any()) if mu_arr.size else False
  diag["sigma_has_nan"] = bool(np.isnan(sigma_arr).any()) if sigma_arr.size else False
  diag["sigma_has_inf"] = bool(np.isinf(sigma_arr).any()) if sigma_arr.size else False
  diag["upper_bounds_has_nan"] = bool(np.isnan(upper_arr).any()) if upper_arr is not None and upper_arr.size else False
  diag["upper_bounds_has_inf"] = bool(np.isinf(upper_arr).any()) if upper_arr is not None and upper_arr.size else False
  diag["lower_bounds_has_nan"] = bool(np.isnan(lower_arr).any()) if lower_arr is not None and lower_arr.size else False
  diag["lower_bounds_has_inf"] = bool(np.isinf(lower_arr).any()) if lower_arr is not None and lower_arr.size else False

  if n == 0:
    diag["fail_reason"] = "no_solution"
    return None, diag
  if sigma_arr.ndim != 2 or sigma_arr.shape[0] != n or sigma_arr.shape[1] != n:
    diag["fail_reason"] = "shape_error"
    return None, diag
  if upper_arr is not None and upper_arr.size != n:
    diag["fail_reason"] = "shape_error"
    return None, diag
  if lower_arr is not None and lower_arr.size != n:
    diag["fail_reason"] = "shape_error"
    return None, diag
  if (
    diag["mu_has_nan"]
    or diag["mu_has_inf"]
    or diag["sigma_has_nan"]
    or diag["sigma_has_inf"]
    or diag["lower_bounds_has_nan"]
    or diag["lower_bounds_has_inf"]
    or diag["upper_bounds_has_nan"]
    or diag["upper_bounds_has_inf"]
  ):
    diag["fail_reason"] = "nan_inf"
    return None, diag

  sigma_work = (sigma_arr + sigma_arr.T) / 2
  diag["sigma_symmetrized"] = True
  jitter = 0.0
  try:
    eigvals = np.linalg.eigvalsh(sigma_work)
    min_eig = float(np.min(eigvals)) if eigvals.size else 0.0
    diag["sigma_min_eig"] = _to_json_float(min_eig)
    if min_eig < -1e-10:
      jitter = float(abs(min_eig) + 1e-10)
      sigma_work = sigma_work + np.eye(n) * jitter
  except Exception as exc:
    diag["sigma_min_eig"] = None
    diag["sigma_eig_error"] = str(exc)
  diag["sigma_jitter"] = _to_json_float(jitter)
  sigma_qp = cp.psd_wrap(sigma_work)
  diag["sigma_psd_wrapped"] = True

  if solver:
    solvers_to_try = [str(solver)]
  else:
    solvers_to_try = _qp_default_solvers(cp)
  diag["solvers_to_try"] = list(solvers_to_try)
  if not solvers_to_try:
    diag["fail_reason"] = "no_solver"
    return None, diag

  last_fail_reason: str | None = None
  for solver_name in solvers_to_try:
    attempt: Dict[str, object] = {
      "solver_name": solver_name,
      "status": None,
      "fail_reason": None,
      "prob_value": None,
      "achieved_variance": None,
      "achieved_risk_pct": None,
    }
    w = cp.Variable(n, nonneg=True)
    objective = cp.Maximize(mu_arr @ w - float(gamma) * cp.quad_form(w, sigma_qp))
    constraints = [cp.sum(w) == 1]
    if lower_arr is not None:
      constraints.append(w >= lower_arr)
    if upper_arr is not None:
      constraints.append(w <= upper_arr)
    if risk_cap_pct is not None and scale_to_monthly is not None and scale_to_monthly > 0:
      cap_variance = float(((float(risk_cap_pct) / 100.0) / float(scale_to_monthly)) ** 2)
      constraints.append(cp.quad_form(w, sigma_qp) <= cap_variance)
      attempt["cap_variance"] = _to_json_float(cap_variance)
    prob = cp.Problem(objective, constraints)
    try:
      prob.solve(solver=solver_name, warm_start=True)
    except Exception as exc:
      message = str(exc)
      attempt["exception"] = message
      attempt["fail_reason"] = "dcp_error" if _qp_is_dcp_error(message) else ("psd_issue" if _qp_is_psd_issue(message) else "exception")
      diag["exception"] = message
      diag["solver_attempts"].append(attempt)
      last_fail_reason = str(attempt["fail_reason"])
      continue
    status = str(prob.status) if prob.status is not None else None
    attempt["status"] = status
    attempt["prob_value"] = _to_json_float(prob.value)
    diag["prob.status"] = status
    diag["prob.value"] = _to_json_float(prob.value)
    if w.value is None:
      attempt["fail_reason"] = _qp_fail_reason_from_status(status)
      diag["solver_attempts"].append(attempt)
      last_fail_reason = str(attempt["fail_reason"])
      continue
    weights = np.array(w.value, dtype=float).flatten()
    if not np.isfinite(weights).all():
      attempt["fail_reason"] = "nan_inf"
      diag["solver_attempts"].append(attempt)
      last_fail_reason = str(attempt["fail_reason"])
      continue
    try:
      if (
        sigma_work is not None
        and isinstance(sigma_work, np.ndarray)
        and sigma_work.ndim == 2
        and sigma_work.shape[0] == sigma_work.shape[1]
        and weights.size == sigma_work.shape[0]
      ):
        achieved_variance = float(weights.T @ (sigma_work @ weights))
        attempt["achieved_variance"] = _to_json_float(achieved_variance)
        achieved_risk_pct = np.sqrt(max(0.0, achieved_variance)) * float(scale_to_monthly) * 100.0
        attempt["achieved_risk_pct"] = _to_json_float(achieved_risk_pct)
    except Exception:
      attempt["achieved_variance"] = None
      attempt["achieved_risk_pct"] = None
    diag["chosen_solver"] = solver_name
    diag["solver_attempts"].append(attempt)
    diag["fail_reason"] = None
    return weights, diag

  diag["fail_reason"] = last_fail_reason or _qp_fail_reason_from_status(diag.get("prob.status")) or "no_solution"
  return None, diag


def _portfolio_risk_pct(
  weights: np.ndarray,
  sigma: np.ndarray,
  scale_to_monthly: float | None = None,
) -> float | None:
  if weights.size == 0:
    return None
  variance = float(weights.T @ sigma @ weights)
  if variance < 0:
    return None
  period_vol = np.sqrt(variance)
  if not np.isfinite(period_vol):
    return None
  if scale_to_monthly is None or scale_to_monthly <= 0:
    scale_to_monthly = float(np.sqrt(252 / 12.0))
  monthly_vol = period_vol * float(scale_to_monthly)
  return float(monthly_vol * 100.0)


def _project_weights_with_bounds(
  target: np.ndarray,
  lower: np.ndarray,
  upper: np.ndarray,
  max_iter: int = 300,
) -> np.ndarray | None:
  target = np.array(target, dtype=float)
  lower = np.array(lower, dtype=float)
  upper = np.array(upper, dtype=float)
  if target.size == 0 or lower.size != target.size or upper.size != target.size:
    return None
  if float(np.sum(lower)) > 1.0 + 1e-9:
    return None
  if float(np.sum(upper)) < 1.0 - 1e-9:
    return None
  w = np.minimum(np.maximum(target, lower), upper)
  for _ in range(max_iter):
    diff = 1.0 - float(np.sum(w))
    if abs(diff) <= 1e-10:
      break
    if diff > 0:
      room = upper - w
      free = room > 1e-12
      capacity = float(np.sum(room[free]))
      if capacity <= 1e-12:
        return None
      w[free] += diff * (room[free] / capacity)
    else:
      room = w - lower
      free = room > 1e-12
      capacity = float(np.sum(room[free]))
      if capacity <= 1e-12:
        return None
      w[free] += diff * (room[free] / capacity)
    w = np.minimum(np.maximum(w, lower), upper)
  total = float(np.sum(w))
  if total <= 0:
    return None
  w = w / total
  if np.any(w < (lower - 1e-6)) or np.any(w > (upper + 1e-6)):
    return None
  return w


def _compute_portfolio_sharpe_from_returns(
  returns_tail: pd.DataFrame,
  codes: List[str],
  weights: np.ndarray,
  periods_per_year: int = 52,
) -> float | None:
  if returns_tail is None or returns_tail.empty or not codes:
    return None
  returns_slice = returns_tail.reindex(columns=codes).dropna(how="any")
  if returns_slice.empty:
    return None
  rp = returns_slice.mul(weights, axis=1).sum(axis=1)
  if rp.empty:
    return None
  std = rp.std()
  if std is None or np.isnan(std) or std == 0:
    return None
  ppy = int(periods_per_year) if periods_per_year and periods_per_year > 0 else 52
  annualizer = float(np.sqrt(ppy))
  rf_per_period = _RF_ANNUAL / float(ppy)
  return float(((rp.mean() - rf_per_period) / std) * annualizer)


def _select_qp_holdings_topk(
  holdings_rows: List[Dict[str, object]],
  raw_weights: np.ndarray,
  required_classes: List[str],
  max_holdings: int,
  min_weight: float,
) -> Tuple[List[int], bool]:
  idx_sorted = sorted(
    range(len(holdings_rows)),
    key=lambda i: (-float(raw_weights[i]), -float(holdings_rows[i].get("sharpe_window", 0.0)), holdings_rows[i]["Code"]),
  )
  if not idx_sorted:
    return [], False
  selected: List[int] = []
  for asset_class in required_classes:
    candidate_idx = next((i for i in idx_sorted if holdings_rows[i]["asset_class"] == asset_class), None)
    if candidate_idx is not None and candidate_idx not in selected:
      selected.append(candidate_idx)
  for idx in idx_sorted:
    if len(selected) >= max_holdings:
      break
    if idx in selected:
      continue
    selected.append(idx)
  if not selected:
    return [], False
  if len(selected) > max_holdings:
    selected = selected[:max_holdings]
  # Policy point: if required classes exceed display capacity, we keep the best max_holdings assets and mark constraints unmet in meta/debug.
  truncated = len([i for i in idx_sorted if float(raw_weights[i]) > 1e-6]) > max_holdings
  return selected, truncated


def _soft_select_after_quantize(
  holdings_rows: List[Dict[str, object]],
  selected_rows: List[Dict[str, object]],
  selected_weights: np.ndarray,
  best_weight_by_code: Dict[str, float],
  min_weight: float,
  primary_count: int = 5,
  fallback_count: int = 4,
) -> Tuple[List[Dict[str, object]], np.ndarray, int, int] | None:
  if not selected_rows or len(selected_rows) != len(selected_weights):
    return None
  code_to_row = {str(row["Code"]): row for row in holdings_rows}
  rank_rows = sorted(
    holdings_rows,
    key=lambda row: (
      -float(row.get("qp_score", 0.0)),
      -float(best_weight_by_code.get(str(row["Code"]), 0.0)),
      str(row["Code"]),
    ),
  )
  base_weight_by_code = {
    str(row["Code"]): float(weight)
    for row, weight in zip(selected_rows, selected_weights)
    if float(weight) > 1e-12
  }
  if not base_weight_by_code:
    return None

  def _try_target(target_count: int) -> Tuple[List[Dict[str, object]], np.ndarray, int] | None:
    ranked_by_weight = sorted(
      base_weight_by_code.items(),
      key=lambda kv: (-float(kv[1]), -float(code_to_row[kv[0]].get("qp_score", 0.0)), kv[0]),
    )
    if len(ranked_by_weight) < target_count:
      return None

    chosen_codes = [code for code, _ in ranked_by_weight[:target_count]]
    chosen_set = set(chosen_codes)
    swap_count = 0

    for _ in range(24):
      class_map: Dict[str, List[str]] = {}
      for code in chosen_codes:
        asset_class = str(code_to_row[code].get("asset_class", "Equity"))
        class_map.setdefault(asset_class, []).append(code)
      offender = next((ac for ac, codes in class_map.items() if len(codes) >= 3), None)
      if offender is None:
        break

      offender_codes = class_map[offender]
      drop_code = sorted(
        offender_codes,
        key=lambda c: (
          float(code_to_row[c].get("qp_score", 0.0)),
          float(base_weight_by_code.get(c, 0.0)),
          c,
        ),
      )[0]

      replacement_code = None
      for candidate in rank_rows:
        cand_code = str(candidate["Code"])
        if cand_code in chosen_set:
          continue
        cand_class = str(candidate.get("asset_class", "Equity"))
        if len(class_map.get(cand_class, [])) >= 2:
          continue
        replacement_code = cand_code
        break
      if replacement_code is None:
        return None

      chosen_set.remove(drop_code)
      chosen_set.add(replacement_code)
      chosen_codes = [replacement_code if c == drop_code else c for c in chosen_codes]
      if replacement_code not in base_weight_by_code:
        base_weight_by_code[replacement_code] = max(float(best_weight_by_code.get(replacement_code, 0.0)), min_weight)
      swap_count += 1

    weight_seed = np.array(
      [max(float(base_weight_by_code.get(code, 0.0)), 0.0) for code in chosen_codes],
      dtype=float,
    )
    weight_seed = _normalize_weights(weight_seed)
    lower_bounds = np.full(target_count, float(min_weight), dtype=float)
    upper_bounds = np.full(target_count, 0.4, dtype=float)
    projected = _project_weights_with_bounds(weight_seed, lower_bounds, upper_bounds)
    if projected is None:
      return None
    snapped = _quantize_5pct(projected, upper_bounds, step=0.05)
    if snapped is None:
      return None
    snapped = _normalize_weights(snapped)
    out_rows = [code_to_row[code] for code in chosen_codes]
    return out_rows, snapped, swap_count

  primary = _try_target(primary_count)
  if primary is not None:
    rows_out, weights_out, swap_count = primary
    return rows_out, weights_out, primary_count, swap_count
  fallback = _try_target(fallback_count)
  if fallback is None:
    return None
  rows_out, weights_out, swap_count = fallback
  return rows_out, weights_out, fallback_count, swap_count


def _portfolio_adequacy_meta(
  rows: List[Dict[str, object]],
  weights: np.ndarray,
  portfolio_risk_pct: float | None,
) -> Dict[str, object]:
  weights_arr = [float(w) for w in np.array(weights, dtype=float).tolist() if float(w) > 1e-12]
  herfindahl = float(sum(w * w for w in weights_arr)) if weights_arr else None
  n_assets = int(len(weights_arr))
  risk_weighted = 0.0
  for row, w in zip(rows, weights):
    risk_weighted += float(w) * float(row.get("risk_pct") or 0.0)
  if portfolio_risk_pct is not None and portfolio_risk_pct > 1e-9:
    diversification_ratio = float(risk_weighted / float(portfolio_risk_pct))
  else:
    diversification_ratio = None
  return {
    "herfindahl": herfindahl,
    "n_assets": n_assets,
    "diversification_ratio": diversification_ratio,
  }


def build_portfolio_qp(
  metrics: pd.DataFrame,
  config: Dict[str, object],
  returns_tail: pd.DataFrame | None,
  score_mode: str,
  mode: str = "base",
  debug: bool = False,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
  try:
    import cvxpy
    solver_available = True
  except Exception:
    cvxpy = None
    solver_available = False

  topn_config = config.get("topN_by_class", 20)
  inferred_freq, inferred_ppy = _infer_returns_frequency(returns_tail)
  scaling_policy = _risk_scaling_policy(inferred_freq, inferred_ppy)
  try:
    cov_shrink_alpha = float(config.get("cov_shrink_alpha", 0.0))
  except Exception:
    cov_shrink_alpha = 0.0
  meta = {
    "version": "qp-v3",
    "mode": mode,
    "topN_by_class": topn_config,
    "freq": scaling_policy["freq"],
    "window_periods": scaling_policy["window_periods"],
    "window_days": scaling_policy["window_days"],
    "window_weeks": scaling_policy["window_weeks"],
    "scale_to_monthly": scaling_policy["scale_to_monthly"],
    "cov_shrink_alpha": cov_shrink_alpha,
    "solver_available": solver_available,
    "score_mode": "delta" if mode == "delta" else "sharpe",
    "sharpe_window": scaling_policy["window_periods"],
    "alignment_policy": "intersection",
    "alignment_n_dates": 0,
    "alignment_start": None,
    "alignment_end": None,
    "alignment_drop_pct": None,
  }

  pools = _build_candidate_pools(metrics, config)
  by_class_counts = {asset_class: len(pools.get(asset_class, [])) for asset_class in ASSET_CLASSES}
  required_classes: List[str] = []
  relaxed_classes: List[str] = []
  feasible_before = 1
  meta["universe_counts"] = by_class_counts

  holdings_rows: List[Dict[str, object]] = []
  used_codes: set[str] = set()
  for asset_class in ASSET_CLASSES:
    for row in pools.get(asset_class, []):
      code = row["Code"]
      if code in used_codes:
        continue
      holdings_rows.append(dict(row))
      used_codes.add(code)

  qp_audit = None
  if debug:
    qp_audit = {
      "freq": meta["freq"],
      "window_periods": int(meta["window_periods"]),
      "scale_to_monthly": float(meta["scale_to_monthly"]),
      "cov_shrink_alpha": float(meta["cov_shrink_alpha"]),
      "S0_universe": int(len(metrics.dropna(subset=["Code", "Name"]))),
      "buckets": {},
    }
    for bucket in _bucket_bounds():
      qp_audit["buckets"][bucket["label"]] = {
        "S1_candidate_topN": {
          "by_class": dict(by_class_counts),
          "total": int(len(holdings_rows)),
        },
        "S2_metrics_non_nan": int(len(holdings_rows)),
        "S3_asset_class": {
          "known": int(sum(by_class_counts.values())),
          "unknown": 0,
          "by_class": dict(by_class_counts),
        },
        "S4_class_min_feasible": int(feasible_before),
        "S5_portfolios_produced": 0,
        "solver_failure_reason": None,
        "solver_diag": None,
        "solver_attempts": [],
        # P1 추가 필드
        "band_lo": float(bucket["min"]),
        "band_hi": _to_json_float(bucket.get("max")),
        "achieved_risk_pct": None,
        "feasible_in_band": None,
        "fallback_used": None,
        "fallback_reason": "",
        "quantize_method": None,
        "repair_steps": None,
        "final_hash": None,
        "duplicate_warning": None,
      }

  items: List[Dict[str, object]] = []
  max_holdings = min(int(CONFIG["portfolio_max_holdings"]), 10)
  min_weight = 0.10
  feasible_min_risk_est = _estimate_feasible_min_risk_est(
    holdings_rows,
    required_classes,
    min_weight=min_weight,
    max_holdings=max_holdings,
  )

  if not solver_available:
    for bucket in _bucket_bounds():
      target_risk = _bucket_target(bucket)
      constraints_meta = {
        "required_classes": required_classes,
        "relaxed_classes": relaxed_classes,
        "feasible_before": feasible_before,
      }
      if debug:
        constraints_meta.update({
          "constraints_met": False,
          "missing_required_classes": list(required_classes),
          "holdings_display_limit": int(max_holdings),
          "holdings_display_truncated": False,
        })
        qp_audit["buckets"][bucket["label"]]["solver_failure_reason"] = "missing_solver"
      items.append({
        "risk_bucket": bucket["label"],
        "risk_pct": None,
        "return_6m": None,
        "holdings": [],
        "error": "missing_solver",
        "meta": {
          "target_risk": target_risk,
          "feasible_min_risk_est": feasible_min_risk_est,
          "reason_category": "solver_unavailable",
          "constraints": constraints_meta,
        },
      })
    if qp_audit is not None:
      meta["qp_audit"] = qp_audit
    return items, meta

  if returns_tail is None or returns_tail.empty:
    for bucket in _bucket_bounds():
      target_risk = _bucket_target(bucket)
      constraints_meta = {
        "required_classes": required_classes,
        "relaxed_classes": relaxed_classes,
        "feasible_before": feasible_before,
      }
      if debug:
        constraints_meta.update({
          "constraints_met": False,
          "missing_required_classes": list(required_classes),
          "holdings_display_limit": int(max_holdings),
          "holdings_display_truncated": False,
        })
        qp_audit["buckets"][bucket["label"]]["solver_failure_reason"] = "missing_covariance"
      items.append({
        "risk_bucket": bucket["label"],
        "risk_pct": None,
        "return_6m": None,
        "holdings": [],
        "error": "missing_covariance",
        "meta": {
          "target_risk": target_risk,
          "feasible_min_risk_est": feasible_min_risk_est,
          "reason_category": "data_missing",
          "constraints": constraints_meta,
        },
      })
    if qp_audit is not None:
      meta["qp_audit"] = qp_audit
    return items, meta

  if not holdings_rows:
    for bucket in _bucket_bounds():
      target_risk = _bucket_target(bucket)
      constraints_meta = {
        "required_classes": required_classes,
        "relaxed_classes": relaxed_classes,
        "feasible_before": feasible_before,
      }
      if debug:
        constraints_meta.update({
          "constraints_met": True,
          "missing_required_classes": [],
          "holdings_display_limit": int(max_holdings),
          "holdings_display_truncated": False,
        })
        qp_audit["buckets"][bucket["label"]]["solver_failure_reason"] = "insufficient_candidates"
      items.append({
        "risk_bucket": bucket["label"],
        "risk_pct": None,
        "return_6m": None,
        "holdings": [],
        "error": "insufficient_candidates",
        "meta": {
          "target_risk": target_risk,
          "feasible_min_risk_est": feasible_min_risk_est,
          "reason_category": "insufficient_candidates",
          "constraints": constraints_meta,
        },
      })
    if qp_audit is not None:
      meta["qp_audit"] = qp_audit
    return items, meta

  codes = [h["Code"] for h in holdings_rows]
  returns_slice_all, alignment_meta = _align_returns_intersection(returns_tail, codes)
  meta.update(alignment_meta)
  if returns_slice_all.empty:
    for bucket in _bucket_bounds():
      target_risk = _bucket_target(bucket)
      constraints_meta = {
        "required_classes": required_classes,
        "relaxed_classes": relaxed_classes,
        "feasible_before": feasible_before,
      }
      if debug:
        constraints_meta.update({
          "constraints_met": False,
          "missing_required_classes": list(required_classes),
          "holdings_display_limit": int(max_holdings),
          "holdings_display_truncated": False,
        })
        qp_audit["buckets"][bucket["label"]]["solver_failure_reason"] = "missing_covariance"
      items.append({
        "risk_bucket": bucket["label"],
        "risk_pct": None,
        "return_6m": None,
        "holdings": [],
        "error": "missing_covariance",
        "meta": {
          "target_risk": target_risk,
          "feasible_min_risk_est": feasible_min_risk_est,
          "reason_category": "alignment_empty",
          "constraints": constraints_meta,
        },
      })
    if qp_audit is not None:
      meta["qp_audit"] = qp_audit
    return items, meta

  # P0-2: sigma = 주봉 로그 수익률 공분산
  sigma = returns_slice_all.cov().values
  sigma = (sigma + sigma.T) / 2
  if not np.isfinite(sigma).all():
    for bucket in _bucket_bounds():
      target_risk = _bucket_target(bucket)
      constraints_meta = {
        "required_classes": required_classes,
        "relaxed_classes": relaxed_classes,
        "feasible_before": feasible_before,
      }
      if debug:
        constraints_meta.update({
          "constraints_met": False,
          "missing_required_classes": list(required_classes),
          "holdings_display_limit": int(max_holdings),
          "holdings_display_truncated": False,
        })
        qp_audit["buckets"][bucket["label"]]["solver_failure_reason"] = "nan_inf"
      items.append({
        "risk_bucket": bucket["label"],
        "risk_pct": None,
        "return_6m": None,
        "holdings": [],
        "error": "nan_inf",
        "meta": {
          "target_risk": target_risk,
          "feasible_min_risk_est": feasible_min_risk_est,
          "reason_category": "numerical",
          "constraints": constraints_meta,
        },
      })
    if qp_audit is not None:
      meta["qp_audit"] = qp_audit
    return items, meta

  sharpe_13w_cache = _CACHE.get("sharpe_13w")
  if isinstance(sharpe_13w_cache, pd.Series):
    sharpe_13w = sharpe_13w_cache.to_dict()
  elif isinstance(sharpe_13w_cache, dict):
    sharpe_13w = sharpe_13w_cache
  else:
    sharpe_13w = {}
  if mode == "delta":
    for row in holdings_rows:
      code = str(row.get("Code") or "")
      s_now = float(row.get("sharpe_window") or 0.0)
      s_old = float(sharpe_13w.get(code, s_now))
      delta = s_now - s_old
      risk_floor = max(float(row.get("risk_pct") or 0.0), 2.0)
      row["qp_score"] = s_now + 0.5 * (delta / risk_floor)
    mu = np.array([float(h.get("qp_score") or 0.0) for h in holdings_rows], dtype=float)
  else:
    for row in holdings_rows:
      row["qp_score"] = float(row.get("mean_log_r_ann") or 0.0)
    mu = np.array([float(h.get("mean_log_r_ann") or 0.0) for h in holdings_rows], dtype=float)
  upper_bounds = np.full(len(holdings_rows), 0.4, dtype=float)
  default_solvers = _qp_default_solvers(cvxpy) if cvxpy is not None else ["ECOS", "OSQP", "SCS"]

  # 버킷별 gamma 테이블 (고위험 버킷 = 낮은 gamma)
  _BUCKET_GAMMA: Dict[str, float] = {
    "0-3%": 8.0,
    "3-6%": 4.0,
    "6-9%": 2.0,
    "9-12%": 1.0,
    "12-15%": 0.5,
  }

  # 버킷 해시 수집 (중복 감지용)
  bucket_hashes: Dict[str, str] = {}

  for bucket in _bucket_bounds():
    bucket_label = bucket["label"]
    bucket_lo = float(bucket["min"])
    bucket_hi_v = _to_json_float(bucket.get("max"))  # None이면 상한 없음
    target_risk = _bucket_target(bucket)
    gamma = _BUCKET_GAMMA.get(bucket_label, 1.0)

    constraints_meta = {
      "required_classes": required_classes,
      "relaxed_classes": relaxed_classes,
      "feasible_before": feasible_before,
    }

    # STEP 1: QP 풀기 (연속 완화)
    best_weights = None
    best_risk = None
    best_gamma = gamma
    solver_failure_reason = None
    best_solver_diag = None
    bucket_solver_attempts: List[List[object]] = []
    last_status = None
    chosen_solver = None

    if debug:
      logger.info(
        "[QP] bucket=%s gamma=%.4f trying solvers=%s",
        bucket_label, gamma, default_solvers,
      )

    weights, solve_diag = _qp_solve_weights(
      mu,
      sigma,
      gamma,
      lower_bounds=np.full(len(holdings_rows), min_weight, dtype=float)
      if len(holdings_rows) <= int(round(1.0 / min_weight))
      else None,
      upper_bounds=upper_bounds,
      risk_cap_pct=bucket_hi_v,
      scale_to_monthly=float(scaling_policy["scale_to_monthly"]),
    )
    solve_diag["gamma"] = _to_json_float(gamma)
    attempts = solve_diag.get("solver_attempts")
    if isinstance(attempts, list):
      for attempt in attempts:
        if isinstance(attempt, dict):
          bucket_solver_attempts.append([
            attempt.get("solver_name"),
            attempt.get("status"),
            attempt.get("fail_reason"),
          ])
    last_status = solve_diag.get("prob.status")
    chosen_solver = solve_diag.get("chosen_solver")

    if weights is None:
      solver_failure_reason = str(solve_diag.get("fail_reason") or "solver_failed")
      best_solver_diag = solve_diag
      # 실패 처리 (기존 코드와 동일)
      if debug:
        constraints_meta.update({
          "constraints_met": False,
          "missing_required_classes": list(required_classes),
          "holdings_display_limit": int(max_holdings),
          "holdings_display_truncated": False,
        })
        qp_audit["buckets"][bucket_label]["solver_failure_reason"] = solver_failure_reason
        qp_audit["buckets"][bucket_label]["solver_diag"] = best_solver_diag
        qp_audit["buckets"][bucket_label]["solver_attempts"] = bucket_solver_attempts
      items.append({
        "risk_bucket": bucket_label,
        "risk_pct": None,
        "return_6m": None,
        "holdings": [],
        "error": solver_failure_reason,
        "meta": {
          "gamma": gamma,
          "within_bucket": False,
          "target_risk": target_risk,
          "feasible_min_risk_est": feasible_min_risk_est,
          "reason_category": "solver_failure",
          "lo": bucket["min"],
          "hi": bucket["max"],
          "constraints": constraints_meta,
        },
      })
      continue

    weights = _normalize_weights(weights)
    best_weights = weights
    best_risk = _portfolio_risk_pct(
      weights, sigma, scale_to_monthly=float(scaling_policy["scale_to_monthly"])
    )
    best_solver_diag = solve_diag

    # STEP 2: TopK 선택 (기존 로직 유지)
    selected_indices, display_truncated = _select_qp_holdings_topk(
      holdings_rows,
      best_weights,
      required_classes,
      max_holdings=max_holdings,
      min_weight=min_weight,
    )
    if not selected_indices:
      if debug:
        constraints_meta.update({
          "constraints_met": False,
          "missing_required_classes": list(required_classes),
          "holdings_display_limit": int(max_holdings),
          "holdings_display_truncated": False,
        })
        qp_audit["buckets"][bucket_label]["solver_failure_reason"] = "postprocess_empty"
        qp_audit["buckets"][bucket_label]["solver_diag"] = best_solver_diag
        qp_audit["buckets"][bucket_label]["solver_attempts"] = bucket_solver_attempts
      items.append({
        "risk_bucket": bucket_label,
        "risk_pct": None,
        "return_6m": None,
        "holdings": [],
        "error": "postprocess_empty",
        "meta": {
          "gamma": gamma, "within_bucket": False,
          "target_risk": target_risk,
          "feasible_min_risk_est": feasible_min_risk_est,
          "reason_category": "postprocess",
          "lo": bucket["min"], "hi": bucket["max"],
          "constraints": constraints_meta,
        },
      })
      continue

    selected_rows = [holdings_rows[idx] for idx in selected_indices]
    selected_codes = [h["Code"] for h in selected_rows]
    upper_selected = np.full(len(selected_rows), 0.4, dtype=float)
    sel_sigma_idx = [codes.index(c) for c in selected_codes]
    sel_sigma = sigma[np.ix_(sel_sigma_idx, sel_sigma_idx)]

    # STEP 3: 5% 이산화 (Largest Remainder)
    cont_w_sel = _normalize_weights(best_weights[selected_indices])
    w_q = _quantize_5pct(cont_w_sel, upper_selected, step=0.05)

    if w_q is None:
      # 이산화 실패 -> 기존 _project_weights_with_bounds 로 fallback
      lower_bounds = np.full(len(selected_rows), min_weight, dtype=float)
      w_q_proj = _project_weights_with_bounds(cont_w_sel, lower_bounds, upper_selected)
      if w_q_proj is None:
        if debug:
          qp_audit["buckets"][bucket_label]["solver_failure_reason"] = "postprocess_infeasible"
          qp_audit["buckets"][bucket_label]["solver_diag"] = best_solver_diag
        items.append({
          "risk_bucket": bucket_label, "risk_pct": None, "return_6m": None,
          "holdings": [], "error": "postprocess_infeasible",
          "meta": {
            "gamma": gamma, "within_bucket": False, "target_risk": target_risk,
            "feasible_min_risk_est": feasible_min_risk_est,
            "reason_category": "postprocess",
            "lo": bucket["min"], "hi": bucket["max"],
            "constraints": constraints_meta,
          },
        })
        continue
      w_q = w_q_proj
      quantize_method = "project_fallback"
    else:
      quantize_method = "largest_remainder_5pct"

    # STEP 4: Repair 스왑 (lo <= risk <= hi 달성)
    w_final, repair_diag = _swap_repair_band(
      w_q,
      sel_sigma,
      upper_selected,
      lo_pct=bucket_lo,
      hi_pct=bucket_hi_v if bucket_hi_v is not None else 99.0,
      scale_to_monthly=float(scaling_policy["scale_to_monthly"]),
      max_steps=400,
    )
    feasible_in_band = repair_diag["repair_success"]
    achieved_risk = repair_diag["achieved_risk_pct"]

    # fallback 판단
    fallback_used = not feasible_in_band
    fallback_reason = "" if feasible_in_band else "repair_swap_exhausted"

    # STEP 5: 최종 정리 (상위 5종목 고정, 불가 시 4종목)
    best_weight_by_code = {codes[i]: float(best_weights[i]) for i in range(len(codes))}
    post_processed = _soft_select_after_quantize(
      holdings_rows,
      selected_rows,
      w_final,
      best_weight_by_code,
      min_weight=min_weight,
      primary_count=5,
      fallback_count=4,
    )
    if post_processed is None:
      items.append({
        "risk_bucket": bucket_label, "risk_pct": None, "return_6m": None,
        "holdings": [], "error": "postprocess_zero",
        "meta": {
          "gamma": gamma, "within_bucket": False, "target_risk": target_risk,
          "feasible_min_risk_est": feasible_min_risk_est,
          "reason_category": "postprocess",
          "lo": bucket["min"], "hi": bucket["max"],
          "constraints": constraints_meta,
        },
      })
      continue
    final_rows_out, final_weights, fixed_count, class_swap_count = post_processed
    quantize_method = f"{quantize_method}|top{fixed_count}"
    weights_pct = _format_weight_percentages(final_weights, decimals=2)

    final_classes = {h["asset_class"] for h in final_rows_out}
    missing_required_classes = [
      ac for ac in required_classes if ac not in final_classes
    ]
    constraints_met = len(missing_required_classes) == 0

    holdings_output = []
    total_return = 0.0
    for holding, weight_pct in zip(final_rows_out, weights_pct):
      if float(weight_pct) <= 0:
        continue
      total_return += (weight_pct / 100) * float(holding.get("return_6m", 0.0))
      holdings_output.append({
        "Code": holding["Code"],
        "Name": holding["Name"],
        "weight": float(weight_pct),
        "asset_class": holding["asset_class"],
      })
    holdings_output = sorted(holdings_output, key=lambda h: (-h["weight"], h["Code"]))

    # 최종 리스크/샤프 (최종 종목 기준 sigma 재추출)
    pos_sigma_idx = [codes.index(str(h["Code"])) for h in final_rows_out]
    pos_sigma = sigma[np.ix_(pos_sigma_idx, pos_sigma_idx)]
    final_risk = _portfolio_risk_pct(
      final_weights, pos_sigma,
      scale_to_monthly=float(scaling_policy["scale_to_monthly"]),
    )
    final_sharpe = _compute_portfolio_sharpe_from_returns(
      returns_slice_all,
      [h["Code"] for h in final_rows_out],
      final_weights,
      periods_per_year=int(scaling_policy["periods_per_year"]),
    )
    adequacy_meta = _portfolio_adequacy_meta(final_rows_out, final_weights, final_risk)

    # final_hash (버킷 간 중복 감지)
    final_hash = _compute_final_hash(
      final_weights,
      [h["Code"] for h in final_rows_out],
    )
    if final_hash in bucket_hashes.values():
      dup_warning = (
        f"duplicate_solution: same hash as bucket "
        f"{[k for k,v in bucket_hashes.items() if v == final_hash]}"
      )
    else:
      dup_warning = None
    bucket_hashes[bucket_label] = final_hash

    if debug:
      constraints_meta.update({
        "constraints_met": constraints_met,
        "missing_required_classes": missing_required_classes,
        "holdings_display_limit": int(max_holdings),
        "holdings_display_truncated": bool(display_truncated),
      })
      qp_audit["buckets"][bucket_label].update({
        "solver_failure_reason": None,
        "S5_portfolios_produced": 1,
        "solver_diag": best_solver_diag,
        "solver_attempts": bucket_solver_attempts,
        "band_lo": bucket_lo,
        "band_hi": bucket_hi_v,
        "achieved_risk_pct": _to_json_float(final_risk),
        "feasible_in_band": feasible_in_band,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "quantize_method": quantize_method,
        "fixed_holdings_count": fixed_count,
        "class_soft_swaps": class_swap_count,
        "repair_steps": repair_diag["repair_steps"],
        "final_hash": final_hash,
        "duplicate_warning": dup_warning,
      })
      logger.info(
        "[QP] bucket=%s gamma=%.4f risk=%.4f in_band=%s hash=%s dup=%s",
        bucket_label, gamma,
        final_risk or -1,
        feasible_in_band,
        final_hash[:8],
        dup_warning is not None,
      )

    within_bucket_flag = (
      (final_risk is not None)
      and (bucket_lo <= final_risk)
      and (bucket_hi_v is None or final_risk <= bucket_hi_v)
    )

    items.append({
      "risk_bucket": bucket_label,
      "risk_pct": final_risk,
      "return_6m": total_return,
      "sharpe_120d": final_sharpe,
      "sharpe_window": final_sharpe,
      "holdings": holdings_output,
      "meta": {
        "gamma": gamma,
        "within_bucket": within_bucket_flag,
        "feasible_in_band": feasible_in_band,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "target_risk": target_risk,
        "feasible_min_risk_est": feasible_min_risk_est,
        "lo": bucket["min"],
        "hi": bucket["max"],
        "achieved_risk_pct": _to_json_float(final_risk),
        "quantize_method": quantize_method,
        "fixed_holdings_count": fixed_count,
        "class_soft_swaps": class_swap_count,
        "repair_steps": repair_diag["repair_steps"],
        "final_hash": final_hash,
        "duplicate_warning": dup_warning,
        **adequacy_meta,
        "constraints": constraints_meta,
      },
    })

  if qp_audit is not None:
    for item in items:
      label = item.get("risk_bucket")
      if label in qp_audit["buckets"] and item.get("error"):
        qp_audit["buckets"][label]["S5_portfolios_produced"] = 0
        if qp_audit["buckets"][label]["solver_failure_reason"] is None:
          qp_audit["buckets"][label]["solver_failure_reason"] = str(item.get("error"))
    # 버킷 간 중복 해시 최종 검증
    hash_counter: Dict[str, List[str]] = {}
    for item in items:
      h = (item.get("meta") or {}).get("final_hash")
      label = item.get("risk_bucket", "")
      if h:
        hash_counter.setdefault(h, []).append(label)
    for item in items:
      h = (item.get("meta") or {}).get("final_hash")
      if h and len(hash_counter.get(h, [])) > 1:
        if isinstance(item.get("meta"), dict):
          item["meta"]["duplicate_solution_across_buckets"] = True
          item["meta"]["duplicate_with_buckets"] = [
            b for b in hash_counter[h]
            if b != item.get("risk_bucket")
          ]
    meta["qp_audit"] = qp_audit

  return items, meta


def _generate_portfolios_qp(
  metrics: pd.DataFrame,
  config: Dict[str, object],
  returns_tail: pd.DataFrame | None,
  score_mode: str,
  debug: bool = False,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
  return build_portfolio_qp(
    metrics,
    config,
    returns_tail,
    score_mode,
    mode="base",
    debug=debug,
  )


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
    "topN_by_class": 20,
  }
  if config:
    default_config.update(config)
  score_mode = (score_mode or "sharpe").lower()
  if score_mode not in ("sharpe", "return"):
    score_mode = "sharpe"
  inferred_freq, inferred_ppy = _infer_returns_frequency(returns_tail)
  scaling_policy = _risk_scaling_policy(inferred_freq, inferred_ppy)

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
    units = {
      "risk_unit": "monthly_vol_pct",
      "risk_pct_unit": "monthly_vol_pct",
      "return_6m_unit": "cumulative",
      "return_120d_unit": "cumulative",
      "sharpe_120d_unit": "annualized",
      "return_calc_basis": "weekly_log_internal_return52w_simple_output",
      "freq": scaling_policy["freq"],
      "window_periods": scaling_policy["window_periods"],
      "window_days": scaling_policy["window_days"],
      "window_weeks": scaling_policy["window_weeks"],
      "scale_to_monthly": scaling_policy["scale_to_monthly"],
      "score_mode": score_mode,
    }
    return format_portfolios(
      empty_items,
      strategy,
      {
        "version": "empty",
        "freq": scaling_policy["freq"],
        "window_periods": scaling_policy["window_periods"],
        "scale_to_monthly": scaling_policy["scale_to_monthly"],
        "units": units,
      },
    )

  base_items: List[Dict[str, object]] | None = None
  delta_items: List[Dict[str, object]] | None = None
  base_meta: Dict[str, object] | None = None
  delta_meta: Dict[str, object] | None = None

  if strategy == "sampled":
    items, meta = _generate_portfolios_sampled(metrics, default_config, returns_tail, score_mode)
  elif strategy == "qp":
    score_mode = "sharpe"
    base_items, base_meta = build_portfolio_qp(
      metrics,
      default_config,
      returns_tail,
      score_mode,
      mode="base",
      debug=bool(debug),
    )
    delta_items, delta_meta = build_portfolio_qp(
      metrics,
      default_config,
      returns_tail,
      score_mode,
      mode="delta",
      debug=bool(debug),
    )
    items = list(base_items)
    meta = dict(base_meta or {})
    meta["delta"] = delta_meta or {}
  elif strategy == "grid":
    items, meta = _generate_portfolios_grid(metrics, default_config)
  elif strategy == "bucket_fallback":
    items, meta = _generate_portfolios_bucket_fallback(metrics, default_config)
  else:
    items, meta = _generate_portfolios_sampled(metrics, default_config)
    meta["strategy_warning"] = "unknown_strategy"

  units = {
    "risk_unit": "monthly_vol_pct",
    "risk_pct_unit": "monthly_vol_pct",
    "return_6m_unit": "cumulative",
    "return_120d_unit": "cumulative",
    "sharpe_120d_unit": "annualized",
    "return_calc_basis": "weekly_log_internal_return52w_simple_output",
    "freq": scaling_policy["freq"],
    "window_periods": scaling_policy["window_periods"],
    "window_days": scaling_policy["window_days"],
    "window_weeks": scaling_policy["window_weeks"],
    "scale_to_monthly": scaling_policy["scale_to_monthly"],
    "sharpe_window_periods": scaling_policy["window_periods"],
    "score_mode": score_mode,
  }
  if meta is None:
    meta = {}
  meta["freq"] = meta.get("freq", scaling_policy["freq"])
  meta["window_periods"] = int(meta.get("window_periods", scaling_policy["window_periods"]))
  meta["scale_to_monthly"] = float(meta.get("scale_to_monthly", scaling_policy["scale_to_monthly"]))
  if bool(debug) and str(meta.get("freq", "")).lower() == "daily" and float(meta.get("scale_to_monthly", 0.0)) > 50.0:
    meta["debug_warning"] = "scale_to_monthly_suspicious"
  existing_units = meta.get("units") if isinstance(meta.get("units"), dict) else {}
  meta["units"] = {**existing_units, **units}
  meta["score_mode"] = units["score_mode"]
  qp_audit = meta.get("qp_audit") if isinstance(meta.get("qp_audit"), dict) else None
  if qp_audit is not None:
    cov_alpha = meta.get("cov_shrink_alpha", qp_audit.get("cov_shrink_alpha", 0.0))
    try:
      cov_alpha = float(cov_alpha)
    except Exception:
      cov_alpha = 0.0
    qp_audit.update({
      "freq": meta["freq"],
      "window_periods": int(meta["window_periods"]),
      "scale_to_monthly": float(meta["scale_to_monthly"]),
      "cov_shrink_alpha": cov_alpha,
    })
    if meta.get("debug_warning"):
      qp_audit["warning"] = str(meta["debug_warning"])
    meta["qp_audit"] = qp_audit
  payload = format_portfolios(items, strategy, meta)
  if strategy == "qp":
    payload["base"] = base_items or items
    payload["delta"] = delta_items or []
    payload["items"] = payload["base"]
    payload["count"] = len(payload["items"])
  return payload


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
      "S1_candidate_topN": {label: (data.get("S1_candidate_topN") or {}) for label, data in buckets.items()},
      "S2_metrics_non_nan": {label: int(data.get("S2_metrics_non_nan", 0)) for label, data in buckets.items()},
      "S3_asset_class": {
        label: {
          "known": int((data.get("S3_asset_class") or {}).get("known", 0)),
          "unknown": int((data.get("S3_asset_class") or {}).get("unknown", 0)),
          "by_class": ((data.get("S3_asset_class") or {}).get("by_class") or {}),
        }
        for label, data in buckets.items()
      },
      "S4_class_min_feasible": {label: int(data.get("S4_class_min_feasible", 0)) for label, data in buckets.items()},
      "S5_portfolios_produced": {label: int(data.get("S5_portfolios_produced", 0)) for label, data in buckets.items()},
      "solver_failure_reason": {label: data.get("solver_failure_reason") for label, data in buckets.items()},
      "band_summary": {
        label: {
          "band_lo": (data.get("band_lo")),
          "band_hi": (data.get("band_hi")),
          "achieved_risk_pct": (data.get("achieved_risk_pct")),
          "feasible_in_band": (data.get("feasible_in_band")),
          "repair_steps": (data.get("repair_steps")),
          "final_hash": (data.get("final_hash") or "")[:8],
          "duplicate_warning": (data.get("duplicate_warning")),
        }
        for label, data in buckets.items()
      },
    }

  payload["debug"] = {
    "debug_code": debug_code,
    "note": "B0.5 audit enabled",
    # Stage meanings: S0 universe -> S1 topN candidate pools(by class) -> S2 metric-valid candidates -> S3 class counts -> S4 class-min feasibility -> S5 bucket portfolio produced.
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
