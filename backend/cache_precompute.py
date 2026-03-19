from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
import tempfile

import engine

BASE_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")


def _now_kst_iso() -> str:
  return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")


def _ensure_cache_dir() -> None:
  os.makedirs(CACHE_DIR, exist_ok=True)


def _attach_metadata(payload: object, source: str) -> object:
  if isinstance(payload, dict):
    enriched = dict(payload)
    enriched["_cached_at"] = _now_kst_iso()
    enriched["_source"] = source
    return enriched
  return payload


def _write_json_atomic(path: str, payload: object) -> None:
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp_fd, tmp_path = tempfile.mkstemp(prefix="tmp_", dir=os.path.dirname(path))
  try:
    with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
      json.dump(payload, handle, ensure_ascii=False)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(tmp_path, path)
  except Exception:
    try:
      os.unlink(tmp_path)
    except Exception:
      pass


def _build_scatter() -> dict:
  items = engine.get_scatter_data()
  return {"count": len(items), "items": items}


def _build_portfolios(strategy: str, score: str) -> object:
  return engine.get_portfolios(strategy=strategy, score=score)


def main() -> None:
  _ensure_cache_dir()

  scatter_payload = _attach_metadata(_build_scatter(), "precompute")
  scatter_path = os.path.join(CACHE_DIR, "scatter.json")
  _write_json_atomic(scatter_path, scatter_payload)

  strategy = "sampled"
  score = "sharpe"
  portfolios_payload = _attach_metadata(_build_portfolios(strategy, score), "precompute")
  portfolios_path = os.path.join(CACHE_DIR, f"portfolios_{strategy}_{score}.json")
  _write_json_atomic(portfolios_path, portfolios_payload)

  qp_payload = _attach_metadata(
      _build_portfolios("qp", "sharpe"), "precompute")
  qp_path = os.path.join(CACHE_DIR, "portfolios_qp_sharpe.json")
  _write_json_atomic(qp_path, qp_payload)

  try:
    trend_payload = engine.get_trend_portfolio()
    trend_payload = _attach_metadata(trend_payload, "precompute")
    if "error" not in trend_payload:
      trend_path = os.path.join(CACHE_DIR, "portfolios_trend.json")
      _write_json_atomic(trend_path, trend_payload)
  except Exception as e:
    print(f"[cache_precompute] trend portfolio 캐시 실패: {e}")


if __name__ == "__main__":
  main()
