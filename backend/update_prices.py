#!/usr/bin/env python3
import logging
import sqlite3
import sys
from pathlib import Path


PROJECT_ROOT = Path("/home/ubuntu/ETF_mixer")
DB_PATH = PROJECT_ROOT / "data" / "prices.sqlite"
BACKEND_DIR = PROJECT_ROOT / "backend"


def _ensure_engine_import():
  try:
    import engine  # type: ignore
    return engine
  except ImportError:
    local_backend = Path(__file__).resolve().parent
    fallback_paths = [str(local_backend), str(BACKEND_DIR)]
    for path in fallback_paths:
      if path not in sys.path:
        sys.path.insert(0, path)
    import engine  # type: ignore
    return engine


def _get_max_date(db_path: Path) -> str | None:
  if not db_path.exists():
    return None
  with sqlite3.connect(str(db_path)) as conn:
    cursor = conn.execute(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='prices'"
    )
    if cursor.fetchone() is None:
      return None
    row = conn.execute("SELECT MAX(date) FROM prices").fetchone()
    if not row:
      return None
    return row[0]


def main() -> int:
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
  )
  logger = logging.getLogger("update_prices")
  logger.info("Price update started")

  before_max = _get_max_date(DB_PATH)
  print(f"max(date) before: {before_max}")

  engine = _ensure_engine_import()
  engine.CONFIG["price_db_path"] = str(DB_PATH)

  etf_df = engine.load_etf_list()
  codes = etf_df["Code"].astype(str).str.strip().dropna().unique().tolist()
  engine.update_prices_incremental(codes, buffer_days=14)

  after_max = _get_max_date(DB_PATH)
  print(f"max(date) after: {after_max}")
  logger.info("Price update finished")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
