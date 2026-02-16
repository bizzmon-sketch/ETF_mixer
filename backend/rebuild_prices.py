#!/usr/bin/env python3
import argparse
import csv
import sqlite3
from datetime import date
from pathlib import Path

import FinanceDataReader as fdr


CREATE_PRICES_SQL = """
CREATE TABLE IF NOT EXISTS prices(
  code TEXT NOT NULL,
  date TEXT NOT NULL,
  close REAL NOT NULL,
  PRIMARY KEY(code, date)
)
"""


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Rebuild prices.sqlite from scratch using FinanceDataReader."
  )
  parser.add_argument("--db", default="data/prices.sqlite", help="SQLite DB path")
  parser.add_argument("--start", default="2000-01-01", help="Start date (YYYY-MM-DD)")
  parser.add_argument(
    "--end",
    default=date.today().isoformat(),
    help="End date (YYYY-MM-DD), default is today",
  )
  return parser.parse_args()


def load_codes(csv_path: Path) -> list[str]:
  with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    if not reader.fieldnames:
      raise ValueError(f"No header found in {csv_path}")
    code_col = None
    for field in reader.fieldnames:
      if field and field.strip().lower() == "code":
        code_col = field
        break
    if code_col is None:
      raise ValueError(f"'code' column not found in {csv_path}")

    codes = []
    for row in reader:
      raw = (row.get(code_col) or "").strip()
      if raw:
        codes.append(raw)
  return codes


def normalize_date(value) -> str:
  if hasattr(value, "strftime"):
    return value.strftime("%Y-%m-%d")
  return str(value)[:10]


def ensure_fresh_db(db_path: Path) -> sqlite3.Connection:
  db_path.parent.mkdir(parents=True, exist_ok=True)
  if db_path.exists():
    db_path.unlink()
  conn = sqlite3.connect(str(db_path))
  conn.execute(CREATE_PRICES_SQL)
  conn.commit()
  return conn


def commit_if_needed(
  conn: sqlite3.Connection,
  tickers_since_commit: int,
  rows_since_commit: int,
) -> tuple[int, int]:
  if tickers_since_commit >= 10 or rows_since_commit >= 50000:
    conn.commit()
    return 0, 0
  return tickers_since_commit, rows_since_commit


def main() -> int:
  args = parse_args()
  root = Path(__file__).resolve().parents[1]
  db_path = (root / args.db).resolve() if not Path(args.db).is_absolute() else Path(args.db)
  csv_path = root / "data" / "etf_list.csv"
  failed_path = root / "data" / "rebuild_failed_codes.txt"

  codes = load_codes(csv_path)
  total_codes = len(codes)

  conn = ensure_fresh_db(db_path)
  succeeded = 0
  failed = 0
  tickers_since_commit = 0
  rows_since_commit = 0

  with failed_path.open("w", encoding="utf-8") as fail_file:
    for i, code in enumerate(codes, start=1):
      try:
        df = fdr.DataReader(code, start=args.start, end=args.end)
        if df is None or df.empty:
          succeeded += 1
          tickers_since_commit += 1
          tickers_since_commit, rows_since_commit = commit_if_needed(
            conn, tickers_since_commit, rows_since_commit
          )
          print(f"{i}/{total_codes} {code} rows=0")
          continue

        rows = []
        for idx, row in df.iterrows():
          close_val = row.get("Close")
          if close_val is None:
            continue
          rows.append((code, normalize_date(idx), float(close_val)))

        if rows:
          conn.executemany(
            "INSERT OR REPLACE INTO prices(code, date, close) VALUES (?, ?, ?)",
            rows,
          )
          inserted = len(rows)
          rows_since_commit += inserted
        else:
          inserted = 0

        succeeded += 1
        tickers_since_commit += 1
        tickers_since_commit, rows_since_commit = commit_if_needed(
          conn, tickers_since_commit, rows_since_commit
        )
        print(f"{i}/{total_codes} {code} rows={inserted}")
      except Exception as exc:
        failed += 1
        fail_file.write(f"{code}\t{exc}\n")
        fail_file.flush()
        print(f"{i}/{total_codes} {code} FAILED: {exc}")

  conn.commit()
  stats = conn.execute(
    """
    SELECT
      COUNT(*) AS total_rows,
      MIN(date) AS min_date,
      MAX(date) AS max_date,
      COUNT(DISTINCT code) AS distinct_codes
    FROM prices
    """
  ).fetchone()
  conn.close()

  db_total_rows = stats[0] if stats else 0
  db_min_date = stats[1] if stats else None
  db_max_date = stats[2] if stats else None
  db_distinct_codes = stats[3] if stats else 0

  print("---- SUMMARY ----")
  print(f"succeeded: {succeeded}")
  print(f"failed: {failed}")
  print(f"total rows: {db_total_rows}")
  print(f"MIN(date): {db_min_date}")
  print(f"MAX(date): {db_max_date}")
  print(f"distinct codes in DB: {db_distinct_codes}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
