from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import json
import os
import sqlite3
import uuid
from typing import Dict, Iterable, List, Optional

import engine


DB_FILENAME = "portfolios.sqlite"
ASSET_CLASSES = ["Equity", "Bond", "Alt", "CashLike"]


def _now_iso() -> str:
  return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")


def _db_path() -> str:
  root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
  return os.path.join(root, "data", DB_FILENAME)


def _ensure_dir(path: str) -> None:
  folder = os.path.dirname(path)
  if folder and not os.path.isdir(folder):
    os.makedirs(folder, exist_ok=True)


def _connect() -> sqlite3.Connection:
  path = _db_path()
  _ensure_dir(path)
  conn = sqlite3.connect(path)
  conn.row_factory = sqlite3.Row
  return conn


def init_db() -> None:
  with _connect() as conn:
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS portfolios (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        initial_budget_krw INTEGER NOT NULL,
        notes TEXT
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS portfolio_targets (
        portfolio_id TEXT NOT NULL,
        code TEXT NOT NULL,
        name TEXT NOT NULL,
        asset_class TEXT NOT NULL,
        target_weight_pct REAL NOT NULL,
        PRIMARY KEY (portfolio_id, code)
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS portfolio_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        portfolio_id TEXT NOT NULL,
        ts TEXT NOT NULL,
        code TEXT NOT NULL,
        side TEXT NOT NULL,
        qty INTEGER NOT NULL,
        price REAL
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS portfolio_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        portfolio_id TEXT NOT NULL,
        ts TEXT NOT NULL,
        snapshot_json TEXT NOT NULL
      )
      """
    )
    conn.commit()


def _sum_weights(targets: List[Dict[str, object]]) -> float:
  return sum(float(item.get("target_weight_pct", 0)) for item in targets)


def _normalize_targets(targets: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
  normalized: List[Dict[str, object]] = []
  for item in targets:
    code = str(item.get("code") or item.get("Code") or "").strip()
    name = str(item.get("name") or item.get("Name") or "").strip()
    asset_class = str(item.get("asset_class") or item.get("assetClass") or "").strip()
    target_weight_pct = float(item.get("target_weight_pct", item.get("weight", 0)) or 0)
    if not code or not name:
      continue
    if not asset_class:
      asset_class = engine.classify_asset_class(name)
    normalized.append({
      "code": code,
      "name": name,
      "asset_class": asset_class,
      "target_weight_pct": target_weight_pct,
    })
  return normalized


def create_portfolio(
  name: str,
  initial_budget_krw: int,
  target_allocations: Iterable[Dict[str, object]],
  notes: Optional[str] = None,
) -> Dict[str, object]:
  if not name:
    raise ValueError("name_required")
  targets = _normalize_targets(target_allocations)
  if not targets:
    raise ValueError("targets_required")
  if abs(_sum_weights(targets) - 100) > 0.5:
    raise ValueError("targets_sum_not_100")
  portfolio_id = str(uuid.uuid4())
  now = _now_iso()
  with _connect() as conn:
    conn.execute(
      """
      INSERT INTO portfolios (id, name, created_at, updated_at, initial_budget_krw, notes)
      VALUES (?, ?, ?, ?, ?, ?)
      """,
      (portfolio_id, name, now, now, int(initial_budget_krw), notes),
    )
    conn.executemany(
      """
      INSERT INTO portfolio_targets (portfolio_id, code, name, asset_class, target_weight_pct)
      VALUES (?, ?, ?, ?, ?)
      """,
      [
        (portfolio_id, t["code"], t["name"], t["asset_class"], float(t["target_weight_pct"]))
        for t in targets
      ],
    )
    conn.commit()
  return {
    "id": portfolio_id,
    "name": name,
    "created_at": now,
    "updated_at": now,
    "initial_budget_krw": int(initial_budget_krw),
    "targets": targets,
  }


def list_portfolios() -> List[Dict[str, object]]:
  with _connect() as conn:
    rows = conn.execute(
      """
      SELECT id, name, created_at, updated_at, initial_budget_krw
      FROM portfolios
      ORDER BY created_at DESC
      """
    ).fetchall()
  return [dict(row) for row in rows]


def _load_portfolio(portfolio_id: str) -> Optional[Dict[str, object]]:
  with _connect() as conn:
    row = conn.execute(
      """
      SELECT id, name, created_at, updated_at, initial_budget_krw, notes
      FROM portfolios
      WHERE id = ?
      """,
      (portfolio_id,),
    ).fetchone()
  return dict(row) if row else None


def _load_targets(portfolio_id: str) -> List[Dict[str, object]]:
  with _connect() as conn:
    rows = conn.execute(
      """
      SELECT code, name, asset_class, target_weight_pct
      FROM portfolio_targets
      WHERE portfolio_id = ?
      ORDER BY target_weight_pct DESC, code ASC
      """,
      (portfolio_id,),
    ).fetchall()
  return [dict(row) for row in rows]


def _load_trades(portfolio_id: str) -> List[Dict[str, object]]:
  with _connect() as conn:
    rows = conn.execute(
      """
      SELECT id, ts, code, side, qty, price
      FROM portfolio_trades
      WHERE portfolio_id = ?
      ORDER BY ts ASC, id ASC
      """,
      (portfolio_id,),
    ).fetchall()
  return [dict(row) for row in rows]


def _load_latest_snapshot(portfolio_id: str) -> Optional[Dict[str, object]]:
  with _connect() as conn:
    row = conn.execute(
      """
      SELECT ts, snapshot_json
      FROM portfolio_snapshots
      WHERE portfolio_id = ?
      ORDER BY ts DESC, id DESC
      LIMIT 1
      """,
      (portfolio_id,),
    ).fetchone()
  if not row:
    return None
  payload = json.loads(row["snapshot_json"])
  payload["ts"] = row["ts"]
  return payload


def _get_latest_prices(codes: Iterable[str]) -> Dict[str, Optional[float]]:
  prices: Dict[str, Optional[float]] = {}
  for code in codes:
    if not code:
      continue
    series = engine.load_recent_prices(code, days=10)
    if not series:
      prices[code] = None
      continue
    prices[code] = float(series[-1]["close"])
  return prices


def _compute_holdings(
  portfolio: Dict[str, object],
  targets: List[Dict[str, object]],
  trades: List[Dict[str, object]],
) -> Dict[str, object]:
  holdings_qty: Dict[str, int] = {}
  codes = {t["code"] for t in targets}
  for trade in trades:
    code = str(trade["code"]).strip()
    side = str(trade["side"]).upper()
    qty = int(trade["qty"])
    if not code or qty == 0:
      continue
    if side == "BUY":
      holdings_qty[code] = holdings_qty.get(code, 0) + qty
    elif side == "SELL":
      holdings_qty[code] = holdings_qty.get(code, 0) - qty
    elif side == "CASH_IN":
      pass
    if code:
      codes.add(code)

  prices = _get_latest_prices(codes)

  cash = int(portfolio.get("initial_budget_krw") or 0)
  for trade in trades:
    side = str(trade["side"]).upper()
    if side == "CASH_IN":
      cash += int(trade["qty"])
      continue
    code = str(trade["code"]).strip()
    qty = int(trade["qty"])
    price = trade["price"]
    if price is None:
      price = prices.get(code)
    if price is None:
      continue
    if side == "BUY":
      cash -= qty * float(price)
    elif side == "SELL":
      cash += qty * float(price)

  target_map = {t["code"]: t for t in targets}
  holdings: List[Dict[str, object]] = []
  total_value = float(cash)
  for code, qty in holdings_qty.items():
    if qty == 0:
      continue
    price = prices.get(code)
    value = None if price is None else qty * float(price)
    if value is not None:
      total_value += value
    target = target_map.get(code, {})
    holdings.append({
      "code": code,
      "name": target.get("name"),
      "asset_class": target.get("asset_class"),
      "qty": qty,
      "price": price,
      "value": value,
    })

  for holding in holdings:
    if total_value > 0 and holding["value"] is not None:
      holding["weight_pct"] = (holding["value"] / total_value) * 100
    else:
      holding["weight_pct"] = None

  return {
    "cash": cash,
    "prices": prices,
    "holdings": holdings,
    "total_value": total_value,
  }


def get_portfolio_detail(portfolio_id: str) -> Dict[str, object]:
  portfolio = _load_portfolio(portfolio_id)
  if not portfolio:
    raise ValueError("portfolio_not_found")
  targets = _load_targets(portfolio_id)
  trades = _load_trades(portfolio_id)
  latest_snapshot = _load_latest_snapshot(portfolio_id)
  holdings_info = _compute_holdings(portfolio, targets, trades)
  return {
    "portfolio": portfolio,
    "targets": targets,
    "trades": trades,
    "cash": holdings_info["cash"],
    "holdings": holdings_info["holdings"],
    "total_value": holdings_info["total_value"],
    "latest_snapshot": latest_snapshot,
  }


def add_trade(
  portfolio_id: str,
  code: str,
  side: str,
  qty: int,
  price: Optional[float] = None,
  ts: Optional[str] = None,
) -> Dict[str, object]:
  if not code:
    raise ValueError("code_required")
  side = (side or "").upper()
  if side not in ("BUY", "SELL", "CASH_IN"):
    raise ValueError("side_invalid")
  if qty <= 0:
    raise ValueError("qty_invalid")
  portfolio = _load_portfolio(portfolio_id)
  if not portfolio:
    raise ValueError("portfolio_not_found")
  ts = ts or _now_iso()
  with _connect() as conn:
    conn.execute(
      """
      INSERT INTO portfolio_trades (portfolio_id, ts, code, side, qty, price)
      VALUES (?, ?, ?, ?, ?, ?)
      """,
      (portfolio_id, ts, code, side, int(qty), price),
    )
    conn.execute(
      "UPDATE portfolios SET updated_at = ? WHERE id = ?",
      (_now_iso(), portfolio_id),
    )
    conn.commit()
  return {
    "portfolio_id": portfolio_id,
    "ts": ts,
    "code": code,
    "side": side,
    "qty": int(qty),
    "price": price,
  }


def _asset_class_weights(
  holdings: List[Dict[str, object]],
  cash_value: float,
  total_value: float,
) -> Dict[str, float]:
  totals: Dict[str, float] = {cls: 0.0 for cls in ASSET_CLASSES}
  for holding in holdings:
    asset_class = holding.get("asset_class") or "Equity"
    value = holding.get("value")
    if value is None:
      continue
    totals[asset_class] = totals.get(asset_class, 0.0) + float(value)
  totals["CashLike"] = totals.get("CashLike", 0.0) + float(cash_value)
  weights = {}
  for cls, value in totals.items():
    weights[cls] = 0.0 if total_value <= 0 else (value / total_value) * 100
  return weights


def build_rebalance_plan(
  portfolio_id: str,
  target_allocations: Optional[Iterable[Dict[str, object]]] = None,
  band_pct: float = 5.0,
) -> Dict[str, object]:
  portfolio = _load_portfolio(portfolio_id)
  if not portfolio:
    raise ValueError("portfolio_not_found")
  if target_allocations is None:
    targets = _load_targets(portfolio_id)
  else:
    targets = _normalize_targets(target_allocations)
  if not targets:
    raise ValueError("targets_required")
  trades = _load_trades(portfolio_id)
  holdings_info = _compute_holdings(portfolio, targets, trades)

  holdings = holdings_info["holdings"]
  cash = holdings_info["cash"]
  total_value = holdings_info["total_value"]

  target_by_class: Dict[str, float] = {cls: 0.0 for cls in ASSET_CLASSES}
  target_by_code: Dict[str, Dict[str, object]] = {}
  for target in targets:
    asset_class = target.get("asset_class") or "Equity"
    target_by_class[asset_class] = target_by_class.get(asset_class, 0.0) + float(target["target_weight_pct"])
    target_by_code[target["code"]] = target

  current_by_class = _asset_class_weights(holdings, cash, total_value)
  band_flags = []
  for cls in ASSET_CLASSES:
    current = current_by_class.get(cls, 0.0)
    target = target_by_class.get(cls, 0.0)
    diff = current - target
    band_flags.append({
      "asset_class": cls,
      "current_weight_pct": current,
      "target_weight_pct": target,
      "diff_pct": diff,
      "breached": abs(diff) > float(band_pct),
    })

  price_map = holdings_info["prices"]
  current_map = {h["code"]: h for h in holdings}
  orders = []
  for code, target in target_by_code.items():
    price = price_map.get(code)
    if price is None or price <= 0:
      continue
    current_value = current_map.get(code, {}).get("value") or 0.0
    target_value = (total_value * float(target["target_weight_pct"]) / 100) if total_value > 0 else 0.0
    delta = target_value - float(current_value)
    if abs(delta) < price:
      continue
    qty = int(abs(delta) // price)
    if qty <= 0:
      continue
    side = "BUY" if delta > 0 else "SELL"
    if side == "SELL":
      current_qty = int(current_map.get(code, {}).get("qty") or 0)
      qty = min(qty, current_qty)
    if qty <= 0:
      continue
    orders.append({
      "code": code,
      "name": target.get("name"),
      "side": side,
      "qty": qty,
      "price": price,
      "delta_value": delta,
      "target_weight_pct": target["target_weight_pct"],
    })

  return {
    "portfolio_id": portfolio_id,
    "asof": _now_iso(),
    "total_value": total_value,
    "cash": cash,
    "targets": targets,
    "holdings": holdings,
    "asset_class_bands": band_flags,
    "orders": orders,
    "rules": {
      "band_pct": band_pct,
      "qty_rounding": "floor",
      "price_source": "latest_close",
    },
  }


def _store_targets(portfolio_id: str, targets: List[Dict[str, object]]) -> None:
  with _connect() as conn:
    conn.execute(
      "DELETE FROM portfolio_targets WHERE portfolio_id = ?",
      (portfolio_id,),
    )
    conn.executemany(
      """
      INSERT INTO portfolio_targets (portfolio_id, code, name, asset_class, target_weight_pct)
      VALUES (?, ?, ?, ?, ?)
      """,
      [
        (portfolio_id, t["code"], t["name"], t["asset_class"], float(t["target_weight_pct"]))
        for t in targets
      ],
    )
    conn.execute(
      "UPDATE portfolios SET updated_at = ? WHERE id = ?",
      (_now_iso(), portfolio_id),
    )
    conn.commit()


def _store_snapshot(portfolio_id: str, snapshot: Dict[str, object]) -> Dict[str, object]:
  ts = _now_iso()
  payload = json.dumps(snapshot, ensure_ascii=False)
  with _connect() as conn:
    conn.execute(
      """
      INSERT INTO portfolio_snapshots (portfolio_id, ts, snapshot_json)
      VALUES (?, ?, ?)
      """,
      (portfolio_id, ts, payload),
    )
    conn.execute(
      "UPDATE portfolios SET updated_at = ? WHERE id = ?",
      (ts, portfolio_id),
    )
    conn.commit()
  return {"ts": ts, "snapshot": snapshot}


def commit_rebalance(
  portfolio_id: str,
  target_allocations: Iterable[Dict[str, object]],
  executed_trades: Optional[Iterable[Dict[str, object]]] = None,
) -> Dict[str, object]:
  portfolio = _load_portfolio(portfolio_id)
  if not portfolio:
    raise ValueError("portfolio_not_found")
  targets = _normalize_targets(target_allocations)
  if abs(_sum_weights(targets) - 100) > 0.5:
    raise ValueError("targets_sum_not_100")
  _store_targets(portfolio_id, targets)

  applied_trades = []
  if executed_trades:
    for trade in executed_trades:
      code = str(trade.get("code") or "").strip()
      side = str(trade.get("side") or "").upper()
      qty = int(trade.get("qty") or 0)
      price = trade.get("price")
      if not code or qty <= 0:
        continue
      applied_trades.append(add_trade(portfolio_id, code, side, qty, price=price))

  detail = get_portfolio_detail(portfolio_id)
  snapshot = {
    "targets": detail["targets"],
    "holdings": detail["holdings"],
    "cash": detail["cash"],
    "total_value": detail["total_value"],
  }
  stored_snapshot = _store_snapshot(portfolio_id, snapshot)
  return {
    "portfolio_id": portfolio_id,
    "targets": detail["targets"],
    "applied_trades": applied_trades,
    "snapshot": stored_snapshot,
  }
