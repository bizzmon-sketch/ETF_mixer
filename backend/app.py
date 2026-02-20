from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

import engine
import portfolio_store

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)
portfolio_store.init_db()


BASE_DIR = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache")


def _now_kst_iso() -> str:
  return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds")


def _attach_metadata(payload: object, source: str) -> object:
  if isinstance(payload, dict):
    enriched = dict(payload)
    enriched["_cached_at"] = _now_kst_iso()
    enriched["_source"] = source
    return enriched
  return payload


def _weekly_meta_from_cache() -> dict:
  cache = engine._CACHE if hasattr(engine, "_CACHE") else {}
  return {
    "data_as_of": cache.get("data_asof"),
    "is_current_week_partial": cache.get("is_week_partial", False),
    "weekly_last_label": cache.get("weekly_last_label"),
    "weekly_last_observed": cache.get("weekly_last_observed"),
  }


def _enrich_scatter_payload(payload: object) -> object:
  if not isinstance(payload, dict):
    return payload
  enriched = dict(payload)
  base_meta = dict(enriched.get("meta") or {})
  base_meta.update(_weekly_meta_from_cache())
  enriched["meta"] = base_meta
  return enriched


def _enrich_portfolios_payload(payload: object) -> object:
  if not isinstance(payload, dict):
    return payload
  enriched = dict(payload)
  meta = dict(enriched.get("meta") or {})
  units = dict(meta.get("units") or {})
  units.update({
    "is_current_week_partial": engine._CACHE.get("is_week_partial", False) if hasattr(engine, "_CACHE") else False,
    "weekly_last_label": engine._CACHE.get("weekly_last_label") if hasattr(engine, "_CACHE") else None,
    "weekly_last_observed": engine._CACHE.get("weekly_last_observed") if hasattr(engine, "_CACHE") else None,
  })
  meta["units"] = units
  enriched["meta"] = meta
  return enriched


def _read_json_cache(path: str) -> object | None:
  try:
    with open(path, "r", encoding="utf-8") as handle:
      return json.load(handle)
  except (OSError, json.JSONDecodeError):
    return None


def _write_json_atomic(path: str, payload: object) -> None:
  try:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="tmp_", dir=os.path.dirname(path))
    with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
      json.dump(payload, handle, ensure_ascii=False)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(tmp_path, path)
  except Exception:
    try:
      if 'tmp_path' in locals():
        os.unlink(tmp_path)
    except Exception:
      pass


@app.get("/")
def index():
  return send_from_directory(app.static_folder, "index2.html")


@app.get("/api/health")
def health():
  return jsonify({"status": "ok"})


@app.get("/api/scatter")
def scatter():
  cache_path = os.path.join(CACHE_DIR, "scatter.json")
  cached = _read_json_cache(cache_path)
  if cached is not None:
    return jsonify(_enrich_scatter_payload(cached))
  items = engine.get_scatter_data()
  payload = {"count": len(items), "items": items, "meta": engine.get_scatter_meta()}
  payload = _enrich_scatter_payload(payload)
  payload = _attach_metadata(payload, "runtime")
  _write_json_atomic(cache_path, payload)
  return jsonify(payload)


@app.get("/api/recommendations")
def recommendations():
  items = engine.get_recommendations()
  return jsonify({"count": len(items), "items": items})


@app.get("/api/delta3m")
def delta3m():
  items = engine.get_delta3m()
  return jsonify({"count": len(items), "items": items})


@app.get("/api/prices")
def prices():
  code = request.args.get("code", "").strip()
  days = request.args.get("days", "120")
  payload = engine.get_price_series(code, days)
  return jsonify(payload)


@app.get("/api/portfolios")
def portfolios():
  strategy = request.args.get("strategy", "sampled")
  score = request.args.get("score", "sharpe")
  # `debug=1` forces a live compute path so debug payloads are never stale.
  debug_raw = (request.args.get("debug", "0") or "0").strip().lower()
  debug = debug_raw in ("1", "true", "yes", "y", "on")
  debug_code = (request.args.get("debug_code", "") or "").strip()
  if not debug_code:
    debug_code = None

  cache_path = os.path.join(CACHE_DIR, f"portfolios_{strategy}_{score}.json")
  if not debug:
    cached = _read_json_cache(cache_path)
    if cached is not None:
      return jsonify(_enrich_portfolios_payload(cached))

  # Forward debug kwargs when supported; keep backward compatibility otherwise.
  kwargs = {"debug": int(debug), "debug_code": debug_code}
  try:
    payload = engine.get_portfolios(strategy=strategy, score=score, **kwargs)
  except TypeError:
    payload = engine.get_portfolios(strategy=strategy, score=score)
  payload = _enrich_portfolios_payload(payload)
  payload = _attach_metadata(payload, "runtime")
  if not debug:
    _write_json_atomic(cache_path, payload)
  return jsonify(payload)


@app.get("/api/portfolios/saved")
def list_saved_portfolios():
  items = portfolio_store.list_portfolios()
  return jsonify({"count": len(items), "items": items})


@app.post("/api/portfolios/saved")
def create_saved_portfolio():
  payload = request.get_json(silent=True) or {}
  name = payload.get("name", "").strip()
  initial_budget = int(payload.get("initial_budget_krw") or 0)
  targets = payload.get("target_allocations") or []
  notes = payload.get("notes")
  try:
    item = portfolio_store.create_portfolio(
      name=name,
      initial_budget_krw=initial_budget,
      target_allocations=targets,
      notes=notes,
    )
  except ValueError as err:
    return jsonify({"error": str(err)}), 400
  return jsonify(item), 201


@app.get("/api/portfolios/saved/<portfolio_id>")
def get_saved_portfolio(portfolio_id: str):
  try:
    detail = portfolio_store.get_portfolio_detail(portfolio_id)
  except ValueError as err:
    return jsonify({"error": str(err)}), 404
  return jsonify(detail)


@app.post("/api/portfolios/saved/<portfolio_id>/trade")
def add_trade(portfolio_id: str):
  payload = request.get_json(silent=True) or {}
  code = payload.get("code", "").strip()
  side = payload.get("side", "").strip()
  qty = int(payload.get("qty") or 0)
  price = payload.get("price")
  try:
    trade = portfolio_store.add_trade(
      portfolio_id=portfolio_id,
      code=code,
      side=side,
      qty=qty,
      price=price,
    )
  except ValueError as err:
    return jsonify({"error": str(err)}), 400
  return jsonify(trade), 201


@app.post("/api/portfolios/saved/<portfolio_id>/rebalance/plan")
def rebalance_plan(portfolio_id: str):
  payload = request.get_json(silent=True) or {}
  targets = payload.get("target_allocations")
  try:
    plan = portfolio_store.build_rebalance_plan(
      portfolio_id=portfolio_id,
      target_allocations=targets,
      band_pct=float(payload.get("band_pct") or 5.0),
    )
  except ValueError as err:
    return jsonify({"error": str(err)}), 400
  return jsonify(plan)


@app.post("/api/portfolios/saved/<portfolio_id>/rebalance/commit")
def rebalance_commit(portfolio_id: str):
  payload = request.get_json(silent=True) or {}
  targets = payload.get("target_allocations") or []
  executed_trades = payload.get("executed_trades")
  try:
    result = portfolio_store.commit_rebalance(
      portfolio_id=portfolio_id,
      target_allocations=targets,
      executed_trades=executed_trades,
    )
  except ValueError as err:
    return jsonify({"error": str(err)}), 400
  return jsonify(result)


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=True)


