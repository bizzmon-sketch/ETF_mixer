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
  weekly_meta = _weekly_meta_from_cache()
  for key, value in weekly_meta.items():
    if key == "is_current_week_partial":
      base_meta[key] = bool(value)
      continue
    if value is not None:
      base_meta[key] = value
  if base_meta.get("data_as_of") is None:
    base_meta["data_as_of"] = base_meta.get("data_asof")
  if base_meta.get("freq") is None:
    base_meta["freq"] = "weekly"
  if base_meta.get("return_unit") is None:
    base_meta["return_unit"] = "simple_52w"
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
      if strategy == "qp":
        has_dual = isinstance(cached, dict) and isinstance(cached.get("base"), list) and isinstance(cached.get("delta"), list)
        if not has_dual:
          cached = None
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


def _safe_float(value: object) -> float | None:
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def _is_multiple_of_5(value: float) -> bool:
  if value is None:
    return False
  return abs((value / 5.0) - round(value / 5.0)) < 1e-9


def _load_weekly_prices(codes: list[str], weeks: int) -> object:
  end = engine.pd.Timestamp.today().normalize()
  # 26주 + 리샘플 여유분 확보를 위해 9개월 범위를 조회한다.
  start = end - engine.pd.DateOffset(months=9)
  daily = engine.load_close_prices(codes, start, end)
  if daily is None or daily.empty:
    return engine.pd.DataFrame()
  weekly = daily.resample("W-FRI", label="right", closed="right").last().sort_index()
  if weeks > 0 and len(weekly) > (weeks + 1):
    weekly = weekly.tail(weeks + 1)
  return weekly


def _format_price_series(series: object) -> list[dict[str, object]]:
  if series is None or len(series) == 0:
    return []
  base = float(series.iloc[0])
  if base <= 0:
    return []
  normalized = series / base
  items = []
  for idx, value in normalized.items():
    items.append({
      "date": idx.strftime("%Y-%m-%d"),
      "value": round(float(value), 6),
    })
  return items


@app.post("/api/custom_portfolio_eval")
def custom_portfolio_eval():
  payload = request.get_json(silent=True) or {}
  holdings_raw = payload.get("holdings") or []
  if not isinstance(holdings_raw, list):
    return jsonify({"error": "invalid_holdings", "message": "holdings 형식이 올바르지 않습니다"}), 400
  if not (2 <= len(holdings_raw) <= 6):
    return jsonify({"error": "invalid_count", "message": "종목 수는 2~6개여야 합니다"}), 400

  codes: list[str] = []
  weights: list[float] = []
  seen = set()
  cleaned_holdings = []
  for item in holdings_raw:
    if not isinstance(item, dict):
      return jsonify({"error": "invalid_holdings", "message": "holdings 형식이 올바르지 않습니다"}), 400
    code = str(item.get("code", "")).strip()
    if not code:
      return jsonify({"error": "invalid_code", "message": "종목 코드가 비어 있습니다"}), 400
    if code in seen:
      return jsonify({"error": "duplicate_code", "message": "중복 종목 코드는 허용되지 않습니다"}), 400
    weight = _safe_float(item.get("weight"))
    if weight is None:
      return jsonify({"error": "invalid_weight", "message": "비중 형식이 올바르지 않습니다"}), 400
    if not _is_multiple_of_5(weight):
      return jsonify({"error": "weight_not_5pct", "message": "비중은 5% 단위여야 합니다"}), 400
    seen.add(code)
    codes.append(code)
    weights.append(weight)
    cleaned_holdings.append({
      "code": code,
      "weight": int(round(weight)),
    })

  total_weight = float(sum(weights))
  if abs(total_weight - 100.0) > 1e-9:
    return jsonify({
      "error": "weights_not_100",
      "message": "비중 합계가 100%가 아닙니다",
      "total": round(total_weight, 6),
    }), 400

  weekly_prices = _load_weekly_prices(codes, 26)
  if weekly_prices.empty:
    return jsonify({"error": "code_not_found", "message": "DB에 없는 종목 코드가 포함되어 있습니다"}), 400
  missing_codes = [code for code in codes if code not in weekly_prices.columns]
  if missing_codes:
    return jsonify({
      "error": "code_not_found",
      "message": "DB에 없는 종목 코드가 포함되어 있습니다",
      "codes": missing_codes,
    }), 400

  aligned = weekly_prices[codes].dropna(how="any")
  if len(aligned) < 27:
    return jsonify({
      "error": "insufficient_data",
      "message": "26주 계산에 필요한 주봉 데이터가 부족합니다",
    }), 400
  prices_26w = aligned.tail(27)
  returns_26w = engine.np.log(prices_26w / prices_26w.shift(1)).dropna(how="any")

  bh_return, bh_risk, bh_sharpe, bh_prices = engine.compute_custom_portfolio_bh(codes, weights, returns_26w)
  rb_return, rb_risk, rb_sharpe, rb_prices = engine.compute_custom_portfolio_rb(codes, weights, prices_26w)

  response = {
    "holdings": cleaned_holdings,
    "bh": {
      "return_26w": bh_return,
      "risk_pct": bh_risk,
      "sharpe": bh_sharpe,
      "prices": bh_prices,
    },
    "rb": {
      "return_26w": rb_return,
      "risk_pct": rb_risk,
      "sharpe": rb_sharpe,
      "prices": rb_prices,
    },
    "scatter_point": {
      "bh": {
        "x": bh_risk,
        "y": round(bh_return * 100.0, 4),
        "sharpe": bh_sharpe,
      },
      "rb": {
        "x": rb_risk,
        "y": round(rb_return * 100.0, 4),
        "sharpe": rb_sharpe,
      },
    },
  }
  return jsonify(response)


@app.get("/api/benchmark_prices")
def benchmark_prices():
  weeks_raw = (request.args.get("weeks", "26") or "26").strip()
  try:
    weeks = int(weeks_raw)
  except ValueError:
    weeks = 26
  if weeks <= 0:
    weeks = 26

  benchmarks = {
    "KOSPI": "069500",
    "KOSDAQ": "229200",
    "SNP500": "360750",
  }
  codes = list(benchmarks.values())
  weekly_prices = _load_weekly_prices(codes, weeks)

  result = {}
  for label, code in benchmarks.items():
    if weekly_prices.empty or code not in weekly_prices.columns:
      result[label] = None
      continue
    series = weekly_prices[code].dropna()
    if len(series) < 2:
      result[label] = None
      continue
    if len(series) > (weeks + 1):
      series = series.tail(weeks + 1)
    result[label] = {
      "code": code,
      "prices": _format_price_series(series),
    }

  return jsonify({
    "weeks": weeks,
    "benchmarks": result,
  })


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=True)


