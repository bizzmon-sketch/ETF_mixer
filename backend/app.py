from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import engine
import portfolio_store

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)
portfolio_store.init_db()


@app.get("/")
def index():
  return send_from_directory(app.static_folder, "index2.html")


@app.get("/api/health")
def health():
  return jsonify({"status": "ok"})


@app.get("/api/scatter")
def scatter():
  items = engine.get_scatter_data()
  return jsonify({"count": len(items), "items": items})


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
  payload = engine.get_portfolios(strategy=strategy, score=score)
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


