from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import engine

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)


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


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=True)


