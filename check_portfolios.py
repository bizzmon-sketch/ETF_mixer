import requests

ps = requests.get("http://127.0.0.1:5000/api/portfolios").json()
items = ps.get("items", ps.get("portfolios", []))
req = {"Equity","Bond","Alt","CashLike"}

for p in items:
    hs = p.get("holdings", [])
    w = [h.get("weight", 0) for h in hs]
    classes = {h.get("asset_class") for h in hs}

    print(f"- {p.get('risk_bucket','?')}: n={len(hs)}, sum={sum(w)}, classes={classes}")

    assert len(hs) <= 10
    assert req.issubset(classes)
    s = sum(w)
    assert abs(s-100) < 1e-6 or abs(s-1) < 1e-6
    for x in w:
        assert x >= 5 or x >= 0.05
        assert x <= 30 or x <= 0.30

print("OK")
