import pandas as pd
from pathlib import Path

CONFIG = {
  "input_path": Path("data/data_5731_20251225.csv"),
  "output_path": Path("data/etf_list.csv"),
  "encoding": "euc-kr",
  "manager_list": [
    "미래에셋자산운용",
    "삼성자산운용",
    "신한자산운용",
    "엔에이치아문디자산운용",
    "타임폴리오자산운용",
    "키움투자자산운용",
    "한국투자신탁운용",
    "한화자산운용",
  ],
}


def main() -> None:
  df = pd.read_csv(CONFIG["input_path"], encoding=CONFIG["encoding"])
  filtered = df[(df["운용사"].isin(CONFIG["manager_list"])) & (df["추적배수"] == "일반")].copy()
  result = filtered[["단축코드", "한글종목명"]].rename(
    columns={"단축코드": "Code", "한글종목명": "Name"}
  )
  result = result.dropna().drop_duplicates().sort_values("Code").reset_index(drop=True)
  result.to_csv(CONFIG["output_path"], index=False, encoding="utf-8-sig")
  print(f"Saved {len(result)} rows to {CONFIG['output_path']}")


if __name__ == "__main__":
  main()
