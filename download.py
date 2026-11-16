"""Скачивает дневные свечи индекса IMOEX с ISS Мосбиржи по годам.

Endpoint: /iss/engines/stock/markets/index/securities/IMOEX/candles.json
Параметры: from, till, interval=24 (дневной), start (пагинация).
Каждый год сохраняется в data/raw/imoex_YYYY.csv; уже скачанный год не перекачивается.
После загрузки все годы склеиваются в data/imoex_daily_raw.csv.
"""

import csv
import json
import time
from datetime import date
from pathlib import Path

import requests

URL = "https://iss.moex.com/iss/engines/stock/markets/index/securities/IMOEX/candles.json"
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
MERGED = DATA_DIR / "imoex_daily_raw.csv"
META = DATA_DIR / "download_meta.json"
HEADER = ["begin", "end", "open", "high", "low", "close", "value", "volume"]
TIMEOUT = 30
RETRIES = 3


def get_json(params: dict) -> dict:
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.get(URL, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as error:
            last_error = error
            time.sleep(2 * attempt)
    raise RuntimeError(f"не удалось скачать {params}: {last_error}")


def file_is_valid(path: Path, year: int) -> bool:
    with path.open(encoding="utf-8") as f:
        lines = f.read().splitlines()
    return len(lines) > 1 and lines[-1].startswith(f"{year}-")


def download_year(year: int, till: str) -> int:
    rows, start = [], 0
    while True:
        payload = get_json({"from": f"{year}-01-01", "till": till, "interval": 24, "start": start})["candles"]
        page = payload["data"]
        if not page:
            break
        idx = {name: i for i, name in enumerate(payload["columns"])}
        rows.extend([[r[idx[name]] for name in HEADER] for r in page])
        start += len(page)

    path = RAW_DIR / f"imoex_{year}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today()
    till = today.isoformat()
    counts = {}
    for year in range(2012, today.year + 1):
        path = RAW_DIR / f"imoex_{year}.csv"
        if path.exists() and file_is_valid(path, year):
            counts[year] = sum(1 for _ in path.open(encoding="utf-8")) - 1
            continue
        year_till = till if year == today.year else f"{year}-12-31"
        counts[year] = download_year(year, year_till)
        print(f"{year}: {counts[year]} rows")

    all_rows = []
    for year in range(2012, today.year + 1):
        with (RAW_DIR / f"imoex_{year}.csv").open(encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            all_rows.extend(reader)
    with MERGED.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(all_rows)

    META.write_text(json.dumps({
        "download_date": till,
        "endpoint": URL,
        "params": {"from": "2012-01-01", "till": till, "interval": 24},
        "rows_by_year": counts,
        "rows_total": len(all_rows),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"merged {len(all_rows)} rows -> {MERGED}")


if __name__ == "__main__":
    main()
