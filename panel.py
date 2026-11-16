"""Панель акций TQBR: тот же протокол, что для IMOEX (vol_pilot.py).

Шаги:
1. Список бумаг TQBR с историей (ISS listing) и предварительный отбор по датам истории.
2. Дневные свечи по годам в data/panel/raw/{SECID}_{YEAR}.csv (уже скачанные годы не перекачиваются).
3. Отбор: торговля (value > 0) не менее чем в 95% дней IMOEX с 2012 года; 15 с наибольшим средним оборотом за 2012–2019.
4. Сплиты: дни |доходность| > 25% и даты из справочника сплитов ISS исключаются (доходность = NaN).
5. Для каждой акции — протокол IMOEX с теми же гиперпараметрами (summary.json).
6. Метрики, медианы, доли, Уилкоксон по акциям -> results/panel/.

Запуск: python financeML/panel.py
"""

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import wilcoxon

import vol_pilot as vp

ROOT = Path(__file__).parent
PANEL = ROOT / "data" / "panel"
RAW = PANEL / "raw"
RES = ROOT / "results" / "panel"
BASE = "https://iss.moex.com/iss"
TIMEOUT = 30
RETRIES = 3
MIN_COVER = 0.95
N_STOCKS = 15
SPLIT_THR = 0.25
PREFILTER_FROM = "2012-10-01"
PREFILTER_TILL = "2026-11-13"
TOKEN_HEADER = ["begin", "end", "open", "high", "low", "close", "value", "volume"]


def get_json(path, params):
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(BASE + path, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as error:
            last = error
            time.sleep(2 * attempt)
    raise RuntimeError(f"{path} {params}: {last}")


def listing():
    rows, start, cols = [], 0, None
    while True:
        j = get_json("/history/engines/stock/markets/shares/boards/TQBR/listing.json", {"start": start})["securities"]
        if not j["data"]:
            break
        cols = j["columns"]
        rows.extend(j["data"])
        start += len(j["data"])
    return pd.DataFrame(rows, columns=cols)


def download_candles(secid, year, till):
    path = RAW / f"{secid}_{year}.csv"
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > 1 and lines[-1].startswith(f"{year}-"):
            return len(lines) - 1
    rows, start = [], 0
    while True:
        payload = get_json(f"/engines/stock/markets/shares/boards/TQBR/securities/{secid}/candles.json",
                           {"interval": 24, "from": f"{year}-01-01", "till": till, "start": start})["candles"]
        page = payload["data"]
        if not page:
            break
        idx = {n: i for i, n in enumerate(payload["columns"])}
        rows.extend([[r[idx[k]] for k in TOKEN_HEADER] for r in page])
        start += len(page)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(TOKEN_HEADER)
        w.writerows(rows)
    return len(rows)


def load_secid(secid):
    frames = []
    for y in range(2012, date.today().year + 1):
        f = RAW / f"{secid}_{y}.csv"
        if f.exists():
            frames.append(pd.read_csv(f, parse_dates=["begin"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TOKEN_HEADER)


def imoex_calendar():
    raw, _ = vp.load_raw()
    cal = raw[["begin"]].rename(columns={"begin": "date"}).reset_index(drop=True)
    cal["date"] = pd.to_datetime(cal["date"])
    return cal


def find_gap_2022(traded):
    dates = pd.to_datetime(traded["begin"]).reset_index(drop=True)
    diffs = dates.diff().dt.days
    candidates = []
    for i in range(1, len(dates)):
        prev, nxt = dates.iloc[i - 1], dates.iloc[i]
        if prev >= pd.Timestamp("2022-02-01") and prev <= pd.Timestamp("2022-02-28") \
                and nxt >= pd.Timestamp("2022-03-01") and nxt <= pd.Timestamp("2022-04-30") \
                and diffs.iloc[i] >= 10:
            candidates.append((int(i), prev, nxt, int(diffs.iloc[i]) - 1))
    if not candidates:
        return {"last_trade_before": None, "first_trade_after": None, "calendar_days_missing": 0,
                "first_after_idx": None}
    i, prev, nxt, missing = max(candidates, key=lambda x: x[3])
    return {"last_trade_before": str(prev.date()), "first_trade_after": str(nxt.date()),
            "calendar_days_missing": missing, "first_after_idx": i}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    cal = imoex_calendar()
    n_cal = len(cal)

    lst = listing()
    lst.to_csv(PANEL / "tqbr_listing.csv", index=False)
    till_today = date.today().isoformat()
    max_missing = len(cal) - int(np.ceil(MIN_COVER * len(cal)))
    year_sizes = cal.assign(year=cal["date"].dt.year).groupby("year").size()
    min_first_year_days = max(1, int(year_sizes.loc[2012]) - max_missing)
    min_last_year_days = max(1, int(year_sizes.loc[date.today().year]) - max_missing)

    def year_till(y):
        return till_today if y == date.today().year else f"{y}-12-31"

    def trade_days_in_year(secid, y):
        n = download_candles(secid, y, year_till(y))
        if n == 0:
            return 0
        f = pd.read_csv(RAW / f"{secid}_{y}.csv")
        return int((f["value"] > 0).sum())

    pre = lst[lst["history_till"] >= PREFILTER_TILL]["SECID"].tolist()
    print(f"listing {len(lst)}, traded to {PREFILTER_TILL}: {len(pre)}, "
          f"minimum first/last year days {min_first_year_days}/{min_last_year_days}")

    def stage_a(secid):
        return secid, trade_days_in_year(secid, 2012), trade_days_in_year(secid, date.today().year)

    with ThreadPoolExecutor(max_workers=6) as ex:
        stage = list(ex.map(stage_a, pre))
    secids = [s for s, d12, dlast in stage
              if d12 >= min_first_year_days and dlast >= min_last_year_days]
    print(f"stage A survivors: {len(secids)}")

    def job(secid):
        out = [download_candles(secid, y, year_till(y)) for y in range(2012, date.today().year + 1)]
        return secid, sum(out)

    with ThreadPoolExecutor(max_workers=6) as ex:
        for secid, n in ex.map(job, secids):
            print(secid, n)

    stats = []
    series = {}
    for secid in secids:
        raw = load_secid(secid)
        raw["begin"] = pd.to_datetime(raw["begin"])
        raw = raw.drop_duplicates("begin").sort_values("begin").reset_index(drop=True)
        win = raw[(raw["begin"] >= "2022-02-01") & (raw["begin"] <= "2022-04-30") & (raw["value"] > 0)]
        span = pd.bdate_range("2022-02-25", "2022-03-23")
        extra_dates = pd.Series(list(win.loc[~win["begin"].isin(cal["date"]), "begin"]) + list(span))
        extra = pd.DataFrame({"date": extra_dates[~extra_dates.isin(cal["date"])].drop_duplicates()})
        cal_s = pd.concat([cal, extra], ignore_index=True).sort_values("date").reset_index(drop=True)
        merged = cal_s.merge(raw, left_on="date", right_on="begin", how="left")
        trade = merged["value"].fillna(0) > 0
        on_index = merged["date"].isin(cal["date"]).to_numpy()
        cover = trade.to_numpy()[on_index].sum() / n_cal
        m1219 = merged["date"].dt.year.between(2012, 2019) & trade
        mean_value = merged.loc[m1219, "value"].mean()
        stats.append({"SECID": secid, "trade_days": int(trade.to_numpy()[on_index].sum()), "coverage": cover,
                      "mean_value_2012_2019": mean_value})
        series[secid] = merged.assign(trade=trade).reset_index(drop=True)

    st = pd.DataFrame(stats).sort_values("mean_value_2012_2019", ascending=False)
    st.to_csv(RES / "candidates_coverage.csv", index=False)
    eligible = st[st["coverage"] >= MIN_COVER]
    selected = eligible.sort_values("mean_value_2012_2019", ascending=False).head(N_STOCKS)["SECID"].tolist()
    pd.DataFrame({"SECID": selected}).to_csv(RES / "selected_tickers.csv", index=False)
    print("eligible", len(eligible), "selected", selected)

    splits_ref = get_json("/statistics/engines/stock/splits.json", {})["splits"]
    splits = pd.DataFrame(splits_ref["data"], columns=splits_ref["columns"])
    splits["tradedate"] = pd.to_datetime(splits["tradedate"])

    params = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))["tuned_params"]

    events, gaps, per_stock = [], [], []
    for secid in selected:
        all_rows = load_secid(secid)
        all_rows["begin"] = pd.to_datetime(all_rows["begin"])
        all_rows = all_rows.drop_duplicates("begin").sort_values("begin").reset_index(drop=True)
        # As with IMOEX, omit today's still-open session if it is present.
        all_rows = all_rows.loc[all_rows["begin"] < pd.Timestamp(date.today())].reset_index(drop=True)
        zero_dates = all_rows.loc[all_rows["value"].fillna(0) <= 0, "begin"]
        traded = all_rows.loc[(all_rows["value"].fillna(0) > 0) & all_rows["close"].notna()].copy()
        traded = traded.reset_index(drop=True)
        r = np.log(traded["close"] / traded["close"].shift())
        # A row with no turnover is not a trading observation; break the return
        # from that date to the next active session rather than bridging it.
        for zdate in zero_dates:
            later = np.flatnonzero((traded["begin"] > zdate).to_numpy())
            if len(later):
                r.iloc[later[0]] = np.nan

        ev_ref = splits.loc[splits["secid"] == secid, "tradedate"]
        ref_mask = traded["begin"].isin(ev_ref).to_numpy()
        ret_simple = (traded["close"] / traded["close"].shift()) - 1
        ratio = traded["close"] / traded["close"].shift()
        big = (ret_simple.abs() > SPLIT_THR).to_numpy()
        scale_jump = ((ratio > 20) | (ratio < 1 / 20)).fillna(False).to_numpy()
        excluded = ref_mask | scale_jump
        for i in np.where(big | ref_mask | scale_jump)[0]:
            events.append({
                "SECID": secid,
                "date": str(traded["begin"].iloc[i].date()),
                "simple_return": None if pd.isna(ret_simple.iloc[i]) else float(ret_simple.iloc[i]),
                "log_return": None if pd.isna(r.iloc[i]) else float(r.iloc[i]),
                "in_moex_splits_reference": bool(ref_mask[i]),
                "over_25pct": bool(big[i]),
                "price_scale_jump_over_20x": bool(scale_jump[i]),
                "excluded_as_split_or_scale_error": bool(excluded[i]),
                "decision": "исключено: запись о сплите" if ref_mask[i]
                            else "исключено: скачок масштаба цены >20x" if scale_jump[i]
                            else "оставлено: нет признака сплита (в т.ч. рыночный шок)",
            })
        # Remove confirmed corporate actions and obvious price-scale breaks;
        # retain large genuine market returns (including the 2022 shock).
        r.iloc[np.flatnonzero(excluded)] = np.nan
        gap = find_gap_2022(traded)
        if gap["first_after_idx"] is not None:
            r.iloc[gap["first_after_idx"]] = np.nan
        gaps.append({"SECID": secid, **{k: v for k, v in gap.items() if k != "first_after_idx"},
                     "zero_turnover_rows_excluded": int(len(zero_dates))})

        d = pd.DataFrame({
            "date": traded["begin"], "r": r,
            "high": traded["high"], "low": traded["low"],
            "value": traded["value"],
        })
        d = vp.features(d)
        ok = d[vp.FEAT].notna().all(axis=1).to_numpy()
        pos = np.arange(len(d))
        first_eval = int(d.index[d["date"] >= vp.EVAL_START][0])
        first_test = int(d.index[d["date"] >= vp.TEST_START][0])

        preds = {}
        fits = {}
        for name, kind, mode, first in [
            ("fixed", "hgb", "fixed", first_eval),
            ("expanding", "hgb", "expanding", first_eval),
            ("rolling", "hgb", "rolling", first_eval),
            ("har", "har", "expanding", first_eval),
        ]:
            p, cnt = vp.run_mode(d, kind, mode, first, params)
            preds[name] = p
            fits[name] = cnt
        preds["naive"] = np.where(ok & (pos >= first_eval), d["lnrv5"].to_numpy(), np.nan)

        test = (d["date"] >= vp.TEST_START).to_numpy() & d["y"].notna().to_numpy()
        row = {"SECID": secid, "n_test": int(test.sum())}
        for name in ["fixed", "expanding", "rolling", "har", "naive"]:
            m_ = test & ~np.isnan(preds[name])
            row[f"QLIKE_{name}"] = vp.losses(d["y"][m_], preds[name][m_])["qlike"].mean()
            row[f"n_{name}"] = int(m_.sum())
        row["retrains_expanding"] = fits["expanding"]
        per_stock.append(row)

    pdf = pd.DataFrame(per_stock)
    pdf["rel_fixed_vs_expanding"] = pdf["QLIKE_fixed"] / pdf["QLIKE_expanding"] - 1
    pdf["rel_expanding_vs_rolling"] = pdf["QLIKE_expanding"] / pdf["QLIKE_rolling"] - 1
    pdf["rel_boosting_vs_har"] = pdf["QLIKE_expanding"] / pdf["QLIKE_har"] - 1
    pdf = pdf.merge(pd.DataFrame(gaps), on="SECID")
    pdf.to_csv(RES / "per_stock_metrics.csv", index=False)
    pd.DataFrame(events).to_csv(RES / "split_and_jump_events.csv", index=False)

    w = wilcoxon(pdf["QLIKE_fixed"], pdf["QLIKE_expanding"])
    w_one = wilcoxon(pdf["QLIKE_fixed"], pdf["QLIKE_expanding"], alternative="greater")
    k1_median = float(pdf["rel_fixed_vs_expanding"].median())
    k1 = bool(k1_median >= 0.10 and w.pvalue < 0.05)
    summary = {
        "tickers": selected,
        "median_rel_fixed_vs_expanding": k1_median,
        "share_expanding_better_than_fixed": float((pdf["QLIKE_expanding"] < pdf["QLIKE_fixed"]).mean()),
        "wilcoxon_two_sided_p": float(w.pvalue),
        "wilcoxon_one_sided_greater_p": float(w_one.pvalue),
        "median_rel_expanding_vs_rolling": float(pdf["rel_expanding_vs_rolling"].median()),
        "share_expanding_better_than_rolling": float((pdf["QLIKE_expanding"] < pdf["QLIKE_rolling"]).mean()),
        "median_rel_boosting_vs_har": float(pdf["rel_boosting_vs_har"].median()),
        "share_boosting_better_than_har": float((pdf["QLIKE_expanding"] < pdf["QLIKE_har"]).mean()),
        "K1_panel_pass": k1,
    }
    (RES / "panel_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(pdf.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
