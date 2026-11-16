"""Исследовательское сравнение панели с полным и IMOEX-календарём.

Команда из корня проекта: python financeML/calendar_panel.py
Использует старые панельные прогнозы, чтобы диагностировать не-IMOEX-дни,
и пересчитывает модели только на совпадающих календарных датах. Параметры
модели читаются из сохранённого results/summary.json; подбор не повторяется.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import panel as panel_code  # noqa: E402
import vol_pilot as vp  # noqa: E402

RES = ROOT / "results"
PRES = RES / "panel"
OUT = RES / "calendar"
OLD_FORECASTS = PRES / "forecasts"
NEW_FORECASTS = PRES / "calendar_forecasts"
MODEL_KEYS = ["fixed", "expanding", "rolling", "har", "naive"]
MODEL_LABELS = {
    "fixed": "Фиксированная",
    "expanding": "Расширяющаяся",
    "rolling": "Скользящая",
    "har": "HAR",
    "naive": "Naive",
}
PAIRS = [
    ("fixed", "expanding", "Фиксированная — расширяющаяся"),
    ("expanding", "rolling", "Расширяющаяся — скользящая"),
    ("expanding", "har", "Бустинг — HAR"),
]
PERIODS = {
    "Тест 2022+": (pd.Timestamp("2022-01-01"), None),
    "2022": (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
    "2023 и позже": (pd.Timestamp("2023-01-01"), None),
}


def load_calendar():
    idx = pd.read_csv(RES / "predictions_main.csv", parse_dates=["date"])
    return set(idx["date"].dt.normalize()), idx


def qlike(y, yhat):
    return vp.losses(np.asarray(y, float), np.asarray(yhat, float))["qlike"]


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def prepare_stock(secid, calendar):
    raw = panel_code.load_secid(secid)
    raw["begin"] = pd.to_datetime(raw["begin"]).dt.normalize()
    raw = raw.drop_duplicates("begin").sort_values("begin").reset_index(drop=True)
    download_date = pd.Timestamp(json.loads(
        (ROOT / "data" / "download_meta.json").read_text(encoding="utf-8")
    )["download_date"]).normalize()
    raw = raw.loc[(raw["begin"] < download_date) & raw["begin"].isin(calendar)].copy()
    raw = raw.reset_index(drop=True)

    zero_dates = raw.loc[raw["value"].fillna(0) <= 0, "begin"]
    traded = raw.loc[(raw["value"].fillna(0) > 0) & raw["close"].notna()].copy().reset_index(drop=True)
    returns = np.log(traded["close"] / traded["close"].shift())
    for zdate in zero_dates:
        next_rows = np.flatnonzero((traded["begin"] > zdate).to_numpy())
        if len(next_rows):
            returns.iloc[next_rows[0]] = np.nan

    # Preserve the already-reviewed corporate-action and price-scale exclusions.
    actions = pd.read_csv(PRES / "split_and_jump_events.csv")
    excluded_dates = pd.to_datetime(actions.loc[
        (actions["SECID"] == secid) &
        actions.get("excluded_as_split_or_scale_error", pd.Series(False, index=actions.index)).fillna(False),
        "date",
    ]).dt.normalize()
    action_mask = traded["begin"].isin(excluded_dates).to_numpy()
    returns.iloc[np.flatnonzero(action_mask)] = np.nan

    gap = panel_code.find_gap_2022(traded)
    if gap["first_after_idx"] is not None:
        returns.iloc[gap["first_after_idx"]] = np.nan

    d = pd.DataFrame({
        "date": traded["begin"],
        "r": returns,
        "close": traded["close"],
        "high": traded["high"],
        "low": traded["low"],
        "value": traded["value"],
        "volume": traded["volume"],
    })
    return vp.features(d), gap


def restore_calendar_forecasts(secid, params, calendar, cached_old):
    d, gap = prepare_stock(secid, calendar)
    cache_path = NEW_FORECASTS / f"{secid}.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        return cached, gap, {}
    first = int(d.index[d["date"] >= vp.EVAL_START][0])
    pred = {}
    counts = {}
    for name in ("fixed", "expanding", "rolling", "har"):
        kind = "har" if name == "har" else "hgb"
        mode = "expanding" if name == "har" else name
        pred[name], counts[name] = vp.run_mode(d, kind, mode, first, params)
    ok = d[vp.FEAT].notna().all(axis=1).to_numpy()
    pred["naive"] = np.where(ok & (np.arange(len(d)) >= first), d["lnrv5"].to_numpy(), np.nan)

    f = d[["date", "r", "y", "vol20", "value"]].copy()
    for name, values in pred.items():
        f[name] = values
    NEW_FORECASTS.mkdir(parents=True, exist_ok=True)
    f.to_csv(cache_path, index=False)
    return f, gap, counts


def valid_mask(df, period):
    start, end = PERIODS[period]
    mask = df["date"].ge(start) & df["y"].notna()
    if end is not None:
        mask &= df["date"].le(end)
    return mask


def per_ticker_metrics(old, new, tickers):
    rows = []
    for secid in tickers:
        o = old[secid]
        n = new[secid]
        for period in PERIODS:
            for sample, df in (("Исходный календарь", o), ("Календарь IMOEX", n)):
                m = valid_mask(df, period)
                row = {"SECID": secid, "sample": sample, "period": period, "n_test_days": int(m.sum())}
                for model in MODEL_KEYS:
                    mm = m & df[model].notna()
                    row[f"QLIKE_{model}"] = float(qlike(df.loc[mm, "y"], df.loc[mm, model]).mean())
                    row[f"n_{model}"] = int(mm.sum())
                row["rel_fixed_vs_expanding"] = row["QLIKE_fixed"] / row["QLIKE_expanding"] - 1
                row["rel_expanding_vs_rolling"] = row["QLIKE_expanding"] / row["QLIKE_rolling"] - 1
                row["rel_expanding_vs_har"] = row["QLIKE_expanding"] / row["QLIKE_har"] - 1
                rows.append(row)
    return pd.DataFrame(rows)


def aggregate_metrics(metric_df):
    rows = []
    for (sample, period), whole_group in metric_df.groupby(["sample", "period"], sort=False):
        for exclude_pikk in (False, True):
            group = whole_group if not exclude_pikk else whole_group[whole_group["SECID"] != "PIKK"]
            sample_name = sample if not exclude_pikk else f"{sample}, без PIKK"
            for comparison, col, a, b in [
                ("Фиксированная — расширяющаяся", "rel_fixed_vs_expanding", "QLIKE_fixed", "QLIKE_expanding"),
                ("Расширяющаяся — скользящая", "rel_expanding_vs_rolling", "QLIKE_expanding", "QLIKE_rolling"),
                ("Бустинг — HAR", "rel_expanding_vs_har", "QLIKE_expanding", "QLIKE_har"),
            ]:
                diffs = group[col].to_numpy()
                w = wilcoxon(group[a], group[b], alternative="two-sided", method="auto")
                rows.append({
                    "sample": sample_name, "period": period, "comparison": comparison,
                    "median_relative_difference": float(np.median(diffs)),
                    "first_A_better_n": int((diffs < 0).sum()),
                    "second_B_better_n": int((diffs > 0).sum()),
                    "n_stocks": int(len(group)),
                    "wilcoxon_statistic": float(w.statistic),
                    "wilcoxon_p": float(w.pvalue),
                })
    return pd.DataFrame(rows)


def dm_tables(old, new, tickers):
    rows = []
    for sample, forecast_map in (("Исходный календарь", old), ("Календарь IMOEX", new)):
        for exclude_pikk in (False, True):
            sample_tickers = [s for s in tickers if not (exclude_pikk and s == "PIKK")]
            sample_name = sample if not exclude_pikk else f"{sample}, без PIKK"
            for secid in sample_tickers:
                df = forecast_map[secid]
                for period in PERIODS:
                    base = valid_mask(df, period)
                    for a, b, label in PAIRS:
                        m = base & df[a].notna() & df[b].notna()
                        la = qlike(df.loc[m, "y"], df.loc[m, a])
                        lb = qlike(df.loc[m, "y"], df.loc[m, b])
                        dm = vp.dm_test(la, lb, lag=5)
                        rows.append({
                            "SECID": secid, "sample": sample_name, "period": period, "comparison": label,
                            "n": int(m.sum()), "mean_loss_diff_A_minus_B": dm["mean_diff"],
                            "dm_stat": dm["dm_stat"], "p_raw": dm["p_value"],
                        })
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    # Family-wise correction within each comparison and sample/period; 15 or 14 assets.
    for _, group in df.groupby(["sample", "period", "comparison"]):
        df.loc[group.index, "p_holm"] = holm(group["p_raw"].to_numpy())
    df["significant_holm_0_05"] = df["p_holm"] < 0.05
    summary = df.groupby(["sample", "period", "comparison"], as_index=False).agg(
        significant_after_holm=("significant_holm_0_05", "sum"),
        n_stocks=("SECID", "count"),
    )
    return df, summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    calendar, idx = load_calendar()
    tickers = pd.read_csv(PRES / "selected_tickers.csv")["SECID"].tolist()
    params = json.loads((RES / "summary.json").read_text(encoding="utf-8"))["tuned_params"]

    old, new = {}, {}
    for secid in tickers:
        old_path = OLD_FORECASTS / f"{secid}.csv"
        if not old_path.exists():
            raise FileNotFoundError(f"Нет исходного сохранённого прогноза: {old_path}")
        old[secid] = pd.read_csv(old_path, parse_dates=["date"])
        new[secid], gap, _ = restore_calendar_forecasts(secid, params, calendar, old[secid])
        print(secid, "old n", int(old[secid].loc[old[secid].date >= vp.TEST_START, "expanding"].notna().sum()),
              "IMOEX n", int(new[secid].loc[new[secid].date >= vp.TEST_START, "expanding"].notna().sum()),
              "gap", gap.get("last_trade_before"), gap.get("first_trade_after"))

    # 1) Weekday counts, off-calendar counts, and comparable IMOEX date counts.
    imoex_dates = idx.loc[idx["y"].notna() & idx["Расширяющееся окно (б)"].notna(), "date"].dt.normalize()
    weekday_rows = []
    for period, (start, end) in PERIODS.items():
        im = (imoex_dates >= start) & (True if end is None else (imoex_dates <= end))
        weekday_rows.append({"ticker": "IMOEX", "period": period, "n_test_days": int(im.sum()),
                             "not_in_imoex_calendar": 0, "weekday": "ALL"})
        for wd in range(7):
            weekday_rows.append({"ticker": "IMOEX", "period": period, "n_test_days": int((im & (imoex_dates.dt.dayofweek == wd)).sum()),
                                 "not_in_imoex_calendar": 0, "weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][wd]})
        for secid in tickers:
            f = old[secid]
            m = valid_mask(f, period) & f["expanding"].notna()
            dates = f.loc[m, "date"].dt.normalize()
            extra = ~dates.isin(calendar)
            weekday_rows.append({"ticker": secid, "period": period, "n_test_days": int(m.sum()),
                                 "not_in_imoex_calendar": int(extra.sum()), "weekday": "ALL"})
            for wd in range(7):
                wdmask = dates.dt.dayofweek == wd
                weekday_rows.append({
                    "ticker": secid, "period": period,
                    "n_test_days": int(wdmask.sum()),
                    "not_in_imoex_calendar": int((wdmask & extra.to_numpy()).sum()),
                    "weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][wd],
                })
    pd.DataFrame(weekday_rows).to_csv(OUT / "weekday_and_off_calendar_counts.csv", index=False)

    # 2) Per-stock old/new metrics and aggregate medians/shares/Wilcoxon.
    metric_df = per_ticker_metrics(old, new, tickers)
    metric_df.to_csv(OUT / "per_stock_old_new_metrics.csv", index=False)
    aggregate = aggregate_metrics(metric_df)
    aggregate.to_csv(OUT / "old_new_aggregate_comparison.csv", index=False)

    # 3) Holm-corrected DM tests per ticker, both calendars and date periods.
    dm_df, dm_summary = dm_tables(old, new, tickers)
    dm_df.to_csv(OUT / "dm_by_stock_holm.csv", index=False)
    dm_summary.to_csv(OUT / "dm_holm_summary.csv", index=False)

    # 4) QLIKE share from stock dates absent in IMOEX calendar.
    extra_rows = []
    for period in PERIODS:
        for model in MODEL_KEYS:
            total, extra = 0.0, 0.0
            by_ticker_total, by_ticker_extra = [], []
            for secid in tickers:
                f = old[secid]
                m = valid_mask(f, period) & f[model].notna()
                loss = qlike(f.loc[m, "y"], f.loc[m, model])
                off = ~f.loc[m, "date"].dt.normalize().isin(calendar).to_numpy()
                extra += float(loss[off].sum())
                total += float(loss.sum())
                by_ticker_total.extend(loss.tolist())
                by_ticker_extra.extend(loss[off].tolist())
            extra_rows.append({
                "period": period, "model": model, "sum_QLIKE": total,
                "sum_QLIKE_off_calendar": extra,
                "share_sum_QLIKE_off_calendar": extra / total if total else np.nan,
            })
    pd.DataFrame(extra_rows).to_csv(OUT / "off_calendar_qlike_share.csv", index=False)

    # Old/new comparison table at model-level and aggregate level.
    old_new = aggregate.copy()
    old_new.to_csv(OUT / "old_new_comparison.csv", index=False)

    # Summary JSON for the report.
    selected = {"tickers": tickers, "calendar_days": len(calendar)}
    (OUT / "calendar_summary.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE", OUT)


if __name__ == "__main__":
    main()
