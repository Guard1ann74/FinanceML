"""Дополнительные исследования на уже проведённом пилоте.

Из корня проекта: python financeML/research_addons.py
Переобучение гиперпараметров не выполняется. Для акций восстанавливаются
только дневные прогнозы: исходные panel-прогнозы по дням не были сохранены.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import panel as panel_code  # noqa: E402
import vol_pilot as vp  # noqa: E402

RES = ROOT / "results"
PRES = RES / "panel"
OUT = RES / "research"
FORECASTS = PRES / "forecasts"
MODELS = {
    "fixed": "Фиксированная (а)",
    "expanding": "Расширяющаяся (б)",
    "rolling": "Скользящая (в)",
    "har": "HAR",
    "naive": "Naive",
}
PAIRS = [
    ("fixed", "expanding", "Фиксированная — расширяющаяся"),
    ("expanding", "rolling", "Расширяющаяся — скользящая"),
    ("expanding", "har", "Бустинг — HAR"),
]
PERIODS = {
    "2022-01…2022-12": (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
    "2023 и позже": (pd.Timestamp("2023-01-01"), None),
}


def load_tickers():
    return pd.read_csv(PRES / "selected_tickers.csv")["SECID"].tolist()


def stock_features(secid):
    raw = panel_code.load_secid(secid)
    raw["begin"] = pd.to_datetime(raw["begin"])
    raw = raw.drop_duplicates("begin").sort_values("begin").reset_index(drop=True)
    download_date = pd.Timestamp(json.loads((ROOT / "data" / "download_meta.json").read_text(encoding="utf-8"))["download_date"])
    raw = raw.loc[raw["begin"] < download_date].reset_index(drop=True)

    zero_dates = raw.loc[raw["value"].fillna(0) <= 0, "begin"]
    traded = raw.loc[(raw["value"].fillna(0) > 0) & raw["close"].notna()].copy().reset_index(drop=True)
    r = np.log(traded["close"] / traded["close"].shift())
    for zero_date in zero_dates:
        after = np.flatnonzero((traded["begin"] > zero_date).to_numpy())
        if len(after):
            r.iloc[after[0]] = np.nan

    ret_ratio = traded["close"] / traded["close"].shift()
    scale_jump = ((ret_ratio > 20) | (ret_ratio < 1 / 20)).fillna(False).to_numpy()
    event_path = PRES / "split_and_jump_events.csv"
    events = pd.read_csv(event_path)
    split_dates = pd.to_datetime(events.loc[
        (events["SECID"] == secid) & events["in_moex_splits_reference"].fillna(False), "date"
    ])
    split_mask = traded["begin"].isin(split_dates).to_numpy()
    r.iloc[np.flatnonzero(scale_jump | split_mask)] = np.nan

    gap = panel_code.find_gap_2022(traded)
    if gap["first_after_idx"] is not None:
        r.iloc[gap["first_after_idx"]] = np.nan

    d = pd.DataFrame({
        "date": traded["begin"],
        "r": r,
        "close": traded["close"],
        "high": traded["high"],
        "low": traded["low"],
        "value": traded["value"],
        "volume": traded["volume"],
    })
    return vp.features(d), traded, gap, int(scale_jump.sum()), int(split_mask.sum()), int(len(zero_dates))


def get_predictions(secid, params):
    d, raw, gap, n_scale, n_split, n_zero = stock_features(secid)
    FORECASTS.mkdir(parents=True, exist_ok=True)
    cached = FORECASTS / f"{secid}.csv"
    if cached.exists():
        return pd.read_csv(cached, parse_dates=["date"]), raw, gap, {}, n_scale, n_split, n_zero
    first = int(d.index[d["date"] >= vp.EVAL_START][0])
    predictions = {}
    fit_counts = {}
    for name in ("fixed", "expanding", "rolling", "har"):
        kind = "har" if name == "har" else "hgb"
        mode = "expanding" if name == "har" else name
        p, count = vp.run_mode(d, kind, mode, first, params)
        predictions[name] = p
        fit_counts[name] = count
    ok = d[vp.FEAT].notna().all(axis=1).to_numpy()
    pos = np.arange(len(d))
    predictions["naive"] = np.where(ok & (pos >= first), d["lnrv5"].to_numpy(), np.nan)
    result = d[["date", "r", "y", "vol20", "value"]].copy()
    for name, values in predictions.items():
        result[name] = values
    result.to_csv(FORECASTS / f"{secid}.csv", index=False)
    return result, raw, gap, fit_counts, n_scale, n_split, n_zero


def qloss(y, pred):
    return vp.losses(y, pred)["qlike"]


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (n - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def spearman_bootstrap(x, y, rng, reps=10000):
    rho = float(spearmanr(x, y).statistic)
    n = len(x)
    vals = []
    for _ in range(reps):
        ix = rng.integers(0, n, size=n)
        if np.unique(x[ix]).size > 1 and np.unique(y[ix]).size > 1:
            vals.append(spearmanr(x[ix], y[ix]).statistic)
    return rho, np.quantile(vals, 0.025), np.quantile(vals, 0.975)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    params = json.loads((RES / "summary.json").read_text(encoding="utf-8"))["tuned_params"]
    tickers = load_tickers()

    # Restore only daily panel predictions, absent from the prior result files.
    panel_forecasts, diagnostics = {}, []
    for secid in tickers:
        f, raw, gap, fits, n_scale, n_split, n_zero = get_predictions(secid, params)
        panel_forecasts[secid] = f
        qret = np.log(raw["close"] / raw["close"].shift())
        zero_ret = qret.notna() & (qret.abs() < 1e-15)
        vol20 = np.sqrt(252 / 20 * qret.pow(2).rolling(20).sum())
        base = vol20[(raw["begin"] >= "2012-01-01") & (raw["begin"] <= "2019-12-31")]
        hi22 = vol20[(raw["begin"] >= "2022-01-01") & (raw["begin"] <= "2022-12-31")]
        diagnostics.append({
            "SECID": secid,
            "vol20_max_2022_over_p95_2012_2019": hi22.max() / base.quantile(0.95),
            "mean_value_2012_2019": raw.loc[raw["begin"].dt.year.between(2012, 2019), "value"].mean(),
            "zero_return_share_2012_2019": zero_ret[raw["begin"].dt.year.between(2012, 2019)].mean(),
            "zero_return_count_2012_2019": int(zero_ret[raw["begin"].dt.year.between(2012, 2019)].sum()),
            "split_scale_events_removed": n_scale,
            "official_split_events": n_split,
            "zero_turnover_rows": n_zero,
            "last_trade_before_2022_gap": gap["last_trade_before"],
            "first_trade_after_2022_gap": gap["first_trade_after"],
        })
    diag = pd.DataFrame(diagnostics)

    # PIKK diagnostics and ten largest daily QLIKE contributions for every model.
    pikk = panel_forecasts["PIKK"].copy()
    qrows = []
    for key, label in MODELS.items():
        m = pikk["date"].ge(vp.TEST_START) & pikk["y"].notna() & pikk[key].notna()
        losses = qloss(pikk.loc[m, "y"], pikk.loc[m, key])
        tmp = pikk.loc[m, ["date", "y", key, "r", "vol20"]].copy()
        tmp["QLIKE_contribution"] = losses
        tmp["model"] = label
        tmp = tmp.nlargest(10, "QLIKE_contribution")
        qrows.append(tmp.rename(columns={key: "forecast"}))
    pd.concat(qrows, ignore_index=True).to_csv(OUT / "pikk_top10_qlike_days.csv", index=False)

    p_raw = pd.read_csv(PRES / "per_stock_metrics.csv").set_index("SECID")
    pikk_events = pd.read_csv(PRES / "split_and_jump_events.csv")
    pikk_events = pikk_events[pikk_events["SECID"] == "PIKK"]
    pikk_test_mask = pikk["date"].ge(vp.TEST_START) & pikk["y"].notna()
    pikk_summary = {
        "test_qlike": {key: float(qloss(
            pikk.loc[pikk_test_mask & pikk[key].notna(), "y"],
            pikk.loc[pikk_test_mask & pikk[key].notna(), key]
        ).mean()) for key in MODELS if pikk[key].notna().sum() > 0},
        "n_rows": len(pikk),
        "n_zero_turnover": int((pikk["value"] <= 0).sum()),
        "zero_return_share_2012_2019": float(diag.set_index("SECID").loc["PIKK", "zero_return_share_2012_2019"]),
        "split_or_scale_events": pikk_events.to_dict(orient="records"),
        "max_abs_log_return": float(pikk["r"].abs().max()),
        "min_y": float(pikk["y"].min()),
        "max_y": float(pikk["y"].max()),
        "gap_last_before": str(diag.set_index("SECID").loc["PIKK", "last_trade_before_2022_gap"]),
        "gap_first_after": str(diag.set_index("SECID").loc["PIKK", "first_trade_after_2022_gap"]),
        "stored_qlike_panel": {k: float(p_raw.loc["PIKK", f"QLIKE_{k}"]) for k in ("fixed", "expanding", "rolling", "har", "naive")},
    }
    (OUT / "pikk_diagnostics.json").write_text(json.dumps(pikk_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    panel_rows = []
    for secid, f in panel_forecasts.items():
        for period, (start, end) in PERIODS.items():
            mask = f["date"] >= start
            if end is not None:
                mask &= f["date"] <= end
            mask &= f["y"].notna() & f["fixed"].notna() & f["expanding"].notna() & f["rolling"].notna() & f["har"].notna()
            row = {"SECID": secid, "period": period, "n": int(mask.sum())}
            for key in ("fixed", "expanding", "rolling", "har"):
                row[f"QLIKE_{key}"] = float(qloss(f.loc[mask, "y"], f.loc[mask, key]).mean())
            row["relative_fixed_vs_expanding"] = row["QLIKE_fixed"] / row["QLIKE_expanding"] - 1
            row["relative_expanding_vs_rolling"] = row["QLIKE_expanding"] / row["QLIKE_rolling"] - 1
            row["relative_expanding_vs_har"] = row["QLIKE_expanding"] / row["QLIKE_har"] - 1
            panel_rows.append(row)
    period_df = pd.DataFrame(panel_rows)
    period_df.to_csv(OUT / "period_metrics_by_stock.csv", index=False)

    period_summary = []
    for period in PERIODS:
        sub = period_df[period_df["period"] == period]
        rel = sub["relative_fixed_vs_expanding"].to_numpy()
        w = wilcoxon(sub["QLIKE_fixed"], sub["QLIKE_expanding"],
                     alternative="two-sided", method="auto")
        period_summary.append({
            "period": period,
            "median_relative_fixed_vs_expanding": float(np.median(rel)),
            "stocks_expanding_improved": int((rel > 0).sum()),
            "stocks_total": int(len(rel)),
            "wilcoxon_statistic_on_paired_QLIKE": float(w.statistic),
            "wilcoxon_p": float(w.pvalue),
            "median_relative_expanding_vs_rolling": float(sub["relative_expanding_vs_rolling"].median()),
            "median_relative_expanding_vs_har": float(sub["relative_expanding_vs_har"].median()),
        })
    pd.DataFrame(period_summary).to_csv(OUT / "period_summary.csv", index=False)

    # Comparison across panel with and without PIKK.
    comparison_rows = []
    for label, data in [("С PIKK", period_df[period_df.period == "2023 и позже"]),
                        ("Без PIKK", period_df[(period_df.period == "2023 и позже") & (period_df.SECID != "PIKK")])]:
        for name, column in [
            ("фиксированная vs расширяющаяся", "relative_fixed_vs_expanding"),
            ("расширяющаяся vs скользящая", "relative_expanding_vs_rolling"),
            ("расширяющийся бустинг vs HAR", "relative_expanding_vs_har"),
        ]:
            comparison_rows.append({
                "sample": label, "comparison": name,
                "median_relative_difference": float(data[column].median()),
                "first_strategy_better_n": int((data[column] < 0).sum()),
                "total_stocks": int(len(data)),
            })
    # Full test summaries directly from stored panel metrics, avoiding period-only substitution.
    full_metrics = p_raw.reset_index()
    for label, data in [("С PIKK", full_metrics), ("Без PIKK", full_metrics[full_metrics.SECID != "PIKK"])]:
        comparisons = [
            ("фиксированная vs расширяющаяся", "QLIKE_fixed", "QLIKE_expanding"),
            ("расширяющаяся vs скользящая", "QLIKE_expanding", "QLIKE_rolling"),
            ("расширяющийся бустинг vs HAR", "QLIKE_expanding", "QLIKE_har"),
        ]
        for name, a, b in comparisons:
            rel = data[a] / data[b] - 1
            comparison_rows.append({
                "sample": label, "comparison": name, "period": "тест целиком",
                "median_relative_difference": float(rel.median()),
                "first_strategy_better_n": int((rel < 0).sum()),
                "total_stocks": int(len(data)),
            })
    pd.DataFrame(comparison_rows).to_csv(OUT / "panel_medians_with_without_pikk.csv", index=False)

    # IMOEX, period sensitivity from stored predictions (no model rerun).
    idx = pd.read_csv(RES / "predictions_main.csv", parse_dates=["date"])
    idx_models = {
        "fixed": "Фиксированный бустинг (а)",
        "expanding": "Расширяющееся окно (б)",
    }
    idx_period = []
    for period, (start, end) in PERIODS.items():
        mask = idx["date"] >= start
        if end is not None:
            mask &= idx["date"] <= end
        mask &= idx["y"].notna() & idx[idx_models["fixed"]].notna() & idx[idx_models["expanding"]].notna()
        a = qloss(idx.loc[mask, "y"], idx.loc[mask, idx_models["fixed"]]).mean()
        b = qloss(idx.loc[mask, "y"], idx.loc[mask, idx_models["expanding"]]).mean()
        idx_period.append({"period": period, "n": int(mask.sum()), "QLIKE_fixed": a, "QLIKE_expanding": b,
                           "relative_fixed_vs_expanding": a / b - 1})
    pd.DataFrame(idx_period).to_csv(OUT / "imoex_period_metrics.csv", index=False)

    # Explanatory variables + Spearman with paired bootstrap interval.
    diag = diag.merge(p_raw.reset_index()[["SECID", "QLIKE_fixed", "QLIKE_expanding"]], on="SECID")
    diag["relative_fixed_vs_expanding"] = diag["QLIKE_fixed"] / diag["QLIKE_expanding"] - 1
    diag.to_csv(OUT / "stock_dispersion_and_effect.csv", index=False)
    rng = np.random.default_rng(20261116)
    corr_rows = []
    for column, label in [
        ("vol20_max_2022_over_p95_2012_2019", "Размер скачка волатильности"),
        ("mean_value_2012_2019", "Средний дневной оборот"),
        ("zero_return_share_2012_2019", "Доля нулевых доходностей"),
    ]:
        rho, lo, hi = spearman_bootstrap(diag[column].to_numpy(), diag["relative_fixed_vs_expanding"].to_numpy(), rng)
        p = spearmanr(diag[column], diag["relative_fixed_vs_expanding"]).pvalue
        corr_rows.append({"variable": label, "rho_spearman": rho, "ci95_boot_low": lo, "ci95_boot_high": hi,
                          "p_value_asymptotic": float(p), "n_stocks": len(diag), "bootstrap_reps": 10000})
    pd.DataFrame(corr_rows).to_csv(OUT / "spearman_correlations.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    xcols = [
        ("vol20_max_2022_over_p95_2012_2019", "Скачок волатильности"),
        ("mean_value_2012_2019", "Средний оборот 2012–2019"),
        ("zero_return_share_2012_2019", "Доля нулевых доходностей"),
    ]
    for ax, (col, title) in zip(axes, xcols):
        ax.scatter(diag[col], 100 * diag["relative_fixed_vs_expanding"], alpha=0.8)
        for row in diag.itertuples():
            ax.annotate(row.SECID, (getattr(row, col), 100 * row.relative_fixed_vs_expanding), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(title)
        ax.set_ylabel("Фиксированная хуже расширяющейся, %")
    fig.suptitle("Акции и относительная разница QLIKE — исследовательский анализ")
    fig.tight_layout()
    fig.savefig(OUT / "stock_effect_scatter.png", dpi=150)
    plt.close(fig)

    # Per-stock Diebold-Mariano tests; Holm separately within each comparison.
    dm_rows = []
    for secid, f in panel_forecasts.items():
        mask = (f["date"] >= vp.TEST_START) & f["y"].notna()
        for a, b, label in PAIRS:
            m = mask & f[a].notna() & f[b].notna()
            la, lb = qloss(f.loc[m, "y"], f.loc[m, a]), qloss(f.loc[m, "y"], f.loc[m, b])
            dm = vp.dm_test(la, lb, lag=5)
            dm_rows.append({"SECID": secid, "comparison": label, "n": int(m.sum()),
                            "mean_loss_diff_A_minus_B": dm["mean_diff"], "dm_stat": dm["dm_stat"],
                            "p_raw": dm["p_value"]})
    dm_df = pd.DataFrame(dm_rows)
    dm_df["p_holm"] = np.nan
    for _, group in dm_df.groupby("comparison"):
        dm_df.loc[group.index, "p_holm"] = holm_adjust(group["p_raw"].to_numpy())
    dm_df["significant_holm_0_05"] = dm_df["p_holm"] < 0.05
    dm_df.to_csv(OUT / "dm_by_stock_holm.csv", index=False)
    dm_df.groupby("comparison").agg(
        significant_after_holm=("significant_holm_0_05", "sum"),
        stocks=("SECID", "count"),
    ).reset_index().to_csv(OUT / "dm_holm_summary.csv", index=False)

    # Dynamic QLIKE for IMOEX and six stocks at extremes of stored fixed-vs-expanding effect.
    effect = p_raw.copy()
    effect["relative"] = effect["QLIKE_fixed"] / effect["QLIKE_expanding"] - 1
    ordered = effect["relative"].sort_values()
    extreme = ordered.index[:3].tolist() + ordered.index[-3:].tolist()
    chosen = list(dict.fromkeys(extreme))
    (OUT / "dynamic_graph_tickers.json").write_text(
        json.dumps({"lowest_effect": ordered.index[:3].tolist(), "highest_effect": ordered.index[-3:].tolist(),
                    "selected": chosen}, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(4, 2, figsize=(15, 16), sharex=True)
    plot_series = [("IMOEX", idx, {
        "fixed": idx_models["fixed"], "expanding": idx_models["expanding"],
        "rolling": "Скользящее окно 3 года (в)", "har": "HAR (расширяющееся, переобучение 20 дн.)",
    })]
    for secid in chosen:
        plot_series.append((secid, panel_forecasts[secid], {k: k for k in ("fixed", "expanding", "rolling", "har")}))
    left, right = pd.Timestamp("2021-09-01"), pd.Timestamp("2022-09-30")
    for ax, (label, f, colmap) in zip(axes.flat, plot_series):
        m = f["y"].notna()
        for key, col in colmap.items():
            valid = m & f[col].notna()
            err = pd.Series(qloss(f.loc[valid, "y"], f.loc[valid, col]),
                            index=f.loc[valid, "date"]).rolling(60, min_periods=60).mean()
            in_window = (err.index >= left) & (err.index <= right)
            ax.plot(err.index[in_window], err.to_numpy()[in_window], label=MODELS[key])
        ax.axvline(pd.Timestamp("2022-03-24"), color="black", linestyle="--", lw=0.8)
        ax.set_title(label)
        ax.legend(fontsize=7)
    axes.flat[-1].axis("off")
    fig.suptitle("60-дневная средняя QLIKE вокруг 24.03.2022 — исследовательский анализ")
    fig.tight_layout()
    fig.savefig(OUT / "rolling60_qlike_regime_transition.png", dpi=150)
    plt.close(fig)

    # Consolidated results for the report.
    summary = {
        "analysis_label": "исследовательский анализ, не проверка гипотезы",
        "tickers": tickers,
        "pikk": pikk_summary,
        "periods": period_summary,
        "spearman": corr_rows,
        "dm_holm": pd.read_csv(OUT / "dm_holm_summary.csv").to_dict(orient="records"),
        "dynamic_tickers": chosen,
        "panel_forecasts_reconstructed": True,
    }
    (OUT / "research_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({
        "pikk": pikk_summary,
        "period_summary": period_summary,
        "spearman": corr_rows,
        "dm_holm": summary["dm_holm"],
        "dynamic_tickers": chosen,
    }, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
