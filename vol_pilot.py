"""Пилот: прогноз волатильности IMOEX тремя способами обновления модели.

Запуск (после download.py): python financeML/vol_pilot.py
Результаты: financeML/results/
"""

import itertools
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "imoex_daily_raw.csv"
META = ROOT / "data" / "download_meta.json"
RES = ROOT / "results"

H = 5
GAP = 5
TRAIN_LAG = H + GAP + 1
RETRAIN = 20
ROLL_YEARS = 3
BOOT_BLOCK = 10
BOOT_REPS = 2000
SEED = 42

GAP_PREV = pd.Timestamp("2022-02-25")
GAP_NEXT = pd.Timestamp("2022-03-24")
SHOCK = pd.Timestamp("2022-02-24")
EVAL_START = pd.Timestamp("2020-01-01")
VAL_END = pd.Timestamp("2021-12-31")
TEST_START = pd.Timestamp("2022-01-01")

FEAT = ["lnrv1", "lnrv5", "lnrv22", "lnpark5", "vchg"]
HAR_FEAT = ["lnrv1", "lnrv5", "lnrv22"]
GRID = {"learning_rate": [0.03, 0.1], "max_depth": [2, 3], "max_iter": [100, 300], "min_samples_leaf": [20, 50]}

VARIANTS = {"основной (разрыв исключён)": False, "робастный (разрыв оставлен)": True}


def ln_rv(sum_sq, n):
    return 0.5 * np.log(np.maximum(252.0 / n * sum_sq, 1e-12))


def load_raw():
    raw = pd.read_csv(DATA, parse_dates=["begin"])
    download_date = pd.Timestamp(json.loads(META.read_text(encoding="utf-8"))["download_date"])
    partial = raw["begin"] == download_date
    raw = raw.loc[~partial].sort_values("begin").reset_index(drop=True)
    return raw, int(partial.sum())


def build(raw, keep_gap):
    d = raw.copy()
    d["date"] = d["begin"]
    d["r"] = np.log(d["close"] / d["close"].shift())
    if not keep_gap:
        d.loc[d["date"] == GAP_NEXT, "r"] = np.nan
    return features(d)


def features(d):
    d = d.copy()
    r2 = d["r"] ** 2
    d["lnrv1"] = ln_rv(r2.rolling(1).sum(), 1)
    d["lnrv5"] = ln_rv(r2.rolling(5).sum(), 5)
    d["lnrv22"] = ln_rv(r2.rolling(22).sum(), 22)
    hl2 = np.log(d["high"] / d["low"]) ** 2
    d["lnpark5"] = 0.5 * np.log(np.maximum(252 / (4 * np.log(2) * 5) * hl2.rolling(5).sum(), 1e-12))
    d["vchg"] = np.log(d["value"].rolling(5).mean() / d["value"].rolling(22).mean())
    d["y"] = ln_rv(r2.rolling(H).sum().shift(-H), H)
    d["vol20"] = np.sqrt(252 / 20 * r2.rolling(20).sum())
    return d


def fit_predict(kind, x_tr, y_tr, x_te, params):
    if kind == "har":
        model = LinearRegression().fit(x_tr[HAR_FEAT], y_tr)
        return model.predict(x_te[HAR_FEAT])
    model = HistGradientBoostingRegressor(random_state=0, early_stopping=False, **params)
    model.fit(x_tr[FEAT], y_tr)
    return model.predict(x_te[FEAT])


def run_mode(d, kind, mode, first_pos, params):
    n = len(d)
    pos = np.arange(n)
    ok = d[FEAT].notna().all(axis=1).to_numpy()
    yok = d["y"].notna().to_numpy()
    pred = np.full(n, np.nan)
    if mode == "fixed":
        blocks = [(first_pos, n)]
    else:
        blocks = [(s, min(s + RETRAIN, n)) for s in range(first_pos, n, RETRAIN)]
    fits = 0
    for s, e in blocks:
        last = s - TRAIN_LAG
        train = ok & yok & (pos <= last)
        if mode == "rolling":
            lower = d["date"].iloc[last] - pd.DateOffset(years=ROLL_YEARS)
            train &= (d["date"] > lower).to_numpy()
        test = ok & (pos >= s) & (pos < e)
        if test.sum() == 0 or train.sum() == 0:
            continue
        pred[test] = fit_predict(kind, d.loc[train], d.loc[train, "y"], d.loc[test], params)
        fits += 1
    return pred, fits


def losses(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    v = np.exp(2 * y)
    h = np.exp(2 * yhat)
    ratio = v / h
    return {
        "qlike": ratio - np.log(ratio) - 1,
        "mse": (v - h) ** 2,
        "mae": np.abs(y - yhat),
    }


def tune(d, first_val, first_test):
    pos = np.arange(len(d))
    yok = d["y"].notna().to_numpy()
    rows = []
    for vals in itertools.product(*GRID.values()):
        params = dict(zip(GRID.keys(), vals))
        pred, _ = run_mode(d, "hgb", "fixed", first_val, params)
        m = (pos >= first_val) & (pos < first_test) & yok & ~np.isnan(pred)
        rows.append({**params, "val_qlike": losses(d["y"][m], pred[m])["qlike"].mean(), "val_n": int(m.sum())})
    table = pd.DataFrame(rows)
    best = table.loc[table["val_qlike"].idxmin()]
    params = {k: (int(best[k]) if k in ("max_depth", "max_iter", "min_samples_leaf") else float(best[k]))
              for k in GRID}
    return params, table


def dm_test(loss_a, loss_b, lag=H):
    diff = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(diff)
    mean = diff.mean()
    d0 = diff - mean
    var = np.sum(d0 * d0) / n
    for j in range(1, lag + 1):
        gamma = np.sum(d0[j:] * d0[:-j]) / n
        var += 2 * (1 - j / (lag + 1)) * gamma
    stat = mean / math.sqrt(var / n)
    p = math.erfc(abs(stat) / math.sqrt(2))
    return {"mean_diff": mean, "dm_stat": stat, "p_value": p, "n": n}


def block_bootstrap_ratio(loss_a, loss_b, rng):
    a = np.asarray(loss_a)
    b = np.asarray(loss_b)
    n = len(a)
    starts_max = n - BOOT_BLOCK
    ratios = np.empty(BOOT_REPS)
    n_blocks = math.ceil(n / BOOT_BLOCK)
    for i in range(BOOT_REPS):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(BOOT_BLOCK)[None, :]).ravel()[:n]
        ratios[i] = a[idx].mean() / b[idx].mean() - 1
    return np.percentile(ratios, 2.5), np.percentile(ratios, 97.5)


def main():
    RES.mkdir(exist_ok=True)
    raw, dropped = load_raw()

    main_d = build(raw, keep_gap=False)
    thr = main_d.loc[(main_d["date"] >= "2012-01-01") & (main_d["date"] <= "2019-12-31"), "vol20"].quantile(0.75)

    gap_prev_close = raw.loc[raw["begin"] == GAP_PREV, "close"].item()
    gap_next_close = raw.loc[raw["begin"] == GAP_NEXT, "close"].item()
    shock_ret = np.log(raw.loc[raw["begin"] == SHOCK, "close"].item() / raw.loc[raw["begin"] < SHOCK, "close"].iloc[-1])
    zero_vol = int((raw["volume"] == 0).sum())
    zero_val = int((raw["value"] <= 0).sum())
    value_std = float(raw["value"].std())
    value_all_zero = bool((raw["value"] == 0).all())
    zero_returns = int((np.log(raw["close"] / raw["close"].shift()).abs() < 1e-15).sum())

    checks = {
        "rows_after_drop": len(raw),
        "dropped_partial_last_row": dropped,
        "first_date": str(raw["begin"].min().date()),
        "last_date": str(raw["begin"].max().date()),
        "duplicate_dates": int(raw["begin"].duplicated().sum()),
        "null_cells": int(raw.isna().sum().sum()),
        "zero_volume_rows": zero_vol,
        "zero_value_rows": zero_val,
        "value_std": value_std,
        "value_all_zero": value_all_zero,
        "value_feature_used": bool((not value_all_zero) and value_std > 0),
        "zero_return_rows": zero_returns,
        "gap_last_day_before": str(GAP_PREV.date()),
        "gap_close_before": gap_prev_close,
        "gap_first_day_after": str(GAP_NEXT.date()),
        "gap_close_after": gap_next_close,
        "gap_log_return": float(np.log(gap_next_close / gap_prev_close)),
        "shock_day": str(SHOCK.date()),
        "shock_log_return": float(shock_ret),
        "high_vol_threshold_vol20_p75_2012_2019": float(thr),
    }

    shock_pos = int(main_d.index[main_d["date"] == SHOCK][0])
    first_eval = int(main_d.index[main_d["date"] >= EVAL_START][0])
    first_test = int(main_d.index[main_d["date"] >= TEST_START][0])
    first_2023 = int(main_d.index[main_d["date"] >= pd.Timestamp("2023-01-01")][0])

    params, tune_table = tune(main_d, first_eval, first_test)
    tune_table.to_csv(RES / "tuning_val_2020_2021.csv", index=False)

    all_metrics, all_dm, all_boot = [], [], []
    preds_out = {}
    rng = np.random.default_rng(SEED)
    k1 = {}

    for vname, keep_gap in VARIANTS.items():
        d = build(raw, keep_gap=keep_gap)
        n = len(d)
        pos = np.arange(n)
        specs = [
            ("Фиксированный бустинг (а)", "hgb", "fixed", first_eval),
            ("Расширяющееся окно (б)", "hgb", "expanding", first_eval),
            ("Скользящее окно 3 года (в)", "hgb", "rolling", first_eval),
            ("HAR (расширяющееся, переобучение 20 дн.)", "har", "expanding", first_eval),
            ("Naive (RV за 5 дней)", "naive", None, first_eval),
            ("Фиксированный а2 (обучение 2012–2021)", "hgb", "fixed", first_test),
        ]
        out = pd.DataFrame({"date": d["date"], "y": d["y"],
                            "regime": np.where(d["vol20"].isna(), "нет", np.where(d["vol20"] > thr, "высокий", "обычный"))})
        fits = {}
        for name, kind, mode, first in specs:
            if kind == "naive":
                ok = d[FEAT].notna().all(axis=1).to_numpy()
                pred = np.where(ok & (pos >= first), d["lnrv5"].to_numpy(), np.nan)
                fits[name] = 0
            else:
                pred, cnt = run_mode(d, kind, mode, first, params)
                fits[name] = cnt
            out[name] = pred

        out.to_csv(RES / f"predictions_{'robust' if keep_gap else 'main'}.csv", index=False)
        preds_out[vname] = out

        excl_shock = ~((pos >= shock_pos - H) & (pos <= shock_pos - 1))
        date = out["date"]
        subsets = {
            "2020–2021 (подбор гиперпараметров)": (date >= EVAL_START) & (date <= VAL_END),
            "ТЕСТ 2022+ (все)": date >= TEST_START,
            "ТЕСТ обычный режим": (date >= TEST_START) & (out["regime"] == "обычный"),
            "ТЕСТ высокий режим": (date >= TEST_START) & (out["regime"] == "высокий"),
            "2022-02…2022-04": (date >= "2022-02-01") & (date <= "2022-04-30"),
            "2022-05…2022-12": (date >= "2022-05-01") & (date <= "2022-12-31"),
            "2023 и позже": date >= pd.Timestamp("2023-01-01"),
            "ТЕСТ без целей с 2022-02-24 (доп. проверка)": (date >= TEST_START) & excl_shock,
        }

        for name, *_ in specs:
            for sname, mask in subsets.items():
                m = mask.to_numpy() & out["y"].notna().to_numpy() & out[name].notna().to_numpy()
                if m.sum() == 0:
                    continue
                L = losses(out["y"][m], out[name][m])
                all_metrics.append({
                    "variant": vname, "subset": sname, "model": name, "n": int(m.sum()),
                    "QLIKE": L["qlike"].mean(), "MSE_variance": L["mse"].mean(),
                    "MAE_log_vol": L["mae"].mean(), "retrains_total": fits[name],
                })

        test_mask = (date >= TEST_START).to_numpy() & out["y"].notna().to_numpy()
        pairs = [
            ("Фиксированный бустинг (а)", "Расширяющееся окно (б)"),
            ("Расширяющееся окно (б)", "Скользящее окно 3 года (в)"),
            ("Фиксированный бустинг (а)", "Скользящее окно 3 года (в)"),
            ("Расширяющееся окно (б)", "HAR (расширяющееся, переобучение 20 дн.)"),
            ("Расширяющееся окно (б)", "Naive (RV за 5 дней)"),
        ]
        for a, b in pairs:
            m = test_mask & out[a].notna().to_numpy() & out[b].notna().to_numpy()
            la = losses(out["y"][m], out[a][m])["qlike"]
            lb = losses(out["y"][m], out[b][m])["qlike"]
            dm = dm_test(la, lb)
            lo, hi = block_bootstrap_ratio(la, lb, rng)
            row = {"variant": vname, "A": a, "B": b, "QLIKE_A": la.mean(), "QLIKE_B": lb.mean(),
                   "relative_diff_A_vs_B": la.mean() / lb.mean() - 1, "ci95_low": lo, "ci95_high": hi, **dm}
            all_dm.append(row)
            if a == "Фиксированный бустинг (а)" and b == "Расширяющееся окно (б)":
                k1[vname] = row

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(RES / "metrics.csv", index=False)
    dm_df = pd.DataFrame(all_dm)
    dm_df.to_csv(RES / "dm_and_bootstrap.csv", index=False)

    main_key = "основной (разрыв исключён)"
    k = k1[main_key]
    k1_pass = bool(k["relative_diff_A_vs_B"] >= 0.25 and k["p_value"] < 0.05 and k["mean_diff"] > 0)
    summary = {"checks": checks, "tuned_params": params, "K1": {
        "QLIKE_fixed_test": k["QLIKE_A"], "QLIKE_expanding_test": k["QLIKE_B"],
        "relative_diff": k["relative_diff_A_vs_B"], "ci95": [k["ci95_low"], k["ci95_high"]],
        "dm_stat": k["dm_stat"], "p_value": k["p_value"], "pass": k1_pass}}
    (RES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(12, 6))
    o = preds_out[main_key]
    t = o[o["date"] >= TEST_START]
    ax.plot(t["date"], t["y"], color="black", lw=1.2, label="факт (y)")
    for name in ["Фиксированный бустинг (а)", "Расширяющееся окно (б)", "Скользящее окно 3 года (в)",
                 "HAR (расширяющееся, переобучение 20 дн.)"]:
        ax.plot(t["date"], t[name], lw=1, label=name)
    ax.axvspan(pd.Timestamp("2022-02-25"), pd.Timestamp("2022-03-24"), color="grey", alpha=0.2, label="разрыв торгов")
    ax.set_title("Прогноз и факт: ln годовой реализованной волатильности (тест с 2022-01-01)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "forecast_vs_actual_main.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for name in ["Фиксированный бустинг (а)", "Расширяющееся окно (б)", "Скользящее окно 3 года (в)"]:
        loss = losses(o["y"], o[name])["qlike"]
        loss = pd.Series(loss, index=o.index).rolling(60, min_periods=60).mean()
        mask = (o["date"] >= "2021-09-01") & (o["date"] <= "2022-09-30")
        ax.plot(o.loc[mask, "date"], loss[mask], lw=1.2, label=name)
    ax.axvspan(pd.Timestamp("2022-02-25"), pd.Timestamp("2022-03-24"), color="grey", alpha=0.2, label="разрыв торгов")
    ax.set_title("QLIKE, скользящее среднее за 60 дней, вокруг марта 2022")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "rolling60_qlike_march2022.png", dpi=120)
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    print("\nМетрики (основной вариант):")
    print(metrics[metrics["variant"] == main_key].to_string(index=False))


if __name__ == "__main__":
    main()
