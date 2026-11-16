"""Часть 1б: цели, исключённые проверкой «без 2022-02-24», и DM/bootstrap на этом подмножестве."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import vol_pilot as vp  # noqa: E402

ROOT = Path(__file__).parent
RES = ROOT / "results"

FIX = "Фиксированный бустинг (а)"
EXP = "Расширяющееся окно (б)"


def main():
    out = pd.read_csv(RES / "predictions_main.csv", parse_dates=["date"])
    raw, _ = vp.load_raw()
    dates = raw["begin"].reset_index(drop=True)
    shock_pos = int(dates[dates == vp.SHOCK].index[0])

    affected = []
    for p in range(shock_pos - vp.H, shock_pos):
        origin = dates.iloc[p]
        row = out.iloc[p]
        valid = bool(pd.notna(row[FIX]) and pd.notna(row["y"]))
        affected.append({
            "origin_date": str(origin.date()),
            "target_window": f"{dates.iloc[p + 1].date()}..{dates.iloc[p + vp.H].date()}",
            "in_main_test_sample": valid and origin >= vp.TEST_START,
        })
    affected_df = pd.DataFrame(affected)
    affected_df.to_csv(RES / "shock_window_targets.csv", index=False)

    test = (out["date"] >= vp.TEST_START) & out["y"].notna() & out[FIX].notna() & out[EXP].notna()
    pos = np.arange(len(out))
    excl = ~((pos >= shock_pos - vp.H) & (pos <= shock_pos - 1))
    main_m = test.to_numpy()
    sub_m = (test & excl).to_numpy()

    la = vp.losses(out["y"][main_m], out[FIX][main_m])["qlike"]
    lb = vp.losses(out["y"][main_m], out[EXP][main_m])["qlike"]
    removed = int(main_m.sum() - sub_m.sum())

    la_s = vp.losses(out["y"][sub_m], out[FIX][sub_m])["qlike"]
    lb_s = vp.losses(out["y"][sub_m], out[EXP][sub_m])["qlike"]
    dm = vp.dm_test(la_s, lb_s)
    rng = np.random.default_rng(vp.SEED)
    lo, hi = vp.block_bootstrap_ratio(la_s, lb_s, rng)

    result = {
        "targets_removed": removed,
        "removed_origin_dates": affected_df.loc[affected_df["in_main_test_sample"], "origin_date"].tolist(),
        "all_affected_origin_dates": affected_df["origin_date"].tolist(),
        "n_full_test": int(main_m.sum()),
        "n_subset": int(sub_m.sum()),
        "QLIKE_fixed_full": float(la.mean()),
        "QLIKE_expanding_full": float(lb.mean()),
        "QLIKE_fixed_subset": float(la_s.mean()),
        "QLIKE_expanding_subset": float(lb_s.mean()),
        "rel_diff_subset": float(la_s.mean() / lb_s.mean() - 1),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "dm_stat": float(dm["dm_stat"]),
        "dm_p_value": float(dm["p_value"]),
        "mean_loss_diff": float(dm["mean_diff"]),
    }
    (RES / "part1b_shock_exclusion.json").write_text(
        pd.Series(result).to_json(indent=2, force_ascii=False), encoding="utf-8")
    for k, v in result.items():
        print(k, v)


if __name__ == "__main__":
    main()
