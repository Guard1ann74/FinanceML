"""Publication-focused figures from saved forecasts/tables; no model fitting."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates


ROOT = Path(__file__).parent
RES = ROOT / "results"
CAL = RES / "calendar"
OUT = RES / "figures"

COLORS = {
    "fixed": "#0072B2",
    "expanding": "#D55E00",
    "rolling": "#009E73",
    "har": "#CC79A7",
    "naive": "#E69F00",
    "fact": "#222222",
    "old": "#6B7280",
    "aligned": "#0072B2",
}
LABELS = {
    "fixed": "Фиксированная",
    "expanding": "Расширяющаяся",
    "rolling": "Скользящая (3 года)",
    "har": "HAR",
    "naive": "Naive",
}
MODELS = ["fixed", "expanding", "rolling", "har", "naive"]
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def panel_scores():
    df = pd.read_csv(CAL / "per_stock_old_new_metrics.csv")
    df = df[(df["period"] == "Тест 2022+") & df["SECID"].notna()]
    aligned = df[df["sample"] == "Календарь IMOEX"].set_index("SECID")
    order = aligned.sort_values("QLIKE_expanding").index.tolist()
    n = len(order)
    offsets = np.linspace(-0.25, 0.25, len(MODELS))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 6.6), sharey=True)
    for ax, sample, title in zip(
        axes, ["Исходный календарь", "Календарь IMOEX"],
        ["Исходные даты акций", "Только даты IMOEX"],
    ):
        block = df[df["sample"] == sample].set_index("SECID")
        for j, model in enumerate(MODELS):
            x = block.loc[order, f"QLIKE_{model}"].to_numpy()
            y = np.arange(n) + offsets[j]
            ax.scatter(x, y, s=20, color=COLORS[model], label=LABELS[model], zorder=3)
        ax.set_xscale("log")
        ax.set_title(title, pad=8)
        ax.set_xlabel("QLIKE · меньше лучше")
        ax.grid(axis="x", color="#E5E7EB", lw=0.7)
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(order)
        ax.invert_yaxis()
    axes[0].set_ylabel("")
    axes[1].tick_params(labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "panel_qlike_by_ticker")


def calendar_effect():
    df = pd.read_csv(CAL / "per_stock_old_new_metrics.csv")
    df = df[df["period"].isin(["2022", "2023 и позже"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 6.6), sharey=True)
    for ax, period, title in zip(axes, ["2022", "2023 и позже"], ["2022", "2023+"]):
        g = df[df["period"] == period]
        old = g[g["sample"] == "Исходный календарь"].set_index("SECID")
        new = g[g["sample"] == "Календарь IMOEX"].set_index("SECID")
        order = new["rel_fixed_vs_expanding"].sort_values().index.tolist()
        ax.axvline(0, color="#111827", lw=0.9)
        for yi, ticker in enumerate(order):
            x0 = 100 * old.loc[ticker, "rel_fixed_vs_expanding"]
            x1 = 100 * new.loc[ticker, "rel_fixed_vs_expanding"]
            ax.plot([x0, x1], [yi, yi], color="#D1D5DB", lw=1.2, zorder=1)
            ax.scatter(x0, yi, color=COLORS["old"], s=22, zorder=3)
            ax.scatter(x1, yi, color=COLORS["aligned"], s=22, zorder=3)
        ax.set_title(title, pad=8)
        ax.set_xlabel("QLIKE фикс. / расшир. − 1, %")
        ax.grid(axis="x", color="#E5E7EB", lw=0.7)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(order)
        ax.invert_yaxis()
    axes[1].tick_params(labelleft=False)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["old"],
                   markeredgecolor="none", label="Исходный", markersize=6),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["aligned"],
                   markeredgecolor="none", label="Календарь IMOEX", markersize=6),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "calendar_effect_by_ticker")


def off_calendar_share():
    df = pd.read_csv(CAL / "off_calendar_qlike_share.csv")
    df = df[df["period"] == "Тест 2022+"].set_index("model").loc[MODELS].reset_index()
    labels = [LABELS[m] for m in df["model"]]
    off = 100 * df["share_sum_QLIKE_off_calendar"].to_numpy()
    common = 100 - off
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    y = np.arange(len(df))
    ax.barh(y, common, color="#CBD5E1", height=0.58, label="Даты IMOEX")
    ax.barh(y, off, left=common, color=COLORS["expanding"], height=0.58, label="Вне календаря IMOEX")
    for yi, val in zip(y, off):
        ax.text(99.2, yi, f"{val:.1f}%", va="center", ha="right", color="white", fontsize=8, fontweight="bold")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Доля суммы QLIKE, %")
    ax.grid(axis="x", color="#E5E7EB", lw=0.7)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    save(fig, "off_calendar_qlike_share")


def pikk_june():
    f = pd.read_csv(RES / "panel" / "forecasts" / "PIKK.csv", parse_dates=["date"])
    f = f[f["date"].between("2026-06-01", "2026-06-19")]
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.plot(f["date"], f["y"], color=COLORS["fact"], lw=2.0, label="Факт")
    for model in ["fixed", "expanding", "rolling", "har"]:
        ax.plot(f["date"], f[model], color=COLORS[model], lw=1.4, label=LABELS[model])
    event_date = pd.Timestamp("2026-06-11")
    ax.axvline(event_date, color="#6B7280", lw=1, ls="--")
    ax.annotate("r = +8,7%", xy=(event_date, f.loc[f.date.eq(event_date), "y"].iloc[0]),
                xytext=(7, 9), textcoords="offset points", fontsize=8, color="#374151")
    ax.set_ylabel("Лог годовой RV")
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(axis="y", color="#E5E7EB", lw=0.7)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    save(fig, "pikk_june_2026_forecasts")


def imoex_zoom():
    f = pd.read_csv(RES / "predictions_main.csv", parse_dates=["date"])
    f = f[f["date"].between("2022-01-01", "2022-06-30")].copy()
    columns = {
        "Фиксированный бустинг (а)": ("fixed", COLORS["fixed"]),
        "Расширяющееся окно (б)": ("expanding", COLORS["expanding"]),
        "Скользящее окно 3 года (в)": ("rolling", COLORS["rolling"]),
        "HAR (расширяющееся, переобучение 20 дн.)": ("har", COLORS["har"]),
    }
    fig, ax = plt.subplots(figsize=(9.2, 4.3))
    ax.plot(f["date"], np.exp(f["y"]), color=COLORS["fact"], lw=1.8, label="Факт")
    for col, (key, color) in columns.items():
        ax.plot(f["date"], np.exp(f[col]), color=color, lw=1.3, label=LABELS[key])
    ax.axvspan(pd.Timestamp("2022-02-25"), pd.Timestamp("2022-03-24"), color="#E5E7EB", zorder=0)
    ax.axvline(pd.Timestamp("2022-03-24"), color="#6B7280", lw=1, ls="--")
    ax.set_ylabel("Годовая RV")
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(axis="y", color="#E5E7EB", lw=0.7)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    save(fig, "imoex_volatility_2022_zoom")


def main():
    panel_scores()
    calendar_effect()
    off_calendar_share()
    pikk_june()
    imoex_zoom()
    print(f"Saved publication figures to {OUT}")


if __name__ == "__main__":
    main()
