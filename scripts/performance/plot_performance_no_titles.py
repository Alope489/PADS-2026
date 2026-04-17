#!/usr/bin/env python3
"""Regenerate performance PNGs from CSVs in results/performance/ into results/figures/performance/."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
CSV_DIR = ARTIFACT_ROOT / "results" / "performance"
FIG_DIR = ARTIFACT_ROOT / "results" / "figures" / "performance"
CSV_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _fluid_var_col(df):
    return df["fluid_variance"] if "fluid_variance" in df.columns else df["variance"]


def plot_sim_time():
    df = pd.read_csv(CSV_DIR / "sim_time_runtime_data.csv")
    plt.figure(figsize=(8, 6))
    plt.plot(df["sim_time_us"], df["fluid_5ns_runtime_s"], "o-", linewidth=2, markersize=6, label="Fluid Model (dt=5ns)")
    plt.plot(df["sim_time_us"], df["fluid_10ns_runtime_s"], "s-", linewidth=2, markersize=6, label="Fluid Model (dt=10ns)")
    plt.plot(df["sim_time_us"], df["codes_runtime_s"], "^-", linewidth=2, markersize=6, label="CODES Simulation")
    plt.xlabel("Simulation Time (µs)", fontsize=18, fontweight="bold")
    plt.ylabel("Wall-Clock Runtime (s)", fontsize=18, fontweight="bold")
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14, loc="upper left")
    plt.tight_layout()
    out = FIG_DIR / "runtime_vs_sim_time_pypy.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_granularity():
    df = pd.read_csv(CSV_DIR / "granularity_tradeoff_pypy_data.csv")
    fv = _fluid_var_col(df)
    dts = df["dt_ns"].tolist()
    speedups = df["speedup"].tolist()
    var_vs_codes = df["variance_preserved_pct"].tolist()
    gt_mask = df["dt_ns"] == 1
    if not gt_mask.any():
        gt_mask = df["dt_ns"] == df["dt_ns"].min()
    gt_fluid = float(fv[gt_mask].iloc[0])
    var_vs_fluid_1ns = (fv / gt_fluid * 100.0).tolist()
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.set_xlabel("Granularity dt (ns)", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Speedup (CODES runtime / Fluid runtime)", fontsize=16, fontweight="bold", color="tab:blue")
    ax1.plot(dts, speedups, "o-", color="tab:blue", linewidth=2.5, markersize=8, zorder=10, label="Speedup")
    ax1.tick_params(axis="y", labelcolor="tab:blue", labelsize=14)
    ax1.tick_params(axis="x", labelsize=14)
    ax1.set_ylim(0, None)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.patch.set_visible(False)
    ax2.set_ylabel("Variance Preserved (%)", fontsize=16, fontweight="bold")
    ax2.plot(dts, var_vs_fluid_1ns, "s--", color="tab:red", linewidth=2.5, markersize=8, zorder=10, label="vs Fluid dt=1ns")
    ax2.plot(dts, var_vs_codes, "D:", color="tab:orange", linewidth=2.5, markersize=8, zorder=10, label="vs CODES")
    ax2.tick_params(axis="y", labelsize=14)
    ax2.set_ylim(0, 110)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "granularity_tradeoff_pypy.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_offered_load():
    df = pd.read_csv(CSV_DIR / "offered_load_pypy_data.csv")
    plt.figure(figsize=(8, 6))
    plt.plot(df["offered_load_gbps"], df["fluid_runtime_s"], "o-", linewidth=2, markersize=6, label="Fluid Model (dt=10ns)")
    plt.plot(df["offered_load_gbps"], df["codes_runtime_s"], "s-", linewidth=2, markersize=6, label="CODES Simulation")
    plt.xlabel("Offered Load (GB/s per flow)", fontsize=18, fontweight="bold")
    plt.ylabel("Wall-Clock Runtime (s)", fontsize=18, fontweight="bold")
    plt.tick_params(axis="both", which="major", labelsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14, loc="upper left")
    plt.tight_layout()
    out = FIG_DIR / "offered_load_pypy.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    which = (sys.argv[1:] or ["all"])[0].lower()
    if which in ("all", ""):
        plots = (
            ("plot_sim_time", plot_sim_time),
            ("plot_granularity", plot_granularity),
            ("plot_offered_load", plot_offered_load),
        )
        ok = 0
        fnf = 0
        for name, fn in plots:
            try:
                fn()
                ok += 1
            except FileNotFoundError as e:
                fnf += 1
                print(f"skip {name}: missing CSV or path ({e})")
        sys.exit(0 if (ok >= 1 or fnf == len(plots)) else 1)
    if which in ("granularity", "tradeoff", "gran"):
        try:
            plot_granularity()
        except FileNotFoundError as e:
            print(f"skip plot_granularity: missing CSV or path ({e})")
            sys.exit(0)
        return
    if which in ("simtime", "sim_time", "sim"):
        try:
            plot_sim_time()
        except FileNotFoundError as e:
            print(f"skip plot_sim_time: missing CSV or path ({e})")
            sys.exit(0)
        return
    if which in ("offered", "load"):
        try:
            plot_offered_load()
        except FileNotFoundError as e:
            print(f"skip plot_offered_load: missing CSV or path ({e})")
            sys.exit(0)
        return
    print("Usage: python3 plot_performance_no_titles.py [all|granularity|simtime|offered]")
    sys.exit(1)


if __name__ == "__main__":
    main()
