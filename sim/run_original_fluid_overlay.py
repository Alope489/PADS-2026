#!/usr/bin/env python3
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


ARTIFACT_ROOT = Path(os.environ.get("ARTIFACT_ROOT", Path(__file__).resolve().parent.parent)).resolve()
LINK_BW_GBPS = 25.0
BUFFER_BYTES = 8448.0
BUFFER_GB = BUFFER_BYTES / 1e9
FLOW_TO_TERM_ID = {0: 14, 1: 15}


def compute_service_rates(arrival):
    agg_arrivals_output = np.sum(arrival, axis=0)
    service = np.zeros_like(arrival)
    for input_port in range(arrival.shape[0]):
        for output_port in range(arrival.shape[1]):
            if agg_arrivals_output[output_port] > 0:
                service[input_port, output_port] = LINK_BW_GBPS * arrival[input_port, output_port] / agg_arrivals_output[output_port]
    return service


def compute_original_fluid_lines():
    arrival = np.array([[25.0], [20.0]], dtype=float)
    occupancy = np.zeros(2, dtype=float)
    initial_service = compute_service_rates(arrival)
    fill_times = np.full(2, np.inf, dtype=float)
    for input_port in range(2):
        net_input_rate = np.sum(arrival[input_port, :] - initial_service[input_port, :])
        if net_input_rate > 0:
            fill_times[input_port] = (BUFFER_GB - occupancy[input_port]) / net_input_rate
    if not np.isfinite(fill_times).any():
        return initial_service, initial_service.copy(), initial_service.copy()
    first_buffer_fill_port = int(np.argmin(fill_times))
    first_buffer_fill_time = float(fill_times[first_buffer_fill_port])
    for input_port in range(2):
        occupancy[input_port] += (arrival[input_port, 0] - initial_service[input_port, 0]) * first_buffer_fill_time
        occupancy[input_port] = min(max(occupancy[input_port], 0.0), BUFFER_GB)
    phase1_arrival = arrival.copy()
    phase1_arrival[first_buffer_fill_port, 0] = initial_service[first_buffer_fill_port, 0]
    phase1_service = compute_service_rates(phase1_arrival)
    equilibrium_arrival = np.zeros_like(arrival)
    equilibrium_arrival[first_buffer_fill_port, 0] = initial_service[first_buffer_fill_port, 0]
    equilibrium_arrival[1 - first_buffer_fill_port, 0] = phase1_service[1 - first_buffer_fill_port, 0]
    equilibrium_service = compute_service_rates(equilibrium_arrival)
    return initial_service, phase1_service, equilibrium_service


def load_two_flow_codes_data():
    term_files = sorted(
        glob.glob(str(ARTIFACT_ROOT / "multiflow" / "results" / "two-flow" / "terminal-packet-stats-*")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not term_files:
        raise RuntimeError(
            f"No terminal-packet-stats-* files found under {ARTIFACT_ROOT / 'multiflow' / 'results' / 'two-flow'}."
        )
    terminal_df = pd.read_table(term_files[0], sep=r"\s+")
    for column in list(terminal_df.columns):
        if column.startswith("qos-") or column.startswith("vc") or column in {"Unnamed: 0", "qos-level", "downstream-credits"}:
            terminal_df = terminal_df.drop(column, axis=1)
    terminal_df = terminal_df[terminal_df["bw-consumed"] != 0].copy()
    terminal_df["time-stamp"] = terminal_df["time-stamp"] / 1000.0
    terminal_df["bw-consumed"] = terminal_df["bw-consumed"] * LINK_BW_GBPS / 100.0
    codes_data = {}
    for flow_id, term_id in FLOW_TO_TERM_ID.items():
        flow_df = terminal_df[terminal_df["term-id"] == term_id]
        if len(flow_df) == 0:
            raise RuntimeError(f"Missing term-id {term_id} data in {term_files[0]}.")
        codes_data[flow_id] = {
            "x": flow_df["time-stamp"].to_numpy(),
            "y": flow_df["bw-consumed"].to_numpy(),
        }
    return codes_data


def main():
    initial_service, phase1_service, equilibrium_service = compute_original_fluid_lines()
    codes_data = load_two_flow_codes_data()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for flow_id in (0, 1):
        axes[flow_id].plot(codes_data[flow_id]["x"], codes_data[flow_id]["y"], "b-", linewidth=2, zorder=1, label="CODES")
        axes[flow_id].axhline(
            initial_service[flow_id, 0],
            color="red",
            linestyle="--",
            alpha=0.8,
            linewidth=1.5,
            label=f"Initial: {initial_service[flow_id, 0]:.3f} GB/s",
        )
        axes[flow_id].axhline(
            phase1_service[flow_id, 0],
            color="orange",
            linestyle="--",
            alpha=0.8,
            linewidth=1.5,
            label=f"1st Fill: {phase1_service[flow_id, 0]:.3f} GB/s",
        )
        axes[flow_id].axhline(
            equilibrium_service[flow_id, 0],
            color="grey",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"2nd Fill (Equilibrium): {equilibrium_service[flow_id, 0]:.3f} GB/s",
        )
        axes[flow_id].set_title(f"Flow {flow_id}", fontsize=22, fontweight="bold")
        axes[flow_id].set_xlabel("Time (us)", fontsize=24, fontweight="bold")
        axes[flow_id].set_xlim(0, 10)
        axes[flow_id].set_ylim(0, 35)
        axes[flow_id].tick_params(axis="both", which="major", labelsize=18)
        axes[flow_id].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[flow_id].grid(True, alpha=0.3)
        axes[flow_id].legend(loc="upper right", fontsize=14)
    axes[0].set_ylabel("Injection Rate (GB/s)", fontsize=24, fontweight="bold")
    axes[1].tick_params(axis="y", labelleft=False)
    out_dir = ARTIFACT_ROOT / "results" / "figures" / "original_fluid"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "original_fluid_overlay.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
