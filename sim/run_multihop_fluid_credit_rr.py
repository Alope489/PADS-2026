import glob
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ARTIFACT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARTIFACT_ROOT / "sim"))

from multihop_fluid_sim import FlowConfig, PortConfig, NetworkConfig, MultiHopFluidSimulator

LINK_DELAY = 100e-9
ROUTER_DELAY = 100e-9
CHUNK_SIZE = 64.0
LINK_BW = 25e9
BUF_SIZE = 8448.0
CREDIT_DRAIN_TIME = BUF_SIZE / LINK_BW


def load_codes_terminal_stats(term_ids):
    term_files = sorted(
        glob.glob(str(ARTIFACT_ROOT / "multiflow" / "**" / "terminal-packet-stats-*"), recursive=True),
        key=os.path.getmtime,
        reverse=True,
    )
    if not term_files:
        return {}, {}
    tdf = pd.read_table(term_files[0], sep=r"\s+")
    for col in ("qos-level", "downstream-credits", "Unnamed: 0"):
        if col in tdf.columns:
            tdf = tdf.drop(columns=[col])
    tdf = tdf[tdf["bw-consumed"] != 0].copy()
    if len(tdf) == 0:
        return {}, {}
    raw_tdf = tdf.copy()
    plot_tdf = tdf.copy()
    plot_tdf["time-stamp"] = plot_tdf["time-stamp"] / 1000.0
    plot_tdf["bw-consumed"] = plot_tdf["bw-consumed"] * LINK_BW / 100.0 / 1e9
    codes_data = {}
    raw_data = {}
    for fid, term_id in term_ids.items():
        raw_flow = raw_tdf[raw_tdf["term-id"] == term_id]
        plot_flow = plot_tdf[plot_tdf["term-id"] == term_id]
        if len(plot_flow) > 0:
            codes_data[fid] = {"t": plot_flow["time-stamp"].to_numpy(), "x": plot_flow["bw-consumed"].to_numpy()}
            raw_data[fid] = {"t_raw": raw_flow["time-stamp"].to_numpy(), "bw_percent": raw_flow["bw-consumed"].to_numpy()}
    return codes_data, raw_data


def derive_effective_port_capacities(flows, raw_codes_data):
    effective = {}
    flow_to_rate = {}
    for fid, raw in raw_codes_data.items():
        if len(raw["t_raw"]) == 0:
            continue
        tmax = float(np.max(raw["t_raw"]))
        cutoff = tmax - 1000.0
        mask = raw["t_raw"] >= cutoff
        if np.any(mask):
            flow_to_rate[fid] = float(np.mean(raw["bw_percent"][mask]) * LINK_BW / 100.0)
    port_to_flow_ids = {}
    for flow in flows:
        for hop_idx, port_key in enumerate(flow.route):
            if hop_idx == 0:
                port_to_flow_ids.setdefault(port_key, set()).add(flow.flow_id)
    for port_key, flow_ids in port_to_flow_ids.items():
        if len(flow_ids) > 1 and all(fid in flow_to_rate for fid in flow_ids):
            effective[port_key] = sum(flow_to_rate[fid] for fid in flow_ids)
    return effective


def main():
    dt_codes = 5.3e-9
    T_total = 10e-6
    flow_rates = {0: 15e9, 1: 15e9, 2: 25e9}
    term_ids = {0: 14, 1: 15, 2: 1}
    flows = [
        FlowConfig(
            flow_id=0,
            lambda_rate=flow_rates[0],
            credit_drain_time=CREDIT_DRAIN_TIME,
            route=[(0, 0), (1, 0), (2, 0)],
            forward_delays=[300e-9, 310e-9, 100e-9],
            backward_delays=[100e-9, 100e-9, 100e-9],
            initial_credits_per_hop=[BUF_SIZE, BUF_SIZE, BUF_SIZE],
            initial_backlogs=[0.0, 0.0, 0.0],
        ),
        FlowConfig(
            flow_id=1,
            lambda_rate=flow_rates[1],
            credit_drain_time=CREDIT_DRAIN_TIME,
            route=[(0, 0), (1, 1)],
            forward_delays=[300e-9, 300e-9],
            backward_delays=[100e-9, 100e-9],
            initial_credits_per_hop=[BUF_SIZE, BUF_SIZE],
            initial_backlogs=[0.0, 0.0],
        ),
        FlowConfig(
            flow_id=2,
            lambda_rate=flow_rates[2],
            credit_drain_time=CREDIT_DRAIN_TIME,
            route=[(1, 0), (2, 1)],
            forward_delays=[300e-9, 300e-9],
            backward_delays=[100e-9, 100e-9],
            initial_credits_per_hop=[BUF_SIZE, BUF_SIZE],
            initial_backlogs=[64.0, 0.0],
        ),
    ]
    codes_data, raw_codes_data = load_codes_terminal_stats(term_ids)
    if len(codes_data) == 0:
        raise RuntimeError(
            f"No CODES terminal stats found under {ARTIFACT_ROOT / 'multiflow' / '**' / 'terminal-packet-stats-*'}. "
            "Run multiflow/exec.sh successfully before generating overlays."
        )
    missing_flow_ids = [fid for fid in term_ids if fid not in codes_data]
    if len(missing_flow_ids) > 0:
        raise RuntimeError(
            f"Missing CODES terminal stats for flow IDs {missing_flow_ids} (term IDs {[term_ids[fid] for fid in missing_flow_ids]}). "
            "Regenerate replay outputs before plotting."
        )
    effective_caps = derive_effective_port_capacities(flows, raw_codes_data)
    ports = [
        PortConfig(router_id=0, port_id=0, capacity=effective_caps.get((0, 0), LINK_BW)),
        PortConfig(router_id=1, port_id=0, capacity=effective_caps.get((1, 0), LINK_BW)),
        PortConfig(router_id=1, port_id=1, capacity=LINK_BW),
        PortConfig(router_id=2, port_id=0, capacity=LINK_BW),
        PortConfig(router_id=2, port_id=1, capacity=LINK_BW),
    ]
    print(f"credit_drain_time = {CREDIT_DRAIN_TIME*1e9:.1f} ns")
    print(f"dt  = {dt_codes*1e9:.2f} ns,  T = {T_total*1e6:.1f} µs")
    if len(effective_caps) > 0:
        print("Effective shared-port capacities from CODES steady-state:")
        for (router_id, port_id), cap in sorted(effective_caps.items()):
            print(f"  Port ({router_id},{port_id}): {cap/1e9:.3f} GB/s")
    else:
        print("Effective shared-port capacities unavailable; using nominal LINK_BW.")
    for f in flows:
        print(f"  Flow {f.flow_id}: {f.lambda_rate/1e9:.0f} GB/s  route={f.route}")
    sim = MultiHopFluidSimulator(NetworkConfig(flows=flows, ports=ports, chunk_size=CHUNK_SIZE), dt_codes)
    res = sim.run(T_total, verbose=True)
    t_us = res["t"] * 1e6
    s = res["s"]
    idx = sim.hop_to_idx
    flow0_idx = idx[(0, 0)]
    flow1_idx = idx[(1, 0)]
    flow2_idx = idx[(2, 0)]
    n_steady = int(1e-6 / dt_codes)
    print("\nSteady-state averages (last 1 µs):")
    for fid, hop_idx in ((0, flow0_idx), (1, flow1_idx), (2, flow2_idx)):
        print(f"  Flow {fid}: service = {res['s'][-n_steady:, hop_idx].mean()/1e9:.3f} GB/s,  offered = {res['o'][-n_steady:, hop_idx].mean()/1e9:.3f} GB/s")
    for fid in sorted(codes_data):
        print(f"  CODES Flow {fid}: mean bw = {np.mean(codes_data[fid]['x']):.3f} GB/s  ({len(codes_data[fid]['x'])} pts)")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    flow_indices = {0: flow0_idx, 1: flow1_idx, 2: flow2_idx}
    flow_colors = {0: "tab:red", 1: "tab:green", 2: "tab:purple"}
    for fid in (0, 1, 2):
        ax = axes[fid]
        ax.plot(t_us, s[:, flow_indices[fid]] / 1e9, color=flow_colors[fid], linewidth=2, label="Fluid Model (service)")
        ax.plot(t_us, res["o"][:, flow_indices[fid]] / 1e9, color=flow_colors[fid], linewidth=1, linestyle="--", alpha=0.6, label="Fluid Model (offered)")
        if fid in codes_data:
            ax.plot(codes_data[fid]["t"], codes_data[fid]["x"], color="blue", linewidth=2, label="CODES")
        ax.axhline(flow_rates[fid] / 1e9, linestyle=":", color="gray", linewidth=1.5, alpha=0.5, label=f"Offered ({flow_rates[fid]/1e9:.0f} GB/s)")
        ax.set_title(f"Flow {fid} ({len(flows[fid].route)} hops, {flow_rates[fid]/1e9:.0f} GB/s)", fontsize=18, fontweight="bold")
        ax.set_xlabel("Time (µs)", fontsize=14)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 30)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=11)
    axes[0].set_ylabel("Rate (GB/s)", fontsize=14)
    fig.suptitle(f"Multi-hop fluid vs CODES  |  credit_drain_time = {CREDIT_DRAIN_TIME*1e9:.0f} ns", fontsize=14)
    out_dir = ARTIFACT_ROOT / "results" / "figures" / "multi_hop_HOL"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "multihop_codes_overlay.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
