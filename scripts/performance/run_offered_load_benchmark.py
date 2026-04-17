#!/usr/bin/env python3
"""Offered-load sweep: fluid vs CODES wall-clock; CSV under results/performance."""
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ARTIFACT_ROOT / "sim"))
os.environ.setdefault("ARTIFACT_ROOT", str(ARTIFACT_ROOT))

from fluid_sim_pypy import run_simulation_pypy

N = 2
D_f = 300e-9
D_b = 200e-9
MU = 25e9
TAU = 20e-9
B_INIT = [0.0, 0.0]
C_INIT = [8448.0, 8448.0]
DT = 10e-9
SIM_TIME_US = 50
OFFERED_LOAD_GBPS_VALUES = list(range(5, 26, 1))
NUM_ITERATIONS = 10

PERF_DIR = ARTIFACT_ROOT / "results" / "performance"
PERF_DIR.mkdir(parents=True, exist_ok=True)
MULTIFLOW_DIR = ARTIFACT_ROOT / "multiflow"
SCRIPTS_DIR = ARTIFACT_ROOT / "scripts"
CODES_EXECUTABLE = Path(
    os.environ.get(
        "CODES_REPLAY",
        str(ARTIFACT_ROOT.parent / "codes" / "build" / "bin" / "bin" / "model-net-mpi-replay"),
    )
)
FLOW_RATES_FILE = SCRIPTS_DIR / "flow.rates"
CSV_FILE = PERF_DIR / "offered_load_pypy_data.csv"


def _cleanup_codes_outputs():
    for p in MULTIFLOW_DIR.glob("riodir*"):
        shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)
    for p in MULTIFLOW_DIR.glob("router-bw-tracker-*"):
        p.unlink(missing_ok=True)
    for p in MULTIFLOW_DIR.glob("terminal-packet-stats-*"):
        p.unlink(missing_ok=True)


def update_flow_rates(rate_gbps):
    with open(FLOW_RATES_FILE, "w") as f:
        json.dump({"0": {"0": rate_gbps, "1": rate_gbps}}, f, indent=4)


def regenerate_codes_inputs():
    subprocess.run(
        ["python3", str(SCRIPTS_DIR / "rate_gen.py"), str(FLOW_RATES_FILE)],
        cwd=str(SCRIPTS_DIR),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "ARTIFACT_ROOT": str(ARTIFACT_ROOT)},
    )


def run_fluid_runtime_seconds(offered_load_gbps):
    wall_start = time.perf_counter()
    run_simulation_pypy(
        N,
        SIM_TIME_US * 1e-6,
        DT,
        D_f,
        D_b,
        [offered_load_gbps * 1e9, offered_load_gbps * 1e9],
        MU,
        TAU,
        B_INIT,
        C_INIT,
    )
    return time.perf_counter() - wall_start


def run_codes_runtime_seconds():
    _cleanup_codes_outputs()
    end_ns = int(SIM_TIME_US * 1_000)
    (MULTIFLOW_DIR / "riodir").mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()
    subprocess.run(
        [
            str(CODES_EXECUTABLE),
            "--synch=1",
            "--workload_type=online",
            "--workload_conf_file=../conf/work.load",
            "--alloc_file=../conf/alloc.conf",
            "--lp-io-dir=riodir",
            "--lp-io-use-suffix=1",
            "--payload_sz=64",
            f"--end={end_ns}",
            "--workload_period_file=period.file",
            "--",
            "../conf/modsim-dfdally72-min.conf",
        ],
        cwd=str(MULTIFLOW_DIR),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    elapsed = time.perf_counter() - wall_start
    _cleanup_codes_outputs()
    return elapsed


def main():
    print("=" * 60)
    print("OFFERED LOAD BENCHMARK (PyPy fluid model)")
    print(f"Simulation time: {SIM_TIME_US} us")
    print("=" * 60 + "\n")
    records = []
    for i, load in enumerate(OFFERED_LOAD_GBPS_VALUES):
        print(f"[{i+1}/{len(OFFERED_LOAD_GBPS_VALUES)}] offered_load={load} GB/s ({NUM_ITERATIONS} iters)...")
        update_flow_rates(load)
        regenerate_codes_inputs()
        fluid_times = [run_fluid_runtime_seconds(load) for _ in range(NUM_ITERATIONS)]
        fluid_avg = sum(fluid_times) / NUM_ITERATIONS
        print(f"  Fluid AVG: {fluid_avg:.4f}s")
        codes_times = [run_codes_runtime_seconds() for _ in range(NUM_ITERATIONS)]
        codes_avg = sum(codes_times) / NUM_ITERATIONS
        print(f"  CODES AVG: {codes_avg:.4f}s")
        records.append({"offered_load_gbps": load, "fluid_runtime_s": fluid_avg, "codes_runtime_s": codes_avg})
        with open(CSV_FILE, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["offered_load_gbps", "fluid_runtime_s", "codes_runtime_s"])
            w.writeheader()
            w.writerows(records)
    print(f"\nCSV saved to {CSV_FILE}")


if __name__ == "__main__":
    main()
