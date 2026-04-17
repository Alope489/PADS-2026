#!/usr/bin/env python3
"""Granularity trade-off: fluid runtime/variance vs CODES reference run."""
import csv
import glob
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ARTIFACT_ROOT / "sim"))
os.environ.setdefault("ARTIFACT_ROOT", str(ARTIFACT_ROOT))

from fluid_sim_pypy import run_simulation_pypy, column

N = 2
D_f = 300e-9
D_b = 200e-9
LAMBDA_ARR = [25e9, 20e9]
MU = 25e9
TAU = 20e-9
B_INIT = [0.0, 0.0]
C_INIT = [8448.0, 8448.0]
SIM_TIME_US = 70
DT_NS_VALUES = [1, 2, 5, 10, 15, 20, 30, 50]
NUM_ITERATIONS = 5
LINK_BW = 25e9

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
CSV_FILE = PERF_DIR / "granularity_tradeoff_pypy_data.csv"


def _cleanup_codes_outputs():
    for p in MULTIFLOW_DIR.glob("riodir*"):
        shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)
    for p in MULTIFLOW_DIR.glob("router-bw-tracker-*"):
        p.unlink(missing_ok=True)
    for p in MULTIFLOW_DIR.glob("terminal-packet-stats-*"):
        p.unlink(missing_ok=True)


def _codes_variance_term14():
    files = sorted(glob.glob(str(MULTIFLOW_DIR / "terminal-packet-stats-*")))
    if not files:
        return float("nan")
    tdf = pd.read_table(files[0], sep=r"\s+")
    for col in ("qos-level", "downstream-credits", "Unnamed: 0"):
        if col in tdf.columns:
            tdf = tdf.drop(columns=[col])
    sub = tdf[(tdf["term-id"] == 14) & (tdf["bw-consumed"] != 0)]
    if len(sub) == 0:
        return float("nan")
    bps = sub["bw-consumed"].to_numpy(dtype=float) / 100.0 * LINK_BW
    m = float(bps.mean())
    return float(((bps - m) ** 2).mean())


def run_codes_once():
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
    codes_var = _codes_variance_term14()
    _cleanup_codes_outputs()
    return elapsed, codes_var


def run_fluid_and_measure(dt_ns):
    dt_sec = dt_ns * 1e-9
    runtimes = []
    for _ in range(NUM_ITERATIONS):
        wall_start = time.perf_counter()
        t_arr, B_arr, C_arr, x_arr, s_arr = run_simulation_pypy(
            N,
            SIM_TIME_US * 1e-6,
            dt_sec,
            D_f,
            D_b,
            LAMBDA_ARR,
            MU,
            TAU,
            B_INIT,
            C_INIT,
        )
        runtimes.append(time.perf_counter() - wall_start)
    n_steps = int(math.ceil((SIM_TIME_US * 1e-6) / dt_sec))
    s0 = column(s_arr, N, 0, n_steps + 1)
    mean_s0 = sum(s0) / len(s0)
    fluid_var = sum((v - mean_s0) ** 2 for v in s0) / len(s0)
    return sum(runtimes) / len(runtimes), fluid_var


def main():
    subprocess.run(
        ["python3", str(SCRIPTS_DIR / "rate_gen.py"), str(SCRIPTS_DIR / "flow.rates")],
        cwd=str(SCRIPTS_DIR),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "ARTIFACT_ROOT": str(ARTIFACT_ROOT)},
    )
    print("Running CODES once for reference runtime + terminal variance (term 14)...")
    codes_runtime, codes_variance = run_codes_once()
    print(f"  CODES runtime: {codes_runtime:.4f}s  codes_variance (B/s)^2: {codes_variance:.6e}")
    records = []
    for i, dt_ns in enumerate(DT_NS_VALUES):
        print(f"[{i+1}/{len(DT_NS_VALUES)}] dt={dt_ns}ns ...")
        runtime, fluid_var = run_fluid_and_measure(dt_ns)
        speedup = codes_runtime / runtime if runtime > 0 else float("inf")
        vpct = (fluid_var / codes_variance * 100.0) if (math.isfinite(codes_variance) and codes_variance > 0) else 0.0
        records.append(
            {
                "dt_ns": dt_ns,
                "runtime_s": runtime,
                "fluid_variance": fluid_var,
                "codes_variance": codes_variance,
                "variance_preserved_pct": vpct,
                "speedup": speedup,
            }
        )
        print(f"    runtime={runtime:.4f}s fluid_var={fluid_var:.4e} speedup={speedup:.2f}x vpct={vpct:.2f}%")
    with open(CSV_FILE, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "dt_ns",
                "runtime_s",
                "fluid_variance",
                "codes_variance",
                "variance_preserved_pct",
                "speedup",
            ],
        )
        w.writeheader()
        w.writerows(records)
    print(f"\nCSV saved to {CSV_FILE}")


if __name__ == "__main__":
    main()
