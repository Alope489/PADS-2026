#!/usr/bin/env python3
"""Sweep simulated wall-clock time (us): fluid dt 5/10 ns vs CODES; CSV output."""
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
LAMBDA_ARR = [25e9, 20e9]
MU = 25e9
TAU = 20e-9
B_INIT = [0.0, 0.0]
C_INIT = [8448.0, 8448.0]
NUM_ITERATIONS = 5
SIM_TIME_RANGE = range(10, 101)

PERF_DIR = ARTIFACT_ROOT / "results" / "performance"
PERF_DIR.mkdir(parents=True, exist_ok=True)
MULTIFLOW_DIR = ARTIFACT_ROOT / "multiflow"
SCRIPTS_DIR = ARTIFACT_ROOT / "scripts"
FLOW_RATES_FILE = SCRIPTS_DIR / "flow.rates"
CODES_EXECUTABLE = Path(
    os.environ.get(
        "CODES_REPLAY",
        str(ARTIFACT_ROOT.parent / "codes" / "build" / "bin" / "bin" / "model-net-mpi-replay"),
    )
)
CSV_FILE = PERF_DIR / "sim_time_runtime_data.csv"
FIXED_FLOW_RATES = {"0": {"0": 25, "1": 20}}


def _cleanup_codes_outputs():
    for p in MULTIFLOW_DIR.glob("riodir*"):
        shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)
    for p in MULTIFLOW_DIR.glob("router-bw-tracker-*"):
        p.unlink(missing_ok=True)
    for p in MULTIFLOW_DIR.glob("terminal-packet-stats-*"):
        p.unlink(missing_ok=True)


def regenerate_codes_inputs():
    subprocess.run(
        ["python3", str(SCRIPTS_DIR / "rate_gen.py"), str(FLOW_RATES_FILE)],
        cwd=str(SCRIPTS_DIR),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "ARTIFACT_ROOT": str(ARTIFACT_ROOT)},
    )


def mean_fluid_runtime(sim_time_us, dt_ns):
    dt_sec = dt_ns * 1e-9
    times = []
    for _ in range(NUM_ITERATIONS):
        t0 = time.perf_counter()
        run_simulation_pypy(
            N,
            sim_time_us * 1e-6,
            dt_sec,
            D_f,
            D_b,
            LAMBDA_ARR,
            MU,
            TAU,
            B_INIT,
            C_INIT,
        )
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def mean_codes_runtime(sim_time_us):
    end_ns = int(sim_time_us * 1000)
    times = []
    for _ in range(NUM_ITERATIONS):
        _cleanup_codes_outputs()
        (MULTIFLOW_DIR / "riodir").mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
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
        times.append(time.perf_counter() - t0)
        _cleanup_codes_outputs()
    return sum(times) / len(times)


def main():
    with open(FLOW_RATES_FILE, "w") as f:
        json.dump(FIXED_FLOW_RATES, f, indent=4)
    regenerate_codes_inputs()
    rows = []
    for sim_time_us in SIM_TIME_RANGE:
        print(f"sim_time_us={sim_time_us} ...")
        rows.append(
            {
                "sim_time_us": sim_time_us,
                "fluid_5ns_runtime_s": mean_fluid_runtime(sim_time_us, 5),
                "fluid_10ns_runtime_s": mean_fluid_runtime(sim_time_us, 10),
                "codes_runtime_s": mean_codes_runtime(sim_time_us),
            }
        )
        with open(CSV_FILE, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["sim_time_us", "fluid_5ns_runtime_s", "fluid_10ns_runtime_s", "codes_runtime_s"],
            )
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {CSV_FILE}")


if __name__ == "__main__":
    main()
