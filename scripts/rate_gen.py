#!/usr/bin/env python3
"""Generate multiflow/period.file, conf/work.load, conf/alloc.conf from a JSON rates file."""
import argparse
import json
import os
from pathlib import Path
from pprint import pprint

ARTIFACT_ROOT = Path(os.environ.get("ARTIFACT_ROOT", Path(__file__).resolve().parent.parent)).resolve()

parser = argparse.ArgumentParser(description="Generate CODES flow configs for artifact runs.")
parser.add_argument("flows_file", help="Path to flow rate JSON.")
parser.add_argument(
    "--codes_flows_file",
    default="multiflow/period.file",
    help="Output period file path relative to ARTIFACT_ROOT unless absolute.",
)
args = parser.parse_args()
flows_path = Path(args.flows_file).resolve()
codes_rel = Path(args.codes_flows_file)
codesflowsfile = codes_rel if codes_rel.is_absolute() else (ARTIFACT_ROOT / codes_rel)

with open(flows_path, "r") as file:
    flowrates = json.load(file)
pprint(flowrates)

message_size = 64
bw = 25 * (1024 * 1024 * 1024) / (1000 * 1000 * 1000)
base_rate = message_size / bw

flows = {}
for key in flowrates.keys():
    for k, v in flowrates[key].items():
        k = int(k)
        key = int(key)
        if v == 0:
            v = 0.1
        tmp = (v / 25) * 100
        myrate = base_rate * (100 / tmp)
        if k not in flows:
            flows[k] = []
        flows[k].append((key, round(myrate, 6)))
pprint(flows)

codesflowsfile.parent.mkdir(parents=True, exist_ok=True)
with open(codesflowsfile, "w") as rate_file:
    rate_file.write("0\n")
    for flow in sorted(flows.keys()):
        rate_file.write("%d " % (len(flows[flow])))
        for i, rate in flows[flow]:
            rate_file.write("%d:%f " % (i * 1000, rate))
        rate_file.write("\n")

flowconfigs = {
    0: {"workload": "2 synthetic0 0 4.0", "allocation": "14 3"},
    1: {"workload": "2 synthetic0 0 4.0", "allocation": "15 0"},
    2: {"workload": "2 synthetic0 0 4.0", "allocation": "1 2"},
}
workload_path = ARTIFACT_ROOT / "conf" / "work.load"
alloc_path = ARTIFACT_ROOT / "conf" / "alloc.conf"
workload_path.parent.mkdir(parents=True, exist_ok=True)
with open(workload_path, "w") as workload_file:
    workload_file.write("4 ../conf/allreduce_workload.json 0 0.0\n")
    for flow in sorted(flows.keys()):
        workload_file.write(flowconfigs[flow]["workload"] + "\n")
with open(alloc_path, "w") as allocation_file:
    allocation_file.write("50 51 52 53\n")
    for flow in sorted(flows.keys()):
        allocation_file.write(flowconfigs[flow]["allocation"] + "\n")
