# Artifact bundle

Self-contained ModSim-style fluid vs CODES pipeline. From the artifact root run `./artifact` (requires `python3`, `pandas`, `matplotlib`, and a built `model-net-mpi-replay`; optional `pypy3`). Set `CODES_REPLAY` to the replay binary, or place codes at `../codes/build/bin/bin/model-net-mpi-replay`. Set `ARTIFACT_SKIP_BENCHES=1` to skip long benchmarks while still running multiflow replays, multihop overlay, original-fluid overlay, and plots.
