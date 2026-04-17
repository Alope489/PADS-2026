# Performance benchmarks

Scripts under `scripts/performance/`: `run_sim_time_benchmark.py` (10–100 µs sweep), `run_granularity_benchmark.py` (dt sweep with CODES reference variance on terminal 14), `run_offered_load_benchmark.py` (5–25 GB/s per flow). CSVs land in `results/performance/`. PNGs are written under `results/figures/performance/` by `python3 scripts/performance/plot_performance_no_titles.py all` (reads CSVs from `results/performance/`).
