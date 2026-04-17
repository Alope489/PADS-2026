# Fluid engines

`sim/fluid_sim_pypy.py` — PyPy-friendly RK4 fluid model. `sim/multihop_fluid_sim.py` — multi-hop credit fluid. `sim/run_multihop_fluid_credit_rr.py` overlays the multi-hop fluid run against the latest CODES terminal stats found under `multiflow/` and writes `results/figures/multi_hop_HOL/multihop_codes_overlay.png`. `sim/run_original_fluid_overlay.py` reproduces the two-flow original-fluid line overlay against two-flow CODES stats and writes `results/figures/original_fluid/original_fluid_overlay.png`.
