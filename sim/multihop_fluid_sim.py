from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class FlowConfig:
    """
    Multi-hop flow with per-hop credits.

    credit_drain_time : credit discharge timescale (s).
        Derived from CODES config: credit_drain_time = vc_buffer_size / link_bandwidth.
        Example: 8448 bytes / 25 GB/s = 337.9 ns.
        Physical meaning: time to drain one full VC credit pool at link speed.
        The credit-limited offered rate is C / credit_drain_time.

    route[h]                 = (router_id, port_id) for hop h
    initial_credits_per_hop[h] = initial credit pool for hop h (bytes, = buf_size)
    initial_backlogs[h]      = initial queue depth at hop h
    """

    flow_id: int
    lambda_rate: float  # bytes/s
    credit_drain_time: float
    route: List[Tuple[int, int]]
    forward_delays: List[float]
    backward_delays: List[float]
    initial_credits_per_hop: List[float]
    initial_backlogs: List[float] = field(default_factory=list)


@dataclass
class PortConfig:
    router_id: int
    port_id: int
    capacity: float  # bytes/s


@dataclass
class NetworkConfig:
    flows: List[FlowConfig]
    ports: List[PortConfig]
    chunk_size: float = 64.0

    def get_port_capacity(self, router_id: int, port_id: int) -> float:
        for port in self.ports:
            if port.router_id == router_id and port.port_id == port_id:
                return port.capacity
        raise ValueError(f"Port ({router_id}, {port_id}) not found")


class DelayHistory:
    """Ring buffer for values with linear interpolation lookup."""

    def __init__(self, shape, max_steps: int, dt: float, init_value=None):
        self.shape = shape if isinstance(shape, tuple) else (shape,)
        self.dt = dt
        self.times = deque(maxlen=max_steps)
        self.values = deque(maxlen=max_steps)
        init_value = np.zeros(self.shape) if init_value is None else init_value
        for k in range(-max_steps + 1, 1):
            self.times.append(k * dt)
            self.values.append(init_value.copy())

    def append(self, t: float, value: np.ndarray) -> None:
        self.times.append(t)
        self.values.append(value.copy())

    def get_delayed(self, t_query: float) -> np.ndarray:
        times_arr = np.array(self.times)
        t_query = np.clip(t_query, times_arr[0], times_arr[-1])
        idx = max(0, min(np.searchsorted(times_arr, t_query, side="right") - 1, len(times_arr) - 2))
        t_m, t_m1 = times_arr[idx], times_arr[idx + 1]
        if abs(t_m1 - t_m) < 1e-20:
            return self.values[idx].copy()
        alpha = (t_query - t_m) / (t_m1 - t_m)
        return (1.0 - alpha) * self.values[idx] + alpha * self.values[idx + 1]


class MultiHopFluidSimulator:
    """
    RK4 simulator for multi-hop fluid model with per-hop credits.

    Service discipline: credit-aware work-conserving max-min RR analog.
    Matches dragonfly-dally `get_next_router_vc_excess` (lines 3769-3800):
    - Eligible = has demand (offered > 0) AND downstream credit > 0.
    - Eligible flows share port capacity by work-conserving max-min fairness.
    - Credit-starved flows get zero service; backlog grows until credits recover.

    Credit dynamics (Option B: bottleneck return for source):
    - Source credit C_{i,0}: returns from final bottleneck hop b_i.
    - Per-hop credits C_{i,h} (h>=1): return from local service at hop h.

    Offered rate (credit-gated injection):
    - o_{i,h}(t) = min(lambda_i,   C_{i,0}(t) / credit_drain_time)  for h=0
    - o_{i,h}(t) = min(s_{i,h-1}(t), C_{i,h}(t) / credit_drain_time)  for h>0
    where credit_drain_time = vc_buffer_size / link_bandwidth (same for all flows, from CODES PARAMS).

    Credit ODE:
    - dC_{i,0}/dt  = s_{i,b}(t - D_{i,b})  - o_{i,0}(t)
    - dC_{i,h}/dt  = s_{i,h}(t - d_{i,h})  - o_{i,h}(t)   for h>=1
    where D_{i,b} is the cumulative backward delay to the bottleneck and
    d_{i,h} is the local credit-return delay at hop h.
    """

    def __init__(self, config: NetworkConfig, dt: float):
        self.config = config
        self.dt = dt
        self.total_hops = sum(len(f.route) for f in config.flows)
        self._build_index_maps()
        self._compute_bottlenecks()
        self._compute_max_delay()

    def _build_index_maps(self) -> None:
        self.hop_to_idx: Dict[Tuple[int, int], int] = {}
        self.idx_to_hop: Dict[int, Tuple[int, int]] = {}
        self.flow_by_id = {f.flow_id: f for f in self.config.flows}

        idx = 0
        for f in self.config.flows:
            for h in range(len(f.route)):
                self.hop_to_idx[(f.flow_id, h)] = idx
                self.idx_to_hop[idx] = (f.flow_id, h)
                idx += 1

        self.port_contenders: Dict[Tuple[int, int], List[int]] = {}
        for f in self.config.flows:
            for h, (r, p) in enumerate(f.route):
                self.port_contenders.setdefault((r, p), []).append(self.hop_to_idx[(f.flow_id, h)])

    def _compute_bottlenecks(self) -> None:
        self.final_bottleneck: Dict[int, int] = {}
        self.source_backward_delay: Dict[int, float] = {}

        for f in self.config.flows:
            last_contended = -1
            for h, (r, p) in enumerate(f.route):
                if len(self.port_contenders[(r, p)]) > 1:
                    last_contended = h
            bottleneck_hop = last_contended if last_contended >= 0 else len(f.route) - 1
            self.final_bottleneck[f.flow_id] = bottleneck_hop
            self.source_backward_delay[f.flow_id] = sum(
                f.backward_delays[h] if h < len(f.backward_delays) else 0.0
                for h in range(bottleneck_hop + 1)
            )

    def _compute_max_delay(self) -> None:
        max_fwd = max((max(f.forward_delays) if f.forward_delays else 0.0) for f in self.config.flows)
        max_local_bwd = max((max(f.backward_delays) if f.backward_delays else 0.0) for f in self.config.flows)
        max_source_bwd = max((self.source_backward_delay[f.flow_id] for f in self.config.flows), default=0.0)
        self.max_delay = max(max_fwd, max_local_bwd, max_source_bwd)
        self.max_delay_steps = int(np.ceil(self.max_delay / self.dt)) + 10

    def compute_offered_rates(self, C: np.ndarray, s: np.ndarray) -> np.ndarray:
        """
        Credit-limited offered rate at each hop.

        o_{i,0} = min(lambda_i,     C_{i,0} / credit_drain_time)
        o_{i,h} = min(s_{i,h-1},   C_{i,h} / credit_drain_time)   for h >= 1

        credit_drain_time = vc_buffer_size / link_bandwidth (from CODES PARAMS, same for all flows).
        When credits are full (C = buf_size = 132 chunks), credit_limit >> lambda,
        so offered = lambda (no constraint).  When C drops to one chunk,
        credit_limit = link_bandwidth, matching the CODES chunk-quantized gate.
        """
        offered = np.zeros(self.total_hops)
        for f in self.config.flows:
            for h in range(len(f.route)):
                idx = self.hop_to_idx[(f.flow_id, h)]
                credit_limit = max(0.0, C[idx]) / f.credit_drain_time
                if h == 0:
                    offered[idx] = min(f.lambda_rate, credit_limit)
                else:
                    offered[idx] = min(s[self.hop_to_idx[(f.flow_id, h - 1)]], credit_limit)
        return offered

    def compute_arrivals(self, t: float, o_hist: DelayHistory, s_hist: DelayHistory) -> np.ndarray:
        """
        Arrivals at each hop after the hop's forward propagation delay.

        Hop 0 receives the (delayed) offered rate from the terminal.
        Hop h>0 receives the (delayed) service from hop h-1.
        """
        arrivals = np.zeros(self.total_hops)
        for f in self.config.flows:
            for h in range(len(f.route)):
                idx = self.hop_to_idx[(f.flow_id, h)]
                delay = f.forward_delays[h] if h < len(f.forward_delays) else 0.0
                if h == 0:
                    arrivals[idx] = o_hist.get_delayed(t - delay)[idx]
                else:
                    prev_idx = self.hop_to_idx[(f.flow_id, h - 1)]
                    arrivals[idx] = s_hist.get_delayed(t - delay)[prev_idx]
        return arrivals

    def compute_service(self, offered_or_arrivals: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        Credit-aware work-conserving max-min service.

        Eligibility for flow (fid, h) at port (r, p):
          - has demand:  offered_or_arrivals[idx] > 0
          - has credit:  C[next_hop_idx] >= chunk_size (or last hop: no downstream gate)

        Matches dragonfly-dally get_next_router_vc_excess: credit-starved VCs
        are skipped entirely (no slot wasted), port capacity fully reallocated
        among eligible flows by work-conserving max-min fairness.
        """
        service = np.zeros(self.total_hops)
        for (r, p), contenders in self.port_contenders.items():
            mu = self.config.get_port_capacity(r, p)
            eligible = []
            for idx in contenders:
                fid, h = self.idx_to_hop[idx]
                f = self.flow_by_id[fid]
                if offered_or_arrivals[idx] <= 1e-20:
                    continue
                if h + 1 < len(f.route):
                    next_idx = self.hop_to_idx[(fid, h + 1)]
                    if C[next_idx] < self.config.chunk_size:
                        continue
                eligible.append(idx)

            if not eligible:
                continue

            demand = {idx: offered_or_arrivals[idx] for idx in eligible}
            remaining_mu = float(mu)
            remaining = list(eligible)
            while remaining and remaining_mu > 1e-20:
                fair_share = remaining_mu / len(remaining)
                unsaturated = [idx for idx in remaining if demand[idx] < fair_share]
                for idx in unsaturated:
                    service[idx] = demand[idx]
                    remaining_mu -= demand[idx]
                saturated = [idx for idx in remaining if demand[idx] >= fair_share]
                if not unsaturated:
                    for idx in saturated:
                        service[idx] = fair_share
                    break
                remaining = saturated
        return service

    def compute_derivatives(
        self, t: float, B: np.ndarray, C: np.ndarray,
        o_hist: DelayHistory, s_hist: DelayHistory
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (dB, dC, offered, service).

        dB_i_h = arrivals_{i,h}(t)  -  service_{i,h}(t)

        dC_{i,0}/dt = s_{i,b}(t - D_{i,b})  -  o_{i,0}(t)
        dC_{i,h}/dt = s_{i,h}(t - d_{i,h})  -  o_{i,h}(t)   for h >= 1

        The credit ODE is driven by *offered* (not service), matching the CODES
        semantics that credits at hop h are consumed by the upstream terminal/router
        at rate offered[h] and replenished by the downstream service at rate s[h].
        """
        s_current = s_hist.get_delayed(t)
        offered = self.compute_offered_rates(C, s_current)
        arrivals = self.compute_arrivals(t, o_hist, s_hist)
        service = self.compute_service(arrivals, C)

        dB = arrivals - service
        dC = np.zeros(self.total_hops)

        for f in self.config.flows:
            source_idx = self.hop_to_idx[(f.flow_id, 0)]
            bottleneck_idx = self.hop_to_idx[(f.flow_id, self.final_bottleneck[f.flow_id])]
            dC[source_idx] = (
                s_hist.get_delayed(t - self.source_backward_delay[f.flow_id])[bottleneck_idx]
                - offered[source_idx]
            )
            for h in range(1, len(f.route)):
                idx = self.hop_to_idx[(f.flow_id, h)]
                bwd_delay = f.backward_delays[h] if h < len(f.backward_delays) else 0.0
                dC[idx] = s_hist.get_delayed(t - bwd_delay)[idx] - offered[idx]

        return dB, dC, offered, service

    def rk4_step(
        self, t: float, B: np.ndarray, C: np.ndarray,
        o_hist: DelayHistory, s_hist: DelayHistory
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        k1_B, k1_C, _, _ = self.compute_derivatives(t, B, C, o_hist, s_hist)

        B2 = np.maximum(B + 0.5 * self.dt * k1_B, 0.0)
        C2 = np.maximum(C + 0.5 * self.dt * k1_C, 0.0)
        k2_B, k2_C, _, _ = self.compute_derivatives(t + 0.5 * self.dt, B2, C2, o_hist, s_hist)

        B3 = np.maximum(B + 0.5 * self.dt * k2_B, 0.0)
        C3 = np.maximum(C + 0.5 * self.dt * k2_C, 0.0)
        k3_B, k3_C, _, _ = self.compute_derivatives(t + 0.5 * self.dt, B3, C3, o_hist, s_hist)

        B4 = np.maximum(B + self.dt * k3_B, 0.0)
        C4 = np.maximum(C + self.dt * k3_C, 0.0)
        k4_B, k4_C, _, _ = self.compute_derivatives(t + self.dt, B4, C4, o_hist, s_hist)

        B_new = np.maximum(B + (self.dt / 6.0) * (k1_B + 2 * k2_B + 2 * k3_B + k4_B), 0.0)
        C_new = np.maximum(C + (self.dt / 6.0) * (k1_C + 2 * k2_C + 2 * k3_C + k4_C), 0.0)

        s_current = s_hist.get_delayed(t + self.dt)
        o_new = self.compute_offered_rates(C_new, s_current)
        arrivals_new = self.compute_arrivals(t + self.dt, o_hist, s_hist)
        s_new = self.compute_service(arrivals_new, C_new)

        return B_new, C_new, o_new, s_new

    def compute_initial_values(
        self, C: np.ndarray, B: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        offered = np.zeros(self.total_hops)
        for f in self.config.flows:
            for h in range(len(f.route)):
                idx = self.hop_to_idx[(f.flow_id, h)]
                credit_limit = max(0.0, C[idx]) / f.credit_drain_time
                if h == 0:
                    offered[idx] = min(f.lambda_rate, credit_limit)
                else:
                    prev_idx = self.hop_to_idx[(f.flow_id, h - 1)]
                    offered[idx] = min(offered[prev_idx], credit_limit)
        return offered, self.compute_service(offered, C)

    def run(self, T_total: float, verbose: bool = True) -> Dict:
        n_steps = int(np.ceil(T_total / self.dt))
        B = np.zeros(self.total_hops)
        C = np.zeros(self.total_hops)
        for f in self.config.flows:
            for h in range(len(f.route)):
                idx = self.hop_to_idx[(f.flow_id, h)]
                C[idx] = f.initial_credits_per_hop[h] if h < len(f.initial_credits_per_hop) else 8448.0
                B[idx] = f.initial_backlogs[h] if h < len(f.initial_backlogs) else 0.0

        o_init, s_init = self.compute_initial_values(C, B)
        o_hist = DelayHistory(self.total_hops, self.max_delay_steps, self.dt, o_init)
        s_hist = DelayHistory(self.total_hops, self.max_delay_steps, self.dt, s_init)

        t_arr = np.zeros(n_steps + 1)
        B_arr = np.zeros((n_steps + 1, self.total_hops))
        C_arr = np.zeros((n_steps + 1, self.total_hops))
        o_arr = np.zeros((n_steps + 1, self.total_hops))
        s_arr = np.zeros((n_steps + 1, self.total_hops))
        B_arr[0], C_arr[0], o_arr[0], s_arr[0] = B.copy(), C.copy(), o_init.copy(), s_init.copy()

        for k in range(n_steps):
            t_k = k * self.dt
            B, C, o_new, s_new = self.rk4_step(t_k, B, C, o_hist, s_hist)
            t_k1 = (k + 1) * self.dt
            o_hist.append(t_k1, o_new)
            s_hist.append(t_k1, s_new)
            t_arr[k + 1] = t_k1
            B_arr[k + 1], C_arr[k + 1] = B.copy(), C.copy()
            o_arr[k + 1], s_arr[k + 1] = o_new.copy(), s_new.copy()
            if verbose and (k + 1) % 2000 == 0:
                print(f"  Step {k + 1}/{n_steps} ({100*(k+1)/n_steps:.1f}%)")

        if verbose:
            print(f"Simulation complete. {len(t_arr)} time points.")

        return {
            "t": t_arr, "B": B_arr, "C": C_arr,
            "o": o_arr, "s": s_arr,
        }
