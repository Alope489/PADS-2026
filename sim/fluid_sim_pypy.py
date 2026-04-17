"""
PyPy-compatible RK4 fluid simulation using array.array.

No NumPy, no Numba — pure Python with typed double arrays so PyPy's
tracing JIT can compile the hot loops to efficient machine code.

Drop-in replacement for fluid_sim_numba.run_simulation_numba().

2-D result arrays use **flat row-major** layout:
    access flow *j* at step *k*  →  ``result[k * n_flows + j]``

Use :func:`column` to pull a single flow's time-series out of a flat result.
"""
from array import array
import math


# ── tiny helpers ────────────────────────────────────────────────────────────

def _d(n):
    """Zero-filled double array of length *n*."""
    return array('d', [0.0]) * n


def column(flat, n_flows, j, n_rows):
    """Extract flow *j* from a flat (n_rows × n_flows) result array.

    Returns a plain ``list`` suitable for matplotlib or CSV export.
    """
    return [flat[k * n_flows + j] for k in range(n_rows)]


# ── ring-buffer delay interpolation ────────────────────────────────────────

def _get_delayed(times, vals, wr, ms, nf, tq):
    """Linearly-interpolated lookup in a ring buffer.

    Parameters
    ----------
    times : array('d')  ring-buffer timestamps, length *ms*
    vals  : array('d')  flat 2-D ring buffer, shape (ms, nf) row-major
    wr    : int          current write index
    ms    : int          ring-buffer capacity  (max_steps)
    nf    : int          number of flows
    tq    : float        query time
    """
    t_old = times[wr]
    t_new = times[(wr + ms - 1) % ms]
    if tq < t_old:
        tq = t_old
    if tq > t_new:
        tq = t_new

    for i in range(ms - 1):
        lo = (wr + i) % ms
        hi = (wr + i + 1) % ms
        t_lo, t_hi = times[lo], times[hi]
        if t_lo <= tq <= t_hi:
            out = _d(nf)
            if (t_hi - t_lo) < 1e-30:
                base = lo * nf
                for j in range(nf):
                    out[j] = vals[base + j]
            else:
                alpha = (tq - t_lo) / (t_hi - t_lo)
                blo, bhi = lo * nf, hi * nf
                for j in range(nf):
                    out[j] = (1.0 - alpha) * vals[blo + j] + alpha * vals[bhi + j]
            return out

    out = _d(nf)
    base = ((wr + ms - 1) % ms) * nf
    for j in range(nf):
        out[j] = vals[base + j]
    return out


# ── service share (injection-proportional) ─────────────────────────────────

def _share_inj_prop(x_del, mu, nf):
    """s_i = mu · x_del_i / Σ x_del_j."""
    A = 0.0
    for j in range(nf):
        A += x_del[j]
    if A > 1e-20:
        out = _d(nf)
        for j in range(nf):
            out[j] = mu * x_del[j] / A
        return out
    return _d(nf)


# ── single RK4 step ────────────────────────────────────────────────────────

def _rk4_step(B_k, C_k, t_k, dt,
              xt, xv, xw, st, sv, sw,
              ms, nf, D_f, D_b, lam, mu, tau):
    """One full RK4 step.  Returns (B_k1, C_k1, x_next, s_next)."""

    # ---- stage 1 --------------------------------------------------------
    xd1 = _get_delayed(xt, xv, xw, ms, nf, t_k - D_f)
    sd1 = _get_delayed(st, sv, sw, ms, nf, t_k - D_b)
    s1  = _share_inj_prop(xd1, mu, nf)
    k1B = _d(nf); k1C = _d(nf)
    for j in range(nf):
        k1B[j] = xd1[j] - s1[j]
        k1C[j] = sd1[j] - (lam[j] if lam[j] < C_k[j] / tau else C_k[j] / tau)

    # ---- stage 2 --------------------------------------------------------
    C2 = _d(nf)
    for j in range(nf):
        C2[j] = C_k[j] + 0.5 * dt * k1C[j]
    xd2 = _get_delayed(xt, xv, xw, ms, nf, t_k + 0.5 * dt - D_f)
    sd2 = _get_delayed(st, sv, sw, ms, nf, t_k + 0.5 * dt - D_b)
    s2  = _share_inj_prop(xd2, mu, nf)
    k2B = _d(nf); k2C = _d(nf)
    for j in range(nf):
        k2B[j] = xd2[j] - s2[j]
        k2C[j] = sd2[j] - (lam[j] if lam[j] < C2[j] / tau else C2[j] / tau)

    # ---- stage 3 --------------------------------------------------------
    C3 = _d(nf)
    for j in range(nf):
        C3[j] = C_k[j] + 0.5 * dt * k2C[j]
    xd3 = _get_delayed(xt, xv, xw, ms, nf, t_k + 0.5 * dt - D_f)
    sd3 = _get_delayed(st, sv, sw, ms, nf, t_k + 0.5 * dt - D_b)
    s3  = _share_inj_prop(xd3, mu, nf)
    k3B = _d(nf); k3C = _d(nf)
    for j in range(nf):
        k3B[j] = xd3[j] - s3[j]
        k3C[j] = sd3[j] - (lam[j] if lam[j] < C3[j] / tau else C3[j] / tau)

    # ---- stage 4 --------------------------------------------------------
    C4 = _d(nf)
    for j in range(nf):
        C4[j] = C_k[j] + dt * k3C[j]
    xd4 = _get_delayed(xt, xv, xw, ms, nf, t_k + dt - D_f)
    sd4 = _get_delayed(st, sv, sw, ms, nf, t_k + dt - D_b)
    s4  = _share_inj_prop(xd4, mu, nf)
    k4B = _d(nf); k4C = _d(nf)
    for j in range(nf):
        k4B[j] = xd4[j] - s4[j]
        k4C[j] = sd4[j] - (lam[j] if lam[j] < C4[j] / tau else C4[j] / tau)

    # ---- combine (RK4 weighted sum, clamped ≥ 0) -------------------------
    c = dt / 6.0
    B1 = _d(nf); C1 = _d(nf)
    for j in range(nf):
        B1[j] = B_k[j] + c * (k1B[j] + 2.0 * k2B[j] + 2.0 * k3B[j] + k4B[j])
        C1[j] = C_k[j] + c * (k1C[j] + 2.0 * k2C[j] + 2.0 * k3C[j] + k4C[j])
        if B1[j] < 0.0:
            B1[j] = 0.0
        if C1[j] < 0.0:
            C1[j] = 0.0

    # ---- values for next history entry ------------------------------------
    xn = _d(nf)
    for j in range(nf):
        xn[j] = lam[j] if lam[j] < C1[j] / tau else C1[j] / tau
    return B1, C1, xn, _share_inj_prop(
        _get_delayed(xt, xv, xw, ms, nf, t_k + dt - D_f), mu, nf
    )


# ── core simulation loop ───────────────────────────────────────────────────

def _run_sim(n_steps, ms, nf, dt, D_f, D_b, lam, mu, tau, B0, C0):
    """Inner loop — everything is already converted to array('d')."""

    # initial injection & service
    x0 = _d(nf)
    for j in range(nf):
        x0[j] = lam[j] if lam[j] < C0[j] / tau else C0[j] / tau
    A = 0.0
    for j in range(nf):
        A += x0[j]
    s0 = _d(nf)
    if A > 1e-20:
        for j in range(nf):
            s0[j] = mu * x0[j] / A

    # ring buffers  (flat 2-D: ms × nf)
    xt = _d(ms);        xv = _d(ms * nf)
    st = _d(ms);        sv = _d(ms * nf)
    for i in range(ms):
        xt[i] = (i - ms + 1) * dt
        st[i] = (i - ms + 1) * dt
        for j in range(nf):
            xv[i * nf + j] = x0[j]
            sv[i * nf + j] = s0[j]
    xw = 0;  sw = 0          # write cursors

    # result storage (flat 2-D row-major, plus 1-D t)
    rows = n_steps + 1
    t_arr = _d(rows)
    B_arr = _d(rows * nf)
    C_arr = _d(rows * nf)
    x_arr = _d(rows * nf)
    s_arr = _d(rows * nf)

    # row 0
    for j in range(nf):
        B_arr[j] = B0[j]
        C_arr[j] = C0[j]
        x_arr[j] = x0[j]
        s_arr[j] = s0[j]

    B = array('d', B0)
    C = array('d', C0)

    for k in range(n_steps):
        B, C, xn, sn = _rk4_step(
            B, C, k * dt, dt,
            xt, xv, xw, st, sv, sw,
            ms, nf, D_f, D_b, lam, mu, tau,
        )
        t_k1 = (k + 1) * dt
        xt[xw] = t_k1
        st[sw] = t_k1
        for j in range(nf):
            xv[xw * nf + j] = xn[j]
            sv[sw * nf + j] = sn[j]
        xw = (xw + 1) % ms
        sw = (sw + 1) % ms

        t_arr[k + 1] = t_k1
        row = (k + 1) * nf
        for j in range(nf):
            B_arr[row + j] = B[j]
            C_arr[row + j] = C[j]
            x_arr[row + j] = xn[j]
            s_arr[row + j] = sn[j]

    return t_arr, B_arr, C_arr, x_arr, s_arr


# ── public API ──────────────────────────────────────────────────────────────

def run_simulation_pypy(N, T_total, dt, D_f, D_b,
                        lambda_arr, mu, tau, B_init, C_init):
    """Run the RK4 credit + buffer fluid simulation — pure Python / PyPy.

    Parameters are identical to ``run_simulation_numba``.
    *lambda_arr*, *B_init*, *C_init* may be any iterable of floats.

    Returns
    -------
    t_arr : array.array('d')
        Time stamps, length ``n_steps + 1``.
    B_arr : array.array('d')
        Backlog, flat ``(n_steps+1) × N`` row-major.
    C_arr : array.array('d')
        Credits, same layout.
    x_arr : array.array('d')
        Injection rates, same layout.
    s_arr : array.array('d')
        Service rates, same layout.

    Access flow *j* at step *k*::

        value = B_arr[k * N + j]

    Or use the helper::

        col = column(B_arr, N, j, n_steps + 1)
    """
    return _run_sim(
        int(math.ceil(T_total / dt)),
        max(int(math.ceil(D_f / dt)), int(math.ceil(D_b / dt))) + 4,
        N, dt, D_f, D_b,
        array('d', lambda_arr), mu, tau,
        array('d', B_init), array('d', C_init),
    )
