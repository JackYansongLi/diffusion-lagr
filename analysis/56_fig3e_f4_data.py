#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""56_fig3e_f4_data.py -- data for Fig. 3e: the per-unit S2 and S4 of the four stages A/B/C/D and of
the DNS.

Conventions (see RESULTS_ANALYSIS.md section 7.15)
-----------------------------------
- F_tau^(4) = S4(tau)/S2(tau)^2 with S_p(tau) = <(d_tau V)^p>, in physical units (de-normalised with
the global u1c scalars, the same constants as in 54)
- stages A/B/C/D: the 358,400 trajectories cut **in order** into 10 blocks of 35,840, the DM panel
  convention of section 7.14
- DNS: `Lagr_u3c_diffusion.h5` (3 components, 327,680 trajectories) as 10 trajectory blocks x 3
  components = **30 units**
  -- the DNS panel convention of section 7.14 (components kept apart)
- engines:
  * `fft` (default): S2 and S4 are expanded in power correlations, an exact identity about two orders
    of magnitude faster than the direct sum;
      Σ_t (x_{t+τ}−x_t)²  = Σ x_{t+τ}² + Σ x_t² − 2Σ x_{t+τ} x_t
      Σ_t (x_{t+τ}−x_t)⁴  = Σ x_{t+τ}⁴ + Σ x_t⁴ − 4Σ x_{t+τ}³x_t + 6Σ x_{t+τ}²x_t² − 4Σ x_{t+τ}x_t³
    every sum is restricted to the **overlap window** t in [0, K-1-tau] (the power terms come from the
    prefix sums `c[:, K-1-tau]` and `tot - c[:, tau-1]`,
    the cross terms from a zero-padded FFT with padding >= 2K, so nothing wraps);
  * `direct`: the per-lag direct sum, written as in 16_fig4a_batches.py, used as the reference for the
    FFT engine.
- self-check: the per-block S2/S4 of stage A against `fig4a_batches_dm1c_repro.npz`, and the DNS
  against `fig4a_batches_dns.npz`
  (both are products of the direct sum, so this validates the FFT engine and the batching at once)
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data")
RESULTS = os.path.join(ROOT, "results")

LAGS = np.arange(1, 1025)                      # tau = 0.1*lag, so the grid aligns point by point with the 1024 lags of Figure3e
U1C_MIN, U1C_MAX = -11.24481512, 10.31416093   # the global u1c scalars, the same as in 54 and in the dataset convention
BLOCK = 512                                    # trajectories per block (the FFT engine holds a (BLOCK, 2K) complex array, about 67 MB)


def rss_gb():
    """Peak memory (Windows has no resource module, so psutil is used; None if neither is available)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    except Exception:
        pass
    try:
        import psutil
        p = psutil.Process(os.getpid())
        m = p.memory_info().rss
        for c in p.children(recursive=True):
            try:
                m = max(m, c.memory_info().rss)
            except Exception:
                pass
        return m / 1024 ** 3
    except Exception:
        return None


def unit_sums(x, engine):
    """x:(B,K) float64 in physical units -> (S2sum[nl], S4sum[nl]): the block sum of d^2 and of d^4 over
the overlap window."""
    K = x.shape[1]
    lag = LAGS
    if engine == "direct":
        S2 = np.zeros(len(lag))
        S4 = np.zeros(len(lag))
        for i, tau in enumerate(lag):
            d = x[:, tau:] - x[:, :K - tau]
            d2 = d * d
            S2[i] = d2.sum()
            S4[i] = (d2 * d2).sum()
            del d, d2
        return S2, S4
    n = 1
    while n < 2 * K:
        n <<= 1
    F1 = np.fft.rfft(x, n=n, axis=1)
    x2 = x * x
    F2 = np.fft.rfft(x2, n=n, axis=1)
    ac11 = np.fft.irfft(F1 * np.conj(F1), n=n, axis=1)[:, :K]
    ac22 = np.fft.irfft(F2 * np.conj(F2), n=n, axis=1)[:, :K]
    del F2
    x3 = x2 * x
    F3 = np.fft.rfft(x3, n=n, axis=1)
    ac31 = np.fft.irfft(F3 * np.conj(F1), n=n, axis=1)[:, :K]   # Σ x_{t+τ}³ x_t
    ac13 = np.fft.irfft(F1 * np.conj(F3), n=n, axis=1)[:, :K]   # Σ x_{t+τ} x_t³
    del F3, F1, x3
    c2 = np.cumsum(x2, axis=1)
    c4 = np.cumsum(x2 * x2, axis=1)
    tot2, tot4 = c2[:, -1], c4[:, -1]
    del x2
    A2 = c2[:, K - 1 - lag]
    B2 = tot2[:, None] - c2[:, lag - 1]
    A4 = c4[:, K - 1 - lag]
    B4 = tot4[:, None] - c4[:, lag - 1]
    S2 = (A2 + B2 - 2.0 * ac11[:, lag]).sum(axis=0)
    S4 = (A4 + B4 - 4.0 * ac31[:, lag] + 6.0 * ac22[:, lag] - 4.0 * ac13[:, lag]).sum(axis=0)
    del c2, c4, A2, B2, A4, B4, ac11, ac22, ac31, ac13
    return S2, S4


def worker(task):
    """(ds, unit, path, i0, i1, comp, engine) → (unit, S2sum[nl], S4sum[nl], B, comp)

    comp<0 selects a single-component dataset (the stages) and comp>=0 one component of the DNS.
    De-normalisation happens in physical units, as in 54.
    """
    ds, unit, path, i0, i1, comp, engine = task
    if ds == "stages":
        a = np.load(path, mmap_mode="r")
        u = np.asarray(a[i0:i1], dtype=np.float32)[:, :, 0]
        mn, mx = np.float32(U1C_MIN), np.float32(U1C_MAX)
        x = ((u + 1.0) * ((mx - mn) / 2.0) + mn).astype(np.float64)
        del u
    else:
        import h5py
        with h5py.File(path, "r") as f:
            # the min/max of a three-component dataset are **per-component** arrays with different constants, so
            # a scalar must be picked per component
            mnv = np.asarray(f["min"][()]).ravel()
            mxv = np.asarray(f["max"][()]).ravel()
            mn = np.float32(mnv[comp] if mnv.size > 1 else mnv[0])
            mx = np.float32(mxv[comp] if mxv.size > 1 else mxv[0])
            arr = f["train"]
            u = np.asarray(arr[i0:i1, :, comp], dtype=np.float32)
        x = ((u + 1.0) * ((mx - mn) / 2.0) + mn).astype(np.float64)
        del u
    s2, s4 = unit_sums(x, engine)
    del x
    return unit, s2, s4, i1 - i0, comp


def build_plan_stages(block):
    """Four stages: 10 consecutive blocks each, every block cut into chunks, giving the task list and the
    unit indices."""
    tasks, meta = [], []
    for idx, (n, tag) in enumerate(((416, "D"), (216, "C"), (48, "B"), (0, "A"))):
        path = os.path.join(DATA, "tmp", "dm1c_repro_stage%d_f32.npy" % n)
        a = np.load(path, mmap_mode="r")
        n_traj = a.shape[0]
        del a
        bs = n_traj // 10
        for b in range(10):
            j0 = b * bs
            j1 = n_traj if b == 9 else (b + 1) * bs
            for i0 in range(j0, j1, block):
                tasks.append(("stages", (tag, b), path, i0, min(i0 + block, j1), -1, ENGINE))
        meta.append((tag, n, bs))
    return tasks, meta


def build_plan_dns(block, n_units_chunks=10):
    """DNS: 10 trajectory blocks x 3 components = 30 units (components kept apart, as in section 7.14)."""
    import h5py
    path = os.path.join(DATA, "Lagr_u3c_diffusion.h5")
    with h5py.File(path, "r") as f:
        n_traj = f["train"].shape[0]
        C = f["train"].shape[2]
    bs = n_traj // n_units_chunks
    tasks = []
    for b in range(n_units_chunks):
        j0 = b * bs
        j1 = n_traj if b == n_units_chunks - 1 else (b + 1) * bs
        for c in range(C):
            for i0 in range(j0, j1, block):
                tasks.append(("dns", (b, c), path, i0, min(i0 + block, j1), c, ENGINE))
    return tasks, n_traj, C


def run(tasks, n_units, workers, say):
    """Accumulate sum d^2, sum d^4 and the counts per unit, returning the per-unit mean curve."""
    S2s = np.zeros((n_units, len(LAGS)))
    S4s = np.zeros((n_units, len(LAGS)))
    Cnt = np.zeros(n_units)                          # the number of trajectories in every unit (a scalar) ...
    done = 0
    t0 = time.time()
    with Pool(workers) as pool:
        for unit, s2, s4, b, comp in pool.imap_unordered(worker, tasks, chunksize=1):
            S2s[unit] += s2
            S4s[unit] += s4
            Cnt[unit] += b
            done += 1
            if done % 50 == 0 or done == len(tasks):
                say("      %d/%d blocks (%.0fs)" % (done, len(tasks), time.time() - t0))
    # ...each trajectory contributes (K-tau) increments at lag tau, so the count is trajectories*(2000-tau),
    # exactly as in 16 and 54
    Cnt = np.maximum(Cnt[:, None] * (2000 - LAGS), 1)
    return S2s / Cnt, S4s / Cnt


def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="both", choices=["stages", "dns", "both"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--engine", default="fft", choices=["fft", "direct"])
    ap.add_argument("--limit-blocks", type=int, default=0,
                    help="use only the first N blocks (smoke test; the self-check against the direct-sum products is then skipped, since the sample is incomplete)")
    ap.add_argument("--out-stages", default=os.path.join(RESULTS, "fig3e_f4_dm1c_repro.npz"))
    ap.add_argument("--out-dns", default=os.path.join(RESULTS, "fig3e_f4_dns.npz"))
    args = ap.parse_args()
    ENGINE = args.engine

    def say(s):
        print(s, flush=True)

    def build_stage_plan_with_engine(block):
        tasks, meta = [], []
        for n, tag in ((416, "D"), (216, "C"), (48, "B"), (0, "A")):
            path = os.path.join(DATA, "tmp", "dm1c_repro_stage%d_f32.npy" % n)
            a = np.load(path, mmap_mode="r")
            n_traj = a.shape[0]
            del a
            bs = n_traj // 10
            for b in range(10):
                j0 = b * bs
                j1 = n_traj if b == 9 else (b + 1) * bs
                for i0 in range(j0, j1, block):
                    tasks.append(("stages", ("%s_%d" % (tag, b), ), path, i0,
                                  min(i0 + block, j1), -1, ENGINE))
            meta.append((tag, n, bs))
        return tasks, meta

    say("[56] engine=%s block=%d workers=%d" % (args.engine, args.block, args.workers))

    if args.dataset in ("stages", "both"):
        tasks, meta = build_stage_plan_with_engine(args.block)
        if args.limit_blocks:
            tasks = tasks[:args.limit_blocks]
        # unit numbering, keyed by (tag, batch)
        unit_index, order = {}, []
        for t in tasks:
            k = t[1][0]
            if k not in unit_index:
                unit_index[k] = len(order)
                order.append(k)
        tasks = [(t[0], unit_index[t[1][0]], t[2], t[3], t[4], t[5], t[6]) for t in tasks]
        say("[56] stage tasks: %d blocks, %d units: %s" % (len(tasks), len(order), order[:4] + ["..."] * (len(order) > 4)))
        S2b, S4b = run(tasks, len(order), args.workers, say)
        np.savez_compressed(args.out_stages, lags=LAGS, order=np.array(order),
                            S2b=S2b, S4b=S4b)
        say("[56] → %s" % args.out_stages)
        for i, k in enumerate(order):
            f = S4b[i] / S2b[i] ** 2
            say("     %-6s S2(τ=1)=%.4e S4(τ=1)=%.4e F(τ=1)=%.4f  F(τ=10)=%.4f"
                % (k, S2b[i][9], S4b[i][9], f[9], f[99]))
        # self-check: the blocks of stage A against fig4a_batches_dm1c_repro.npz (a direct-sum product)
        ref_p = os.path.join(RESULTS, "fig4a_batches_dm1c_repro.npz")
        if args.limit_blocks:
            say("[56] self-check skipped (--limit-blocks => incomplete sample)")
        elif os.path.exists(ref_p):
            ref = np.load(ref_p)
            ours2 = np.array([S2b[order.index("A_%d" % b)] for b in range(10)])
            ours4 = np.array([S4b[order.index("A_%d" % b)] for b in range(10)])
            for key, ours in (("S2b", ours2), ("S4b", ours4)):
                r = ref[key]
                d = np.max(np.abs(ours - r) / np.abs(r))
                say("[56] self-check stage A %s against fig4a_batches_dm1c_repro: max relative difference %.3e %s"
                    % (key, d, "OK" if d < 1e-6 else "MISMATCH"))
        r = rss_gb()
        say("[56] peak RSS %s" % ("%.2f GB" % r if r else "n/a"))

    if args.dataset in ("dns", "both"):
        tasks, n_traj, C = build_plan_dns(args.block)
        units = []
        for t in tasks:
            if t[1] not in units:
                units.append(t[1])
        tasks = [(t[0], units.index(t[1]), t[2], t[3], t[4], t[5], t[6]) for t in tasks]
        say("[56] DNS tasks: %d blocks, %d units (%d trajectory blocks x %d components), n_traj=%d"
            % (len(tasks), len(units), 10, C, n_traj))
        S2u, S4u = run(tasks, len(units), args.workers, say)
        np.savez_compressed(args.out_dns, lags=LAGS, units=np.array(units),
                            S2u=S2u, S4u=S4u, n_traj=n_traj, n_comp=C)
        say("[56] → %s" % args.out_dns)
        F = S4u / S2u ** 2
        say("     DNS F(tau=1) per unit, median %.4f (%.3f..%.3f) | unit mean %.4f"
            % (np.median(F[:, 9]), F[:, 9].min(), F[:, 9].max(), F[:, 9].mean()))
        ref_p = os.path.join(RESULTS, "fig4a_batches_dns.npz")
        if args.limit_blocks:
            say("[56] self-check skipped (--limit-blocks)")
        elif os.path.exists(ref_p):
            ref = np.load(ref_p)
            for key, ours_u, refk in (("S2", S2u, "S2b"), ("S4", S4u, "S4b")):
                pooled = ours_u.reshape(10, C, len(LAGS)).mean(axis=1)   # averaged over the components, so it is comparable with a pooled block
                r = ref[refk]
                d = np.max(np.abs(pooled - r) / np.abs(r))
                say("[56] self-check DNS %s (components averaged) against fig4a_batches_dns[%s]: max relative difference %.3e %s"
                    % (key, refk, d, "OK" if d < 1e-5 else "MISMATCH (check the component order and the convention)"))
        r = rss_gb()
        say("[56] peak RSS %s" % ("%.2f GB" % r if r else "n/a"))


if __name__ == "__main__":
    ENGINE = "fft"
    main()
