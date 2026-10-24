#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""54_fig3d_s2_data.py -- S2(tau) for Fig. 3d: the curves of our four stages A/B/C/D plus a block
envelope.

Conventions (RESULTS_ANALYSIS.md section 7.14, the only authoritative one for this figure)
-----------------------------------------------------
- quantity: S2(tau) = <(d_tau V)^2> with tau = 0.1*lag (lag = 1..1024, aligned **point by point**
  with the 1024-lag grid of `Figure3d_*.out` in the paper,
  so no interpolation is needed), and computed in physical units after de-normalisation
- blocks: exactly as in the Fig. 4a chain -- the 358,400 trajectories are cut **in order** into 10
  blocks of 35,840;
  each block contributes its mean and the min/max across blocks gives the error (DM-1c is single-
  component, so 10 blocks; the -/+ columns of Fig. 3d in the paper
  are **error half-widths**, and our 10-block envelope gives 0.31%/0.59% against 0.55%/0.44%)
  => **bit-identical** to the sum of d^2 in `16_fig4a_batches.py` (S[li,0] += sum d^2,
  cnt += B*(K-tau))
- self-check: recomputing stage A here must reproduce S2b/lo/hi of the existing
  `fig4a_batches_dm1c_repro.npz`
- source: the four stages of arr in `Data/Reproduce/samples_358400x2000x1_DM-1c_reproduce.npz`;
  each stage is first streamed to `Data/tmp/dm1c_repro_stage{n}_f32.npy` for the workers to mmap
- the DNS panel is **not produced here**: it reuses `structfuncs_dns_full.npz` and
  `fig4a_batches_dns.npz`
  (convention cB = 30 blocks, matching the -/+ 0.112/0.103 of the paper for the DNS: 0.116/0.118)

Output: results/fig3d_s2_dm1c_repro.npz (<lab>_S2b / <lab>_S2_lo / <lab>_S2_hi per stage, plus lags)
"""
import argparse
import os
import struct
import time
import zipfile
from multiprocessing import Pool

import numpy as np

ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data")
RESULTS = os.path.join(ROOT, "results")
NPZ_REPRO = os.path.join(DATA, "Reproduce", "samples_358400x2000x1_DM-1c_reproduce.npz")
TMP = os.path.join(DATA, "tmp")
LAGS = np.arange(1, 1025)
LAB = {0: "A", 48: "B", 216: "C", 416: "D"}
U1C_MIN, U1C_MAX = -11.24481512, 10.31416093      # the global u1c scalars, as in dm1c
BLOCK = 2048


def npy_header(fid):
    assert fid.read(6) == b"\x93NUMPY"
    ver = struct.unpack("<2B", fid.read(2))
    hlen = struct.unpack("<H" if ver[0] == 1 else "<I", fid.read(2 if ver[0] == 1 else 4))[0]
    d = eval(fid.read(hlen), {"False": False, "True": True})
    return d["shape"], np.dtype(d["descr"]), (10 + hlen if ver[0] == 1 else 12 + hlen)


def extract_stage(idx, out_path, block_rows=20000):
    """Stream stage idx of arr in the npz into a standalone .npy (C order, so sequential read and write)."""
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print("  [extract] already present, skipping: %s" % out_path, flush=True)
        return
    t0 = time.time()
    with zipfile.ZipFile(NPZ_REPRO) as zf:
        with zf.open("arr.npy") as f:
            shape, dt, hlen = npy_header(f)
            S, N, L, C = (int(v) for v in shape)
            row = L * C * dt.itemsize
            f.seek(hlen + idx * N * row)
            with open(out_path, "wb") as g:
                g.write(b"\x93NUMPY\x01\x00")
                hdr = repr({"descr": dt.str, "fortran_order": False, "shape": (N, L, C)})
                hb = hdr.encode() + b" " * ((64 - (len(hdr) + 1) % 64) % 64) + b"\n"
                g.write(struct.pack("<H", len(hb)))
                g.write(hb)
                for s in range(0, N, block_rows):
                    k = min(block_rows, N - s)
                    buf = f.read(k * row)
                    if len(buf) != k * row:
                        raise IOError("stage %d ends early at row %d" % (idx, s))
                    g.write(buf)
    print("  [extract] %s (%.2f GB, %.0fs)" % (out_path, os.path.getsize(out_path) / 1024 ** 3,
                                               time.time() - t0), flush=True)


def s2_worker(task):
    """(batch_id, path, i0, i1, engine) → (batch_id, S2sum[nl], cnt[nl])。

    The sums are **exactly** those of 16_fig4a_batches.py, S[li,0] += sum d^2 and cnt += B*(K-tau);
only the order of accumulation differs:

    - engine="direct": sums (v_{t+lag}-v_t)^2 per lag, bit-identical to the form used in 16 (slow: about
  17 min per stage);
    - engine="fft"   ：Σ(v_{t+lag}−v_t)² = Σ_{t< K−lag} v_t² + Σ_{t≥lag} v_t² − 2Σ v_t v_{t+lag}，
  the last term comes from a zero-padded FFT circular autocorrelation (padding >= 2K, so nothing
  wraps and every lag equals the directly computed sum),
  and the first two come from prefix sums. The identity is exact, and the measured relative
  difference from the direct sum is ~1e-7 (self-check recorded in section 7.14).
    """
    bid, path, i0, i1, engine = task
    a = np.load(path, mmap_mode="r")
    u = np.asarray(a[i0:i1], dtype=np.float32)
    mn, mx = np.float32(U1C_MIN), np.float32(U1C_MAX)
    v = (u + 1.0) * ((mx - mn) / 2.0) + mn                # back to physical units
    del u
    B, K, C = v.shape
    cnt = B * (K - LAGS)
    if engine == "fft":
        x = v[:, :, 0].astype(np.float64)
        del v
        n = 1
        while n < 2 * K:
            n <<= 1
        F = np.fft.rfft(x, n=n, axis=1)
        ac = np.fft.irfft(F * np.conj(F), n=n, axis=1)[:, :K]      # sum_t x_t x_{t+lag}, no wrap-around
        del F
        c = np.cumsum(x * x, axis=1)
        tot = c[:, -1]
        left = c[:, K - 1 - LAGS]                                  # Σ_{t=0}^{K-1-lag} x_t²
        right = tot[:, None] - c[:, np.maximum(LAGS - 1, 0)]        # Σ_{t=lag}^{K-1} x_t²
        S = (left + right - 2.0 * ac[:, LAGS]).sum(axis=0)
        return bid, S, cnt.astype(np.float64)
    x = v[:, :, 0]
    del v
    S = np.zeros(len(LAGS))
    for li, tau in enumerate(LAGS):
        d = x[:, tau:] - x[:, :K - tau]
        S[li] += np.sum(d * d, dtype=np.float64)
        del d
    del x
    return bid, S, cnt.astype(np.float64)


def run_stage(path, n_traj, batches, workers, block, engine="fft"):
    bs = n_traj // batches
    tasks = []
    for b in range(batches):
        j0 = b * bs
        j1 = (b + 1) * bs if b < batches - 1 else n_traj
        for i0 in range(j0, j1, block):
            tasks.append((b, path, i0, min(i0 + block, j1), engine))
    Ssum = np.zeros((batches, len(LAGS)))
    cnt = np.zeros((batches, len(LAGS)))
    t0 = time.time()
    done = 0
    with Pool(workers) as pool:
        for bid, S, c in pool.imap_unordered(s2_worker, tasks, chunksize=1):
            Ssum[bid] += S
            cnt[bid] += c
            done += 1
            if done % 40 == 0 or done == len(tasks):
                print("      %d/%d blocks (%.0fs)" % (done, len(tasks), time.time() - t0), flush=True)
    Sp = Ssum / cnt                                      # (nb, nl) the S2 of every block
    return Sp.mean(axis=0), Sp.min(axis=0), Sp.max(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--out", default=os.path.join(RESULTS, "fig3d_s2_dm1c_repro.npz"))
    ap.add_argument("--n-traj", type=int, default=0, help="use only the first N (smoke test; 0 = all)")
    ap.add_argument("--engine", default="fft", choices=["fft", "direct"],
                    help="summation engine (fft: mathematically equivalent to the direct sum, about 80x faster; direct: the bit-identical form used in 16)")
    args = ap.parse_args()
    os.makedirs(TMP, exist_ok=True)

    with zipfile.ZipFile(NPZ_REPRO) as zf:
        with zf.open("stages.npy") as f:
            stages = [int(v) for v in np.load(f)]
    print("[54] stages=%s => labels %s" % (stages, [LAB[n] for n in stages]), flush=True)

    payload = dict(lags=LAGS)
    for idx, n in enumerate(stages):
        lab = LAB[n]
        path = os.path.join(TMP, "dm1c_repro_stage%d_f32.npy" % n)
        print("[54] stage %s (n=%d) idx=%d" % (lab, n, idx), flush=True)
        extract_stage(idx, path)
        n_use = 358400 if not args.n_traj else min(args.n_traj, 358400)
        m, lo, hi = run_stage(path, n_use, args.batches, args.workers, args.block, args.engine)
        payload["%s_S2b" % lab] = m
        payload["%s_S2_lo" % lab] = lo
        payload["%s_S2_hi" % lab] = hi
        print("  [%s] S2(lag1)=%.4e S2(lag10)=%.4e S2(lag1024)=%.4e | relative envelope width, median %.4f/%.4f"
              % (lab, m[0], m[9], m[-1], np.median((m - lo) / m), np.median((hi - m) / m)), flush=True)

    # ---- self-check: stage A must match the existing fig4a_batches_dm1c_repro.npz ----
    try:
        ref = np.load(os.path.join(RESULTS, "fig4a_batches_dm1c_repro.npz"))
        # ref["S2b"] is the **per-block** array (nb, nl), not the block mean, so average over axis 0 first
        for k, key in (("S2b", "A_S2b"), ("S2_lo", "A_S2_lo"), ("S2_hi", "A_S2_hi")):
            r = np.asarray(ref[k])
            r = r[:, :len(LAGS)].mean(axis=0) if r.ndim == 2 else r[:len(LAGS)]
            d = np.max(np.abs(payload[key] - r) / np.abs(r))
            print("[54] self-check %-6s against fig4a_batches_dm1c_repro (block mean): max relative difference %.3e %s"
                  % (k, d, "OK" if d < 1e-6 else "MISMATCH"), flush=True)
        ref2 = np.load(os.path.join(RESULTS, "structfuncs_dm1c_repro_full.npz"))
        d2 = np.max(np.abs(payload["A_S2b"] - ref2["S2"][:len(LAGS)]) / ref2["S2"][:len(LAGS)])
        print("[54] self-check A_S2b against structfuncs_dm1c_repro_full['S2']: max relative difference %.3e %s"
              % (d2, "OK" if d2 < 1e-6 else "MISMATCH"), flush=True)
    except FileNotFoundError:
        print("[54] self-check skipped: fig4a_batches_dm1c_repro.npz is missing", flush=True)

    np.savez(args.out, **payload)
    print("[54] wrote %s (%.2f MB)" % (args.out, os.path.getsize(args.out) / 1024 ** 2), flush=True)

    # ---- against the paper ----
    src = os.path.join(DATA, "Li_et_al_2024_source_data", "42256_2024_810_MOESM6_ESM.txt")
    lines = open(src, errors="replace").read().splitlines()
    print("\n[54] against Figure3d_*.out of the paper (mean column, aligned point by point):")
    for lab in ("A", "B", "C", "D"):
        name = "Figure3d_%s.out (tau,mean,-,+)" % lab
        i = [k for k, l in enumerate(lines) if l.strip() == "# " + name][0]
        n = 1024
        ref = np.array([[float(v) for v in lines[i + 1 + k].split()] for k in range(n)])
        y = payload["%s_S2b" % lab]
        r = y / ref[:, 1]
        print("  %s: ratio median %.4f (%.3f..%.3f) | S2(tau=1) ours %.4e paper %.4e | paper -/+ 0.%.4f/%.4f"
              % (lab, np.median(r), r.min(), r.max(), y[9], ref[9, 1],
                 np.median(ref[:, 2] / ref[:, 1]), np.median(ref[:, 3] / ref[:, 1])), flush=True)


if __name__ == "__main__":
    main()
