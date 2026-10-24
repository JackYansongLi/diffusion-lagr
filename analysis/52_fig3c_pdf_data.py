#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""52_fig3c_pdf_data.py -- increment-PDF data for Fig. 3c: the DNS and the four denoising stages
A/B/C/D.

Conventions (fixed 2026-10-15; the single authoritative section is RESULTS_ANALYSIS.md 7.13)
-----------------------------------------------------------------
- quantity: dtauVi = Vi(t+tau) - Vi(t), the velocity-component increment; the caption of the paper
           reads "PDF of dtauVi for tau = tau_eta (c)"
           => this project uses 0.1 tau_eta per lag (verified by Table 1 and the reproductions of
           Figs. 1a and 4a), so **lag = 10**
- sigma: the standard deviation over **all increments of that curve**, taken globally, not per
         trajectory
- axes: y = PDF(dtauVi)*sigma and x = dtauVi/sigma
           the five source curves of the paper give int y dx = 1.00000 and int x^2 y dx =
           1.0006..1.0088, confirming that sigma is the standard deviation of the increments
- resolution: both versions are stored, but **the aggregated one must be used for the comparison**
  (the test in section 7.13):
          (a) fine: a 0.05 sigma histogram, used to draw a smooth curve;
          (b) paper: aggregated onto the x grid of the paper (~0.325 sigma spacing), as 08_mod did.
          the fine-bin peak is 0.7758 against 0.7490 in the paper (a spurious +3.6%), while the
          aggregated version gives 0.7486 against 0.7490 (0.05%).
- sigma estimate: a single pass; the histogram uses the sigma estimated from the first 10% of the
  rows while the exact sigma is accumulated over the whole set as a check
  (sigma only normalises, and a 10% subsample is accurate to ~1e-4, invisible visually and
  statistically)

Data sources
------
- DNS: the train block of Data/Lagr_u1c_diffusion.h5 (983040 x 2000 x 1 float64, all of it)
- stages A/B/C/D: arr[stages.index(n)] of
  Data/Reproduce/samples_358400x2000x1_DM-1c_reproduce.npz
  (arr is (4, 358400, 2000, 1) float32; n = 0/48/216/416 correspond to A/B/C/D)

Output: results/fig3c_pdf_dm1c_repro.npz
     one group of keys per curve: <key>_hist / <key>_edges / <key>_sigma_used / <key>_sigma_exact /
                     <key>_n_incr / <key>_n_traj / <key>_x_paper / <key>_y_paper
     with key in {dns, A, B, C, D}, plus the metadata stages / lag / fine_bin.
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np

LAG = 10                      # τ = τη（0.1 τη/lag）
FINE_BIN = 0.05               # fine-bin width in units of sigma
X_MAX = 40.0                  # histogram range (the figures crop to +-30)
SIGMA_ROWS_FRAC = 0.10        # estimate sigma from the first 10% of the rows

ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data")
RESULTS = os.path.join(ROOT, "results")
NPZ_REPRO = os.path.join(DATA, "Reproduce", "samples_358400x2000x1_DM-1c_reproduce.npz")
H5_DNS = os.path.join(DATA, "Lagr_u1c_diffusion.h5")
OUT = os.path.join(RESULTS, "fig3c_pdf_dm1c_repro.npz")


def npy_header(fid):
    """Read (shape, dtype, header_len, fortran) from an open .npy stream."""
    assert fid.read(6) == b"\x93NUMPY", "not a .npy stream"
    ver = struct.unpack("<2B", fid.read(2))
    if ver[0] == 1:
        hlen = struct.unpack("<H", fid.read(2))[0]
    else:
        hlen = struct.unpack("<I", fid.read(4))[0]
    d = eval(fid.read(hlen), {"False": False, "True": True})   # only parses headers we wrote ourselves
    return d["shape"], np.dtype(d["descr"]), 10 + hlen if ver[0] == 1 else 12 + hlen, d["fortran_order"]


def iter_npz_member_blocks(path, member, block_rows):
    """Stream blocks of rows from an uncompressed member of a .npz as (start, arr[rows, L, C])."""
    import zipfile
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(member)
        assert info.compress_type == zipfile.ZIP_STORED, "the member is compressed; use sequential decompression instead"
        with zf.open(member) as f:
            shape, dt, hlen, fortran = npy_header(f)
            assert not fortran, "row blocking does not support Fortran-ordered arrays"
            L, C = int(shape[1]), int(shape[2])
            row_bytes = L * C * dt.itemsize
            n = int(shape[0])
            for s in range(0, n, block_rows):
                k = min(block_rows, n - s)
                buf = f.read(k * row_bytes)
                if len(buf) != k * row_bytes:
                    raise IOError("member %s ends early at row %d" % (member, s))
                yield s, np.frombuffer(buf, dtype=dt).reshape(k, L, C)


def iter_h5_blocks(path, key, block_rows):
    import h5py
    with h5py.File(path, "r") as f:
        ds = f[key]
        L, C = int(ds.shape[1]), int(ds.shape[2])
        n = int(ds.shape[0])
        for s in range(0, n, block_rows):
            k = min(block_rows, n - s)
            yield s, ds[s:s + k]


def process(name, blocks, n_traj_expected, block_rows, fine_edges, n_bins):
    """One pass: fix sigma from the first SIGMA_ROWS_FRAC of the blocks, then accumulate the histogram
and the exact sum d and sum d^2."""
    t0 = time.time()
    hist = np.zeros(n_bins, dtype=np.int64)
    sum_d = 0.0
    sum_d2 = 0.0
    n_incr = 0
    rows_done = 0
    sigma_used = None
    n_sigma_rows = max(block_rows, int(n_traj_expected * SIGMA_ROWS_FRAC))
    for s, arr in blocks:
        v = np.asarray(arr[:, :, 0], dtype=np.float64)
        d = (v[:, LAG:] - v[:, :-LAG]).ravel()
        sum_d += d.sum()
        sum_d2 += float(np.dot(d, d))
        n_incr += d.size
        rows_done += arr.shape[0]
        if sigma_used is None:
            if rows_done >= n_sigma_rows:
                m = sum_d / n_incr
                sigma_used = float(np.sqrt(sum_d2 / n_incr - m * m))
                print("    [%s] sigma estimated from the first %d trajectories (%.1f%% of the data): %.6f"
                      % (name, rows_done, 100.0 * rows_done / n_traj_expected, sigma_used), flush=True)
        if sigma_used:
            hist += np.histogram(d / sigma_used, bins=fine_edges)[0]
        del v, d
        if (s // block_rows) % 20 == 0:
            print("    [%s] row %7d/%d  increments %.4g  elapsed %.0fs"
                  % (name, rows_done, n_traj_expected, n_incr, time.time() - t0), flush=True)
    m = sum_d / n_incr
    sigma_exact = float(np.sqrt(sum_d2 / n_incr - m * m))
    if sigma_used is None:                      # the data are smaller than expected: fall back
        sigma_used = sigma_exact
    print("  [%s] done: %d trajectories, %.4g increments, exact sigma %.6f, sigma used %.6f (ratio %.5f), %.0fs"
          % (name, n_traj_expected, n_incr, sigma_exact, sigma_used, sigma_used / sigma_exact,
             time.time() - t0), flush=True)
    return dict(hist=hist, sigma_used=sigma_used, sigma_exact=sigma_exact,
                n_incr=n_incr, n_traj=n_traj_expected)


def aggregate_to_paper(x_paper, edges, hist, half_width):
    """Aggregate the fine histogram onto the x grid of the paper (every paper point covers [x-hw, x+hw]
    and takes the mean).

    The abscissa is x = delta/sigma, so the PDF_x computed from it already is PDF(delta)*sigma (the
    sigma sits inside the coordinate change),
    and it must **not be multiplied by sigma again**. A smoke run did so once more and shrank the
    whole curve by a factor of sigma (~3%, which reads as a flattened curve).
    """
    ctr = 0.5 * (edges[1:] + edges[:-1])
    pdf = hist / (hist.sum() * np.diff(edges)[0])          # = PDF(δ)·σ
    y = np.empty(len(x_paper))
    for i, xc in enumerate(x_paper):
        m = (ctr >= xc - half_width) & (ctr < xc + half_width)
        y[i] = pdf[m].mean() if m.any() else np.nan
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-rows", type=int, default=8000, help="rows per block for the DNS (float64, memory grows with the row count)")
    ap.add_argument("--npz-block-rows", type=int, default=20000, help="rows per block for the stages (float32)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit-traj", type=int, default=0, help="process only the first N trajectories (smoke test; 0 = all)")
    args = ap.parse_args()

    fine_edges = np.arange(-X_MAX, X_MAX + FINE_BIN / 2, FINE_BIN)
    n_bins = len(fine_edges) - 1
    payload = dict(stages=None, lag=LAG, fine_bin=FINE_BIN, x_max=X_MAX)

    # the x grid of the paper, the reference for the comparison
    src = os.path.join(DATA, "Li_et_al_2024_source_data", "42256_2024_810_MOESM6_ESM.txt")
    lines = open(src, errors="replace").read().splitlines()
    refs = {}
    for blk in ("Figure3c_DNS", "Figure3c_A", "Figure3c_B", "Figure3c_C", "Figure3c_D"):
        i = [k for k, l in enumerate(lines) if l.strip() == "# " + blk + ".out"][0]
        n = 0
        while i + 1 + n < len(lines) and lines[i + 1 + n].strip() and not lines[i + 1 + n].startswith("#"):
            n += 1
        refs[blk.split("_")[-1]] = np.array([[float(v) for v in lines[i + 1 + k].split()] for k in range(n)])
    grids = {k: v[:, 0] for k, v in refs.items()}          # the x grid of each of the five curves (their lengths differ)
    halves = {k: float(np.median(np.diff(v[:, 0]))) / 2.0 for k, v in refs.items()}

    # ---- DNS ----
    print("[52] DNS：%s" % H5_DNS, flush=True)
    n_dns = 983040 if not args.limit_traj else min(args.limit_traj, 983040)
    def dns_blocks(br):
        for s, a in iter_h5_blocks(H5_DNS, "train", br):
            if args.limit_traj and s >= args.limit_traj:
                break
            yield s, a[:args.limit_traj - s] if args.limit_traj else a
    r = process("dns", dns_blocks(args.block_rows), n_dns, args.block_rows, fine_edges, n_bins)
    payload.update({("dns_" + k): v for k, v in r.items()})
    payload["dns_x_paper"] = grids["DNS"]
    payload["dns_y_paper"] = aggregate_to_paper(grids["DNS"], fine_edges, r["hist"], halves["DNS"])

    # ---- stages A/B/C/D ----
    import zipfile
    with zipfile.ZipFile(NPZ_REPRO) as zf:
        with zf.open("stages.npy") as f:
            stages = np.load(f)
    stages = [int(v) for v in stages]
    print("[52] stages: %s (A=n0, B=n48, C=n216, D=n416)" % stages, flush=True)
    lab = {0: "A", 48: "B", 216: "C", 416: "D"}
    payload["stages"] = np.array(stages)
    for n in stages:
        name = lab[n]
        print("[52] stage %s: n=%d" % (name, n), flush=True)
        # the index of the stage in arr is stages.index(n); it is the idx-th block of the member
        idx = stages.index(n)
        rows_per_stage = 358400 if not args.limit_traj else min(args.limit_traj, 358400)
        # stream the member and slice by stage (the member is (4, n, L, 1) in C order, so the stages read
        # sequentially)
        limit = args.limit_traj
        def stage_blocks(idx=idx, br=args.npz_block_rows, limit=limit):
            import zipfile as _z
            with _z.ZipFile(NPZ_REPRO) as zf:
                info = zf.getinfo("arr.npy")
                assert info.compress_type == _z.ZIP_STORED
                with zf.open("arr.npy") as f:
                    shape, dt, hlen, fortran = npy_header(f)
                    S, N, L, C = (int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]))
                    row_bytes = L * C * dt.itemsize
                    f.seek(hlen + idx * N * row_bytes)          # seek to the start of the stage
                    for s in range(0, N, br):
                        k = min(br, N - s)
                        buf = f.read(k * row_bytes)
                        if len(buf) != k * row_bytes:
                            raise IOError("stage %d of arr.npy ends early at row %d" % (idx, s))
                        yield s, np.frombuffer(buf, dtype=dt).reshape(k, L, C)
                        if limit and s + k >= limit:
                            break
        r = process(name, stage_blocks(idx), rows_per_stage, args.npz_block_rows, fine_edges, n_bins)
        payload.update({(name + "_" + k): v for k, v in r.items()})
        payload[name + "_x_paper"] = grids[name]
        payload[name + "_y_paper"] = aggregate_to_paper(grids[name], fine_edges, r["hist"], halves[name])

    np.savez(args.out, **payload)
    print("[52] wrote %s (%.2f MB)" % (args.out, os.path.getsize(args.out) / 1024 ** 2), flush=True)

    # ---- reading against the curves of the paper (aggregated, the convention of section 7.13) ----
    print("\n[52] against the source curves of the paper (aggregated onto its grid):")
    for key in ("dns", "A", "B", "C", "D"):
        yr = refs["DNS" if key == "dns" else key][:, 1]
        y = payload[key + "_y_paper"]
        x_paper = payload[key + "_x_paper"]
        dx = halves["DNS" if key == "dns" else key] * 2
        L1 = float(np.nansum(np.abs(y - yr)) * dx)
        mass = lambda a, b: float(np.nansum(y[(np.abs(x_paper) >= a) & (np.abs(x_paper) < b)]) * dx)
        mr = lambda a, b: float(np.nansum(yr[(np.abs(x_paper) >= a) & (np.abs(x_paper) < b)]) * dx)
        print("  %-4s peak %.4f vs paper %.4f | L1 %.5f | mass 0-1 sigma %.5f/%.5f  1-2 sigma %.5f/%.5f  2-5 sigma %.5f/%.5f"
              % (key, np.nanmax(y), yr.max(), L1, mass(0, 1), mr(0, 1), mass(1, 2), mr(1, 2),
                 mass(2, 5), mr(2, 5)), flush=True)


if __name__ == "__main__":
    main()
