#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""58_fig3f_traj_data.py -- data for Fig. 3f: one trajectory followed through the four denoising
stages.

Conventions (see RESULTS_ANALYSIS.md section 7.16)
-----------------------------------
- one panel of Fig. 3f in the paper holds 2000 points with x = 0.1*k, so t/tau_eta in [0, 200] (the
  whole record); the four panels show **the same trajectory** at
  four stages (A = the final n = 0, D = the noisiest, n = 0.52N). The source data of the paper show it:
  corr(A,B) = 0.9999,
  corr(A,C) = 0.9958 and corr(A,D) = 0.891, i.e. one trajectory at different stages of denoising.
- the vertical axis is V_i^(n)/sigma(V_i^(n)) with **sigma the standard deviation over every trajectory of
  that stage** (the four panels of the paper have y.std = 1.17..1.22,
  not the standard deviation of the single trajectory, which would be exactly 1).
- our quantity: the samples of the four stages (min-max normalised with the global u1c scalars), mapped
  back to physical units and divided by the
  full-sample sigma of that stage; the complete 2000-point record of trajectory i is then written out.
- this script produces data only; the figure comes from 59_fig3f_traj_plot_mod.py, which never rescans
the large HDF5/npz files.
"""
import argparse
import os
import time

import numpy as np

ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data")
RESULTS = os.path.join(ROOT, "results")

# the stage index follows stages=[416, 216, 48, 0] of samples_*.npz: 0 = D, the noisiest, -> 3 = A, final
STAGES = ((416, "D", 0), (216, "C", 1), (48, "B", 2), (0, "A", 3))
U1C_MIN, U1C_MAX = -11.24481512, 10.31416093   # the same constants as in 54 and 56


def stage_path(n):
    return os.path.join(DATA, "tmp", "dm1c_repro_stage%d_f32.npy" % n)


def scan_stats(path, block=4096):
    """Accumulate sum and sumsq over every sample in blocks, giving the mean and standard deviation of the
    stage in the normalised domain (accumulated in float64)."""
    a = np.load(path, mmap_mode="r")
    n = a.shape[0]
    s1 = 0.0
    s2 = 0.0
    cnt = 0
    t0 = time.time()
    for i0 in range(0, n, block):
        x = np.asarray(a[i0:i0 + block], dtype=np.float64).ravel()
        s1 += x.sum()
        s2 += (x * x).sum()
        cnt += x.size
        if (i0 // block) % 8 == 0:
            print("      %d/%d (%.0fs)" % (i0, n, time.time() - t0), flush=True)
    mu = s1 / cnt
    var = s2 / cnt - mu * mu
    return mu, float(np.sqrt(max(var, 0.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traj", type=int, default=16, help="export the first N trajectories (the plotting script picks one)")
    ap.add_argument("--out", default=os.path.join(RESULTS, "fig3f_traj_dm1c_repro.npz"))
    args = ap.parse_args()

    a_f = (U1C_MAX - U1C_MIN) / 2.0
    b_f = (U1C_MAX + U1C_MIN) / 2.0
    labels, sigmas, mus, series = [], [], [], []
    for n, tag, idx in STAGES:
        p = stage_path(n)
        print("[58] stage %s (n=%d), full-sample statistics: %s" % (tag, n, p), flush=True)
        mu_s, sd_s = scan_stats(p)
        sigma = a_f * sd_s          # standard deviation in physical units (the map is linear, so sigma scales with the factor)
        mu_V = a_f * mu_s + b_f
        print("      normalised domain mean=%.6e std=%.6e => physical mean=%.6f sigma=%.6f" % (mu_s, sd_s, mu_V, sigma), flush=True)
        a = np.load(p, mmap_mode="r")
        raw = np.asarray(a[:args.n_traj], dtype=np.float32)[:, :, 0]      # (N, 2000) in the normalised domain
        # back to physical units: s in [-1, 1] => V = (s+1)*(mx-mn)/2 + mn (the same form as in 54)
        # do not write (s+1)*a + (mx+mn)/2: that adds half a range too much (b already carries the +mn half)
        V = (raw.astype(np.float64) + 1.0) * a_f + U1C_MIN                # physical units
        series.append(V / sigma)
        labels.append(tag)
        sigmas.append(sigma)
        mus.append(mu_V)

    V = np.stack(series)                       # (4, N, 2000), ordered D, C, B, A
    np.savez_compressed(args.out, V=V.astype(np.float32), stages=np.array([s[0] for s in STAGES]),
                        labels=np.array(labels), sigma=np.array(sigmas), mean_V=np.array(mus),
                        u1c_min=U1C_MIN, u1c_max=U1C_MAX, n_traj=args.n_traj,
                        t=np.arange(2000) * 0.1)
    print("[58] → %s  (V shape %s)" % (args.out, V.shape))
    for i, tag in enumerate(labels):
        s = V[i]
        print("      %s: V/sigma over all trajectories, mean %+.4f std %.4f, range [%.2f, %.2f] | trajectory 0 mean %+.4f std %.4f"
              % (tag, s.mean(), s.std(), s.min(), s.max(), s[0].mean(), s[0].std()))
        d = np.diff(s[0])
        print("         trajectory 0, lag-1 increment rms = %.4f" % np.sqrt((d * d).mean()))
    print("[58] note: pick a trajectory with 59_fig3f_traj_plot_mod.py --traj <i> (0 <= i < %d)" % args.n_traj)


if __name__ == "__main__":
    main()
