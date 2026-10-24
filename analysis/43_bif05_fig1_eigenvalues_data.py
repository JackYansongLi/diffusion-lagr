"""43_bif05_fig1_eigenvalues_data.py -- DATA stage for Biferale et al. (2005) Fig. 1 (main panel).

Computes the mean eigenvalues <g_i(t)> of the inertia tensor I = rho rho^T over the tetrad dataset and
writes EVERY raw curve the figure (or a later window/style revision) may need:

    mean, lo, hi, std        per-time curves; lo/hi/std = spread across the 1024 spatial blocks
    blk_mean, blk_labels, blk_cnt   the per-pencil-column curves themselves (full raw data)
    frac_gt_L4, frac_gt_L2   fraction of quartets with r > L/4 and > L/2 (box-limit diagnostic)
    r_max, closed_form, eta2, t_phys, t_tau, source, nq, meta

NO plotting in this file -- see 43_bif05_fig1_eigenvalues_plot.py.  Splitting the two stages means a
style change costs seconds instead of re-running this 5-minute sweep (user request 2026-10-08).

Usage: 43_bif05_fig1_eigenvalues_data.py [--nq N] [--chunk C] [--tag NAME] [--source dns|integrated]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tetrad_lib as tl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq", type=int, default=tl.NQ, help="number of quartets (balanced sampling)")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--tag", default="")
    ap.add_argument("--source", choices=["dns", "integrated"], default="dns")
    a = ap.parse_args()
    tag = f"_{a.tag}" if a.tag else ""

    nblk = 1024
    sum_g = np.zeros((nblk, tl.NT, 3))
    sum_g2 = np.zeros((nblk, tl.NT, 3))
    cnt = np.zeros(nblk)
    n_gt = np.zeros((tl.NT, 2))
    r_max = np.zeros(tl.NT)
    nq_seen = 0
    nq_done = 0
    t0 = time.time()
    idx = tl.balanced(a.nq)
    print(f"  sample: {idx.size} quartets, {len(np.unique(tl.pencil_column(4 * idx)))} distinct pencil columns")
    for sl, rho, W in tl.iter_rhoW(idx, chunk=a.chunk, source=a.source):
        g = tl.inertia(rho)[0]                                   # (n, T, 3) descending
        r = np.sqrt((rho ** 2).sum(axis=(2, 3)))                 # (n, T) = sqrt(tr I) per quartet
        n_gt[:, 0] += (r > tl.LBOX / 4).sum(0)
        n_gt[:, 1] += (r > tl.LBOX / 2).sum(0)
        r_max = np.maximum(r_max, r.max(0))
        nq_seen += r.shape[0]
        blk = tl.pencil_column(4 * sl)
        for i in range(g.shape[0]):
            sum_g[blk[i]] += g[i]
            sum_g2[blk[i]] += g[i] ** 2
            cnt[blk[i]] += 1
        nq_done += g.shape[0]
        el = time.time() - t0
        print(f"  {nq_done:6d}/{a.nq} quartets  el {el/60:4.1f}min  eta {el/nq_done*(a.nq-nq_done)/60:4.1f}min", flush=True)

    used = cnt > 0
    blk_mean = sum_g[used] / cnt[used][:, None, None]
    blk_var = sum_g2[used] / cnt[used][:, None, None] - blk_mean ** 2
    mean = blk_mean.mean(0)
    lo, hi = blk_mean.min(0), blk_mean.max(0)
    std = np.sqrt(np.maximum(blk_var, 0).mean(0))
    frac4 = n_gt[:, 0] / max(nq_seen, 1)
    frac2 = n_gt[:, 1] / max(nq_seen, 1)
    t_phys = np.arange(tl.NT) * tl.DTS
    t_tau = t_phys / tl.TAU_ETA

    h = tl.DX / 2
    cf = np.array([h ** 2, h ** 2, h ** 2 / 4])
    eta2 = tl.ETA ** 2
    print(f"\neta = {tl.ETA} (TURB-Lagr), tau_eta = {tl.TAU_ETA}, dx = {tl.DX:.7f}")
    print("initial values (t=0) in the paper's normalisation <g_i>/eta^2:")
    for i in range(3):
        print(f"   g{i+1}(0)/eta^2: measured {mean[0, i]/eta2:.4f}   closed form {cf[i]/eta2:.4f}   "
              f"ratio {mean[0, i]/cf[i]:.3f}")
    print(f"   (Biferale 2005 regular tetrahedron, g_i(0)=dx^2/2, would start at {tl.DX**2/2/eta2:.4f})")
    print("\nlocal log-slope d ln<g_i>/d ln t (exhaustive windows):")
    for (w0, w1) in [(50, 200), (200, 600), (600, 1200), (1200, 1999)]:
        sl = (np.log(mean[w1]) - np.log(mean[w0])) / (np.log(t_tau[w1]) - np.log(t_tau[w0]))
        print(f"   t={w0:4d}-{w1:4d} (tau_eta {t_tau[w0]:6.1f}-{t_tau[w1]:6.1f}): slopes = {np.round(sl, 2)}")
    r_box = np.sqrt(mean.sum(1))
    w = np.where(r_box <= tl.LBOX / 4)[0]
    print(f"\nbox-limit windows (L = {tl.LBOX:.4f}, L/4 = {tl.LBOX/4:.4f}) -- printed/stored, NOT drawn:")
    print(f"   <r^2>^(1/2) <= L/4 up to t index {w[-1]} (tau_eta {t_tau[w[-1]]:.1f})")
    print("   fraction of quartets with r > L/4 :", np.round(frac4[[0, 200, 400, 630, 1000, 1999]], 4))
    print("   fraction of quartets with r > L/2 :", np.round(frac2[[0, 200, 400, 630, 1000, 1999]], 4))

    os.makedirs(tl.RESULTS, exist_ok=True)
    out = os.path.join(tl.RESULTS, f"bif05_fig1_eigenvalues{tag}.npz")
    np.savez_compressed(out, t_phys=t_phys, t_tau=t_tau, mean=mean, lo=lo, hi=hi, std=std,
                        source=a.source, blk_mean=blk_mean, blk_labels=np.where(used)[0], blk_cnt=cnt[used],
                        closed_form=cf, nq=nq_done, frac_gt_L4=frac4, frac_gt_L2=frac2, r_max=r_max,
                        eta2=eta2, meta=np.array([tl.ETA, tl.TAU_ETA, tl.DTS, tl.DX, tl.LBOX]))
    print("data ->", out)
    print("figure: run 43_bif05_fig1_eigenvalues_plot.py (style changes are free)")


if __name__ == "__main__":
    main()
