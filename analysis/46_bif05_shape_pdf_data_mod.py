"""46_bif05_shape_pdf_data_mod.py -- DATA stage, MODIFIED version (see the contract in TETRAD_DM.md section 5).

Changes vs the initial 46_bif05_shape_pdf_data.py (which stays frozen):
  (1) I3 binning: the initial version binned I3 on [0, 0.2] with 200 bins (width 0.001).  np.histogram DROPS the
      values outside the range -- there is no clipping in either script's DATA histograms (the np.clip calls
      apply only to the Gaussian MC reference).  So the initial version silently loses the I3 tail above 0.2
      (0.001-0.299% of the mass, largest at t = 2.05 tau_eta; the Gaussian reference loses 0.315%) and its
      delivered figure only draws I3 up to 0.16, far short of the support (max I3 = 0.297).  Here I3 is binned
      on [0, 0.36] with the SAME bin
      width as I1/I2 (0.005 -> 72 bins, divisible by 2/3/4/6 so the plot stage can coarsen exactly), which
      also covers the theoretical maximum I3 = 1/3.
  (2) Density magnitude depends on the bin width (peak density ~ 1/width), which matters now that the plot
      stage draws LINEAR y axes (as B2005 and VA08 do): the 0.001-wide I3 bins would give a needle-like peak
      of density ~600.  Equal bin widths across the three panels make the linear-y panels readable and the
      three panels mutually consistent.
Everything else (times, calibres, selection thresholds, Gaussian reference, stored per-tetrad arrays) is
identical to the initial script, so numbers stay comparable.

Writes results/bif05_shape_pdf_mod.npz (no figure).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tetrad_lib as tl

TIMES_TAU = [0.10, 0.49, 2.05, 5.00, 15.30, 35.00, 63.00, 100.00, 195.55]   # 195.55 = index 1999 (last frame)
BINW = 0.005                       # common bin width for all three panels
NB1 = int(round(1.0 / BINW))       # I1, I2: [0, 1]
NB3 = 72                           # I3: [0, 0.36] = 72 * 0.005 (covers 1/3, divisible by 2/3/4/6)
G1_LO, G1_HI = 5e2, 5e5            # B05 selection ranges, in units of eta^2
G2_LO, G2_HI = 5e1, 5e4
G3_LO, G3_HI = 5.0, 5e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--ref-samples", type=int, default=2000000)
    ap.add_argument("--tag", default="mod")
    a = ap.parse_args()
    tag = f"_{a.tag}" if a.tag else ""

    e2 = tl.ETA ** 2
    lim = np.array([[G1_LO * e2, G1_HI * e2], [G2_LO * e2, G2_HI * e2], [G3_LO * e2, G3_HI * e2]])
    # NB: a ragged list, NOT np.array(...) -- the three panels no longer share a bin count (I3 has 72
    # bins on [0, 0.36] vs 200 on [0, 1]), and np.array on ragged rows raises "inhomogeneous shape".
    edges = [np.linspace(0.0, 1.0, NB1 + 1), np.linspace(0.0, 1.0, NB1 + 1),
             np.arange(0.0, NB3 * BINW + 1e-9, BINW)]
    assert len(edges[2]) == NB3 + 1, len(edges[2])
    t_idx = [min(int(round(x * tl.TAU_ETA / tl.DTS)), tl.NT - 1) for x in TIMES_TAU]
    nt, nq = len(t_idx), tl.NQ
    Ii_all = np.zeros((nt, nq, 3))
    g_all = np.zeros((nt, nq, 3))

    t0 = time.time()
    done = 0
    for a0, x in tl.iter_positions(0, nq, chunk=a.chunk, source="pos"):
        n = x.shape[0]
        rho = tl.jacobi_rho(x[:, t_idx, :, :])           # only the requested times
        g, r2, Ii = tl.inertia(rho)
        Ii_all[:, a0:a0 + n, :] = np.transpose(Ii, (1, 0, 2))
        g_all[:, a0:a0 + n, :] = np.transpose(g, (1, 0, 2))
        done += n
        el = time.time() - t0
        print(f"  {done:6d}/{nq} quartets  el {el/60:4.1f}min  eta {el/done*(nq-done)/60:4.1f}min", flush=True)

    hist = [np.zeros((nt, len(edges[c]) - 1)) for c in range(3)]
    hist_sel = [np.zeros((nt, len(edges[c]) - 1)) for c in range(3)]
    stats = np.zeros((nt, 3, 4))                         # mean, median, p10, p90
    pass_frac = np.zeros(nt)
    for it in range(nt):
        g = g_all[it]
        m = ((g[:, 0] >= lim[0, 0]) & (g[:, 0] <= lim[0, 1])
             & (g[:, 1] >= lim[1, 0]) & (g[:, 1] <= lim[1, 1])
             & (g[:, 2] >= lim[2, 0]) & (g[:, 2] <= lim[2, 1]))
        pass_frac[it] = m.mean()
        for c in range(3):
            v = Ii_all[it, :, c]
            v = v[np.isfinite(v)]
            hist[c][it] = np.histogram(v, bins=edges[c], density=True)[0]
            if m.sum() > 50:
                hist_sel[c][it] = np.histogram(Ii_all[it, m, c], bins=edges[c], density=True)[0]
            stats[it, c] = [v.mean(), np.median(v), np.percentile(v, 10), np.percentile(v, 90)]
        print(f"  t = {TIMES_TAU[it]:7.2f} tau_eta (idx {t_idx[it]:4d}): <I> = "
              f"({stats[it,0,0]:.4f}, {stats[it,1,0]:.4f}, {stats[it,2,0]:.4f})  "
              f"pass {100*pass_frac[it]:5.1f}%  n_sel {int(m.sum())}")

    print(f"\nMonte-Carlo isotropic-Gaussian reference ({a.ref_samples} samples), chunked...")
    rng = np.random.default_rng(0)
    ref = [np.zeros(len(edges[c]) - 1) for c in range(3)]
    n_chunk, per_c = 200000, np.zeros(3)
    ref_i3 = []                                        # the reference's own I3 extent (tail check)
    for c in range(3):
        done_n = 0
        while done_n < a.ref_samples:
            n = min(n_chunk, a.ref_samples - done_n)
            X = rng.normal(size=(n, tl.NQUARTET, 3))
            _, _, Iir = tl.inertia(tl.jacobi_rho(X[:, None, :, :])[:, 0])
            v = np.clip(Iir[:, c], 0.0, edges[c][-1] * (1 - 1e-12))
            ref[c] += np.histogram(v, bins=edges[c])[0]
            per_c[c] += v.sum()
            if c == 2:
                ref_i3.append(v.copy())
            done_n += n
            del X, Iir
        ref[c] = ref[c] / (done_n * np.diff(edges[c]))
    ref_mean = np.array([per_c[c] / a.ref_samples for c in range(3)])
    print(f"  reference <I_i> = {np.array2string(ref_mean, precision=4)}   (literature 0.748/0.222/0.030)")
    ri3 = np.concatenate(ref_i3)
    print(f"  reference I3 extent: max {ri3.max():.4f}, p99.9 {np.percentile(ri3, 99.9):.4f}, "
          f"above 0.2 {100*np.mean(ri3 > 0.2):.3f}%, above 0.3 {100*np.mean(ri3 > 0.3):.4f}%"
          f"   (=> the x range that actually carries the reference curve)")
    print("  data I3 max per plotted time:",
          np.array2string(Ii_all.max(axis=1)[:, 2], precision=3))

    print("\n  cross-check of <I_i> against the B03 npz (same tetrads, same times):")
    b3 = os.path.join(tl.RESULTS, "bif05_shape_indices.npz")
    if os.path.exists(b3):
        d3 = np.load(b3)
        for it in range(nt):
            if t_idx[it] >= 1:
                print(f"    t = {TIMES_TAU[it]:7.2f}: here {np.array2string(stats[it,:,0], precision=4)}"
                      f"   B03 {np.array2string(d3['mean'][t_idx[it]], precision=4)}")

    os.makedirs(tl.RESULTS, exist_ok=True)
    out = os.path.join(tl.RESULTS, f"bif05_shape_pdf{tag}.npz")
    np.savez_compressed(out, times_tau=np.array(TIMES_TAU), t_idx=np.array(t_idx),
                        edges1=edges[0], edges2=edges[1], edges3=edges[2],
                        hist1=hist[0], hist2=hist[1], hist3=hist[2],
                        hist_sel1=hist_sel[0], hist_sel2=hist_sel[1], hist_sel3=hist_sel[2],
                        hist_ref1=ref[0], hist_ref2=ref[1], hist_ref3=ref[2],
                        ref_mean=ref_mean, stats=stats, pass_frac=pass_frac, lim=lim, eta=tl.ETA,
                        nq=nq, binw=BINW, Ii=Ii_all, g=g_all)
    print("\ndata ->", out)
    print("figure: run 46_bif05_shape_pdf_plot_mod.py (linear y, style and times are free)")


if __name__ == "__main__":
    main()
