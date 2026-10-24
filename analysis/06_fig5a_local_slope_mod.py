#!/usr/bin/env python3
"""06_fig5a_local_slope_mod.py — Fig. 5a reproduction (**revision**: one comparison panel per dataset)

Differences from the initial `06_fig5a_local_slope.py`:
  the initial version draws **one** figure with (a) the three datasets and (b) the Fig. 5b literature compilation;
  this revision draws **three** comparison panels, one per dataset, and:
    - **indexes** the paper's 31 tau points directly (lag = 10*tau is always an integer, so no interpolation is needed, unlike the linear interpolation of the initial version);
    - error bars follow the **convention of the paper's caption**: the statistics/error-bars note of Fig. 4 gives **30 batches** for the DNS and DM-3c (10 trajectory blocks x 3 components) and **10 batches** for DM-1c;
      one zeta(4,tau) curve is computed per batch and component and its min/max envelope taken (= Sbc[:, :, p=2/p=4, :] of `flatness_<ds>_full.npz`);
    - the axes are fixed to 1.0..2.2 and 3e-2..2.5e2 (the range of the paper's panel) and a check reports whether any element is clipped;
    - **mean, lower and upper bound share one source**: the same 30/10 batch curves (mean = batch average, error = batch min/max),
      which is the convention of the paper's (mean, -, +) columns (the paper's mean differs from a fully pooled estimate by at most 0.5%; both agree here to the 0.1% level);
    - the paper only shows the batch convention, so the full-resolution curve is **not** overlaid by default (pass `--full-curve` for the smooth 1024-point curve).

Method as in the paper: zeta(p,tau) = [dlogS_p/dlogtau] / [dlogS_2/dlogtau], on the integer lag grid 1..1024 with a second-order central difference
(np.gradient, in log10(lag)).

Input: results/flatness_{dns,dm1c,dm3c}_full.npz (mean + Sbc)
       paper source data MOESM8: Figure5a_{DNS,DM-1c,DM-3c}.out (tau, mean, -, +), 31 rows each
Output: results/figs/A6_fig5a_local_slope_mod_{dns,dm1c,dm3c}.png
Usage: python3 06_fig5a_local_slope_mod.py [--datasets dns,dm1c,dm3c]
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Root of the analysis tree: the directory that holds results/ and Data/.
# Override with the LTS_ROOT environment variable; the default is the parent
# of this analysis/ directory.
ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SRC = os.path.join(ROOT, "Data", "Li_et_al_2024_source_data")
FIGS = os.path.join(RESULTS, "figs")
SRC_FN = os.path.join(SRC, "42256_2024_810_MOESM8_ESM.txt")
TAG = {"dns": "DNS", "dm1c": "DM-1c", "dm3c": "DM-3c"}
BLOCK = {"dns": "Figure5a_DNS.out (tau,mean,-,+)", "dm1c": "Figure5a_DM-1c.out (tau,mean,-,+)",
         "dm3c": "Figure5a_DM-3c.out (tau,mean,-,+)"}
YLIM = (1.0, 2.2)
XLIM = (3e-2, 2.5e2)
EVERY = 1          # put a marker on all 31 tau points


def paper_block(ds):
    lines = open(SRC_FN, errors="replace").read().split("\n")
    i = [k for k, l in enumerate(lines) if l.startswith("# " + BLOCK[ds])][0]
    n = 0
    while i + 1 + n < len(lines) and lines[i + 1 + n].strip() and not lines[i + 1 + n].startswith("#"):
        n += 1
    pa = np.array([[float(x) for x in lines[i + 1 + j].split()] for j in range(n)])
    return pa[:, 0], pa[:, 1], pa[:, 1] - pa[:, 2], pa[:, 1] + pa[:, 3]     # tau, mean, lo, hi


def zeta4(S2, S4, lags):
    """ζ(4,τ) = [dlogS4/dlogτ]/[dlogS2/dlogτ]，second-order central difference in log10. S2/S4 are (..., nl)."""
    lg = np.log10(np.asarray(lags, dtype=float))
    d4 = np.gradient(np.log10(S4), lg, axis=-1)
    d2 = np.gradient(np.log10(S2), lg, axis=-1)
    return d4 / d2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dns,dm1c,dm3c")
    ap.add_argument("--full-curve", action="store_true",
                    help="also overlay the pooled full-resolution curve (1024 tau points); the paper only has the batch convention, so this is off by default")
    args = ap.parse_args()
    for ds in args.datasets.split(","):
        z = np.load(os.path.join(RESULTS, "flatness_%s_full.npz" % ds))
        orders = [int(x) for x in z["orders"]]
        i2, i4 = orders.index(2), orders.index(4)
        lags = np.asarray(z["lags"])
        mean = z["mean"]                                    # (nl, 4)
        Sbc = z["Sbc"]                                      # (nb, nl, 4, C)
        nb, nl, _, nc = Sbc.shape

        # our mean curve (full resolution, 1024 points)
        zeta_mean = zeta4(mean[:, i2], mean[:, i4], lags)
        # per batch and component -> envelope (paper caption convention: 10 blocks x 3 components = 30; DM-1c has C = 1, hence 10)
        zb = np.empty((nb, nc, nl))
        for b in range(nb):
            for c in range(nc):
                zb[b, c] = zeta4(Sbc[b, :, i2, c], Sbc[b, :, i4, c], lags)
        zb = zb.reshape(-1, nl)
        lo_b, hi_b = zb.min(axis=0), zb.max(axis=0)

        tau, pm, plo, phi = paper_block(ds)
        L = np.rint(tau * 10).astype(int)                   # lag = 10*tau (always an integer, so it is indexed directly)
        ours = zb.mean(axis=0)[L - 1]        # same source as the paper: mean = average of the batch curves (pooled estimate in zeta_mean, at most 0.5% apart)
        olo = lo_b[L - 1]
        ohi = hi_b[L - 1]
        ratio = ours / pm

        # ---- clipping check ----
        n_clip = int(((ours < YLIM[0]) | (ours > YLIM[1])).sum()
                     + ((olo < YLIM[0]) | (ohi > YLIM[1])).sum()
                     + ((plo < YLIM[0]) | (phi > YLIM[1])).sum())
        print("[%s] batch curves %d (%d blocks x %d components) | median ratio over the 31 points %.4f (%.3f..%.3f) | median absolute difference %.4f, maximum %.4f @tau=%.1f"
              % (TAG[ds], zb.shape[0], nb, nc, np.median(ratio), ratio.min(), ratio.max(),
                 np.median(np.abs(ours - pm)), np.abs(ours - pm).max(), tau[np.argmax(np.abs(ours - pm))]), flush=True)
        print("    y axis %.1f..%.1f | our bar span %.3f..%.3f | paper band span %.3f..%.3f => %d element(s) outside the view%s"
              % (YLIM[0], YLIM[1], olo.min(), ohi.max(), plo.min(), phi.max(), n_clip,
                 " (as in the paper's panel, whose bars also leave that range)" if n_clip else ""), flush=True)

        # ---- figure (lin-log) ----
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        h_ref = ax.axhline(2.0, color="0.45", ls=":", lw=1.2, zorder=1)
        # paper: **dashed line + band** (the style of the Figs. 4a/4b/4c comparisons), merged into one legend entry with the curve
        ln_p = ax.plot(tau, pm, "s--", ms=4.0, mfc="white", mew=0.9, color="0.25", lw=1.1, zorder=3)[0]
        bd_p = ax.fill_between(tau, plo, phi, color="0.35", alpha=0.22, zorder=2)
        # this work: solid line + filled circle + error bars
        eb = ax.errorbar(tau, ours, yerr=[ours - olo, ohi - ours], fmt="o-", ms=3.4, lw=1.5,
                         color="#1f77b4", capsize=2, elinewidth=0.9, zorder=4)
        handles = [(ln_p, bd_p), eb, h_ref]
        labels = ["paper (MOESM8): dashed line + min/max band",
                  "this work: solid line + min/max error bars (%d batches)" % zb.shape[0],
                  r"$\zeta=2$ (non-intermittent)"]
        if args.full_curve:           # optional: pooled full-resolution estimate (the paper has no such curve)
            k_full = lags <= 600
            handles.append(ax.plot(lags[k_full] * 0.1, zeta_mean[k_full], "-", color="#1f77b4",
                                   lw=1.0, alpha=0.5, zorder=2)[0])
            labels.append("this work (pooled, all 1024 lags; not in paper)")
        ax.set_xscale("log")
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_xlabel(r"$\tau/\tau_\eta$", fontsize=11)
        ax.set_ylabel(r"$\zeta(4,\tau)$", fontsize=11)
        ax.set_title("Fig.5a reproduction — %s" % TAG[ds], fontsize=11)
        ax.grid(alpha=0.3, which="both")
        ax.legend(handles, labels, fontsize=8, loc="lower left", framealpha=0.9)
        out = os.path.join(FIGS, "A6_fig5a_local_slope_mod_%s.png" % ds)
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print("    saved %s" % out, flush=True)


if __name__ == "__main__":
    main()
