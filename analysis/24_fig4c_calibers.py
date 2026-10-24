#!/usr/bin/env python3
"""24_fig4c_calibers.py -- checks the error-bar conventions of Fig. 4c: A / B / guess 1 / guess 2,
four envelopes compared point by point with the paper.

Background (analysis of 2026-10-07): for the bars of the mixed-component fourth-order flatness
F_tau^(4,ij),
  convention A (10 trajectory blocks, the three component pairs averaged inside a block, every block
  normalised by its own S2) is 4-10x too narrow,
  while convention B (10 blocks x 3 component pairs = 30 curves taking a global min/max) is 3-9x
  too wide.
Two guesses were made about what the paper actually does:
  guess 1 (global denominator): F_k = (1/3) sum_ij <di^2 dj^2>_Bk / [S2_global]^2, so the
  numerator fluctuations are no longer cancelled by their own denominators and the envelope widens;
  guess 2 (extremes per component pair, then averaged): Lower = (1/3) sum_ij min_k F_ij,Bk and
  likewise for the upper edge, which lands between A and B.

The statistics are not recomputed: the per-block numerator is recovered from Fb / Fb_pairs in
fig4c_mixed_<ds>.npz as
Xbar_Bk = Fb_Bk * (S2_b,Bk)^2 (Fb is normalised by that block's own S2_b),
with S2_b,Bk taken from Sbc[:, :, p=2, :] of flatness_<ds>_full.npz (every block and component) and
S2_b the average over the three components.
The two files are cross-checked through their own S2, which should agree and thus prove that both
batch the data identically.

Output: results/fig4c_calibers_<ds>.npz (the lo/hi and width of each convention plus its point-by-point
deviation from the paper)
Usage: python3 24_fig4c_calibers.py [--datasets dm3c,dns]
"""
import argparse
import os

import numpy as np

ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SRC = os.path.join(ROOT, "Data", "Li_et_al_2024_source_data")
TAG = {"dm3c": "DM-3c", "dns": "DNS"}


def paper_band(ds):
    fn = os.path.join(SRC, "42256_2024_810_MOESM7_ESM.txt")
    lines = open(fn, errors="replace").read().split("\n")
    i = [k for k, l in enumerate(lines) if l.startswith("# Figure4c_%s.out" % TAG[ds])][0]
    pa = np.array([[float(x) for x in lines[i + 1 + j].split()] for j in range(1024)])
    return pa[:, 1] - pa[:, 2], pa[:, 1] + pa[:, 3]        # lo, hi of the paper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dm3c,dns")
    args = ap.parse_args()

    for ds in args.datasets.split(","):
        z = np.load(os.path.join(RESULTS, "fig4c_mixed_%s.npz" % ds))
        f18 = np.load(os.path.join(RESULTS, "flatness_%s_full.npz" % ds))
        F = z["F_mean"]
        Fb = z["Fb"]                                       # (nb, nl) per block, pairs averaged, normalised by the block's own S2
        Fbp = z["Fb_pairs"]                                # (nb, nl, 3)
        nb, nl = Fb.shape
        S2_b = f18["Sbc"][:, :, 0, :].mean(axis=2)         # (nb, nl) S2 per block, averaged over the three components
        S2_g = f18["mean"][:, 0]                           # (nl,) the global S2
        S2_g2 = z["S2_mean"]                               # this script's own global S2, used for the cross-check

        # cross-check: the S2 of the two files should agree, proving identical batching
        rel = np.median(np.abs(S2_g2 / S2_g - 1))
        print("[%s] cross-check: S2_global here against S2_global of the flatness npz => median relative difference %.2e %s"
              % (ds, rel, "OK, identical batching" if rel < 1e-3 else "MISMATCH, the batching differs"), flush=True)

        Xb = Fb * (S2_b ** 2)                              # recover the per-block numerator (already averaged over ij)
        Xbp = Fbp * (S2_b ** 2)[:, :, None]                # recover the numerator of every block and component pair
        c1 = Xb / (S2_g ** 2)                              # guess 1: global denominator
        e1 = (Xbp / (S2_g ** 2)[None, :, None]).reshape(-1, nl)   # variant E: numerator split per component pair with a global denominator (30 curves)
        try:                                               # variant F: the global S2 of the DNS as denominator (a cross-dataset normalisation)
            _f = np.load(os.path.join(RESULTS, "flatness_dns_full.npz"))
            S2_g_dns = _f["mean"][:, 0]
            c_dnsden = Xb / (S2_g_dns ** 2)
        except Exception:                                  # noqa: BLE001
            c_dnsden = c1
        g1_lo = (Xbp / (S2_g ** 2)[None, :, None]).min(axis=0).mean(axis=1)   # variant G: extremes per pair with the global denominator, then averaged
        g1_hi = (Xbp / (S2_g ** 2)[None, :, None]).max(axis=0).mean(axis=1)
        # ---- variant 1 (2026-10-07): the denominator becomes the per-component product S2i*S2j ----
        S2bc = f18["Sbc"][:, :, 0, :]                      # (nb, nl, C) the per-component S2 of every block
        S2gc = S2bc.mean(axis=0)                           # (nl, C) the global per-component S2

        def _pairden(S2c):                                 # (k, nl, C) → (k, nl, 3)
            return np.stack([S2c[:, :, i] * S2c[:, :, j] for i, j in ((0, 1), (0, 2), (1, 2))], axis=2)

        Fd3 = Xbp / _pairden(np.repeat(S2gc[None], nb, axis=0))   # variant 1a: global, per component
        Fd2 = Xbp / _pairden(S2bc)                                # variant 1b: per block, per component

        env = {
            "A(10批,批内对平均)": (Fb.min(axis=0), Fb.max(axis=0)),
            "B(30分量对批)": (Fbp.reshape(-1, nl).min(axis=0), Fbp.reshape(-1, nl).max(axis=0)),
            "猜测1(全局S2分母)": (c1.min(axis=0), c1.max(axis=0)),
            "猜测2(逐对取极值再平均)": (Fbp.min(axis=0).mean(axis=1), Fbp.max(axis=0).mean(axis=1)),
            "E(分子拆分+全局分母)": (e1.min(axis=0), e1.max(axis=0)),
            "F(用DNS的S2做分母)": (c_dnsden.min(axis=0), c_dnsden.max(axis=0)),
            "G(逐对极值+全局分母再平均)": (g1_lo, g1_hi),
            "变体1a(D3 全局逐分量分母,先平均)": (Fd3.mean(2).min(0), Fd3.mean(2).max(0)),
            "变体1a(D3 全局逐分量分母,先取极值)": (Fd3.min(0).mean(1), Fd3.max(0).mean(1)),
            "变体1b(D2 逐批逐分量分母,先平均)": (Fd2.mean(2).min(0), Fd2.mean(2).max(0)),
            "变体1b(D2 逐批逐分量分母,先取极值)": (Fd2.min(0).mean(1), Fd2.max(0).mean(1)),
        }
        plo, phi = paper_band(ds)
        w_pa = np.median((phi - plo) / F)
        print("  relative full width of the paper (median) %.4f" % w_pa, flush=True)
        print("  %-24s %-10s %-10s %-12s %-12s" % ("convention", "rel. width", "ours/paper", "lo dev. (med.)", "hi dev. (med.)"), flush=True)
        best = None
        store = {}
        for name, (lo, hi) in env.items():
            w = np.median((hi - lo) / F)
            dlo = np.median(np.abs(lo - plo) / plo)
            dhi = np.median(np.abs(hi - phi) / phi)
            score = 0.5 * (dlo + dhi)
            print("  %-24s %-10.4f %-10.2f %-12.4f %-12.4f" % (name, w, w / w_pa, dlo, dhi), flush=True)
            store[name] = (lo, hi)
            if best is None or score < best[1]:
                best = (name, score)
        print("  => closest to the envelope of the paper: %s (relative deviation %.4f)" % (best[0], best[1]), flush=True)

        out = os.path.join(RESULTS, "fig4c_calibers_%s.npz" % ds)
        np.savez_compressed(out, tau=z["tau"], F=F, paper_lo=plo, paper_hi=phi,
                            names=np.array(list(store.keys())),
                            los=np.array([store[k][0] for k in store]),
                            his=np.array([store[k][1] for k in store]))
        print("  wrote %s\n" % out, flush=True)


if __name__ == "__main__":
    main()
