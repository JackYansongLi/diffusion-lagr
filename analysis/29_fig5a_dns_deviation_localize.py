#!/usr/bin/env python3
"""29_fig5a_dns_deviation_localize.py -- localises the large-tau deviation of the DNS in Fig. 5a (a
diagnostic script).

**Conclusion (measured 2026-10-07)**
  1) Test: feeding **the paper's own Fig. 4a source data** (Figure4a_DNS_p2/p4 of MOESM7, which sit on the
     same 1024 lags)
     through this project's differentiation gives zeta(4,tau) = **1.7687** at tau = 59, where Fig. 5a of
     the paper reports 1.7852 (ratio 0.991);
     our own S through the same method gives 1.6456 (0.930) => **the rise at large tau comes from their
     S_p itself**, not from the differentiation stencil or the order of the batch average.
  2) our S against theirs: **S2** 1.0000 (tau=1) -> 1.0016 (10) -> 1.0034 (19.7) -> 1.0051 (59), monotone
     and at most 0.6%;
     **S4** 1.0010 (1) -> 1.0101 (10) -> **1.0267 (19.7) -> 1.0294 (28.4)** -> 1.0232 (41) -> 1.0109 (59),
     i.e. **a bulge at intermediate tau of up to +2.9%**.
  3) the exact relation holds to 1e-4 point by point:
        ζ_o − ζ_p = (γ₄ − ζ_p·γ₂) / (d log S₂ᵒ/d log τ),   γ = d log(Sᵒ/Sᵖ)/d log τ
     => the 1-3% difference in the shape of S4 is amplified by **1/(dlogS2/dlogtau)**, the factor rising
     from **1.1** at tau = 6.6 to **4.2** at tau = 59
     (S2 flattens at the integral scale, so the denominator shrinks); **the S4 term contributes 93-97%**.
  4) cross-check with Fig. 4b: the ratio F4 = S4/S2^2 bulges by **+2.3% at tau ~ 20-28**, against the
     predicted dS4/S4 - 2 dS2/S2 = +2.0%,
     so **the large-tau residuals of the DNS in Figs. 4b and 5a share one origin**.

**Ruled out**: the batching and averaging order (our batch average differs from the pooled estimate by at
most 0.0055), the differentiation stencil (test 1) and sampling noise.
**The width of the error bars** is a separate matter (index-block structure; see the shuffled-slice
experiment of Fig. 4c) and has nothing to do with the mean studied here.

Output: results/figs/A15_fig5a_dns_deviation.png (four-panel diagnostic) and results/fig5a_dns_deviation.npz
Usage: python3 29_fig5a_dns_deviation_localize.py
"""
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
M7 = os.path.join(SRC, "42256_2024_810_MOESM7_ESM.txt")
M8 = os.path.join(SRC, "42256_2024_810_MOESM8_ESM.txt")
LAGS = np.arange(1, 1025)
LG = np.log10(LAGS.astype(float))


def blocks(fn):
    out, cur = {}, None
    for l in open(fn, errors="replace"):
        if l.startswith("#"):
            cur = None
            h = l[2:].split("(")[0].strip()
            if h.endswith(".out"):
                out[h[:-4]] = []
                cur = h[:-4]
        elif cur and l.strip():
            out[cur].append([float(x) for x in l.split()])
    return {k: np.array(v) for k, v in out.items()}


def grad(s):
    return np.gradient(np.log10(s), LG)


def main():
    b7, b8 = blocks(M7), blocks(M8)
    z = np.load(os.path.join(RESULTS, "flatness_dns_full.npz"))
    orders = [int(x) for x in z["orders"]]
    i2, i4 = orders.index(2), orders.index(4)
    # our mean curve = batch average (the convention of the delivered Fig. 5a)
    Sbc = z["Sbc"]
    nb, nl, _, nc = Sbc.shape
    S2o = Sbc[:, :, i2, :].mean(axis=(0, 2))
    S4o = Sbc[:, :, i4, :].mean(axis=(0, 2))
    # the S of the paper (Fig. 4a, the same 1024-lag grid)
    S2p, S4p = b7["Figure4a_DNS_p2"][:, 1], b7["Figure4a_DNS_p4"][:, 1]
    # the 31 points of Fig. 5a of the paper
    pa = b8["Figure5a_DNS"]
    tau5, z5 = pa[:, 0], pa[:, 1]
    L5 = np.rint(tau5 * 10).astype(int) - 1

    zel_p = grad(S4p) / grad(S2p)                 # their own S through our differentiation
    zel_o = grad(S4o) / grad(S2o)                 # ours
    r2, r4 = S2o / S2p, S4o / S4p
    g2, g4 = grad(r2), grad(r4)
    b_o = grad(S2o)
    pred = (g4 - zel_p * g2) / b_o
    act = zel_o - zel_p

    print("== 1) test: the paper's own S through our differentiation, against Fig. 5a ==")
    for t in (1.0, 13.7, 28.4, 41.0, 59.0):
        k = int(round(t * 10)) - 1
        print("   tau=%6.1f their S -> zeta %8.4f | paper 5a %8.4f (ratio %.4f) | ours %8.4f (ratio %.4f)"
              % (t, zel_p[k], z5[list(tau5).index(t)], zel_p[k] / z5[list(tau5).index(t)],
                 zel_o[k], zel_o[k] / z5[list(tau5).index(t)]))
    print("== 2) ratio of the S ==")
    print("   %7s %9s %9s" % ("tau", "S2 ratio", "S4 ratio"))
    for t in (1.0, 6.6, 13.7, 19.7, 28.4, 41.0, 59.0):
        k = int(round(t * 10)) - 1
        print("   %7.1f %9.4f %9.4f" % (t, r2[k], r4[k]))
    print("== 3) the exact relation: predicted against measured difference ==")
    print("   %7s %9s %9s %9s %9s %9s %8s" % ("tau", "γ4", "γ2", "ζ_p", "b_o", "predicted", "measured"))
    for t in (6.6, 13.7, 19.7, 28.4, 41.0, 59.0):
        k = int(round(t * 10)) - 1
        print("   %7.1f %+9.4f %+9.4f %9.4f %9.4f %+9.4f %+9.4f"
              % (t, g4[k], g2[k], zel_p[k], b_o[k], pred[k], act[k]))
    print("   amplification 1/b_o: tau=6.6 -> %.2f; tau=59 -> %.2f; S4 term at tau=59 -> %.0f%%"
          % (1 / b_o[uint(6.6 * 10) - 1], 1 / b_o[589], 100 * abs(g4[589]) / (abs(g4[589]) + abs(zel_p[589] * g2[589]))))
    print("== 4) cross-check with Fig. 4b: F4 = S4/S2^2 ==")
    f4o, f4p = S4o / S2o ** 2, b7["Figure4b_DNS_p4"][:, 1]
    r_f4 = f4o / f4p
    for t in (1.0, 6.6, 13.7, 19.7, 28.4, 41.0, 59.0):
        k = int(round(t * 10)) - 1
        print("   tau=%6.1f  F4 ratio %.4f (dS4/S4 - 2 dS2/S2 predicts %.4f)"
              % (t, r_f4[k], r4[k] - 1 - 2 * (r2[k] - 1) + 1))

    np.savez(os.path.join(RESULTS, "fig5a_dns_deviation.npz"), lags=LAGS, tau5=tau5,
             S2_ours=S2o, S4_ours=S4o, S2_paper=S2p, S4_paper=S4p, r2=r2, r4=r4,
             g2=g2, g4=g4, zeta_paper_from_paper_S=zel_p, zeta_ours=zel_o, pred=pred, act=act,
             f4_ratio=r_f4)

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.2))
    x = LAGS * 0.1
    # note: matplotlib on WSL has no CJK font and would draw empty boxes, so all figure text is English
    XR = (0.1, 102.4)
    ax[0, 0].semilogx(x, r2, "-", color="#1f77b4", lw=1.2, label=r"$S_2$ ours/paper")
    ax[0, 0].semilogx(x, r4, "-", color="#d62728", lw=1.2, label=r"$S_4$ ours/paper")
    ax[0, 0].axhline(1.0, color="0.6", ls=":", lw=1)
    ax[0, 0].set_xlim(*XR); ax[0, 0].set_title("(a) structure-function ratio: $S_4$ hump +2.9%", fontsize=10)
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3, which="both")
    ax[0, 1].semilogx(x, g4, "-", color="#d62728", lw=1.2, label=r"$\gamma_4=d\log(S_4^o/S_4^p)/d\log\tau$")
    ax[0, 1].semilogx(x, g2, "-", color="#1f77b4", lw=1.2, label=r"$\gamma_2$")
    ax[0, 1].axhline(0, color="0.6", ls=":", lw=1)
    ax[0, 1].set_xlim(*XR); ax[0, 1].set_ylim(-0.05, 0.06)
    ax[0, 1].set_title("(b) log-derivatives of the ratios", fontsize=10)
    ax[0, 1].legend(fontsize=8, loc="upper left"); ax[0, 1].grid(alpha=0.3, which="both")
    tw = ax[0, 1].twinx()                    # the amplification gets its own axis (log scale) so the spike at large tau does not flatten the main curve
    tw.semilogx(x, 1 / b_o, "-", color="0.45", lw=1.0, label=r"amplification $1/b_o$")
    tw.set_yscale("log"); tw.set_ylim(0.5, 5.0)
    tw.set_ylabel(r"$1/(d\log S_2^o/d\log\tau)$", fontsize=9)
    h1, l1 = ax[0, 1].get_legend_handles_labels(); h2, l2 = tw.get_legend_handles_labels()
    ax[0, 1].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax[1, 0].semilogx(x, act, "-", color="#1f77b4", lw=1.6, label=r"actual $\zeta_o-\zeta_p$")
    ax[1, 0].semilogx(x, pred, "--", color="k", lw=1.0, label=r"$(\gamma_4-\zeta_p\gamma_2)/b_o$")
    ax[1, 0].axhline(0, color="0.6", ls=":", lw=1)
    ax[1, 0].set_xlim(*XR); ax[1, 0].set_ylim(-0.14, 0.06)
    ax[1, 0].set_title("(c) exact identity holds pointwise $\Rightarrow$ root cause is $S_4$", fontsize=10)
    ax[1, 0].set_xlabel(r"$\tau/\tau_\eta$"); ax[1, 0].legend(fontsize=8, loc="lower left"); ax[1, 0].grid(alpha=0.3, which="both")
    ax[1, 1].semilogx(x, r_f4, "-", color="#2ca02c", lw=1.6, label=r"$F_4$ ours/paper (Fig.4b)")
    ax[1, 1].semilogx(x, r4 / r2 ** 2, "--", color="k", lw=1.0, label=r"prediction $r_4/r_2^2$ (exact)")
    ax[1, 1].axhline(1.0, color="0.6", ls=":", lw=1)
    ax[1, 1].set_xlim(*XR); ax[1, 1].set_title("(d) cross-check: same root cause as Fig.4b", fontsize=10)
    ax[1, 1].set_xlabel(r"$\tau/\tau_\eta$"); ax[1, 1].legend(fontsize=8, loc="lower left"); ax[1, 1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = os.path.join(RESULTS, "figs", "A15_fig5a_dns_deviation.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved %s" % out)


def uint(x):
    return int(round(x))


if __name__ == "__main__":
    main()
