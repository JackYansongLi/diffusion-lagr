"""15_fig2_accpdf_mod.py -- one figure, one script: Fig. 2 acceleration PDF with the
autocorrelation inset; revision.

Changes (2026-10-04): the fine histogram (Delta = 0.05 sigma) is re-binned onto the **coarse grid of
the paper** (~1 sigma) before plotting, which removes the sawtooth in the tail;
the axes follow the paper (x linear over +-100; y logarithmic 1e-11..1e1 with a tick every two
decades; the inset is log-x with linear y up to 2e-3);
the tail uses the same rule as Fig. 1a: the contiguous non-empty interval around the peak, with
single-count bins kept.
Split out of 08_fig1_fig2_figs_mod.py; the merged original is frozen and must not be edited.
Output: results/figs/A8_fig2_<ds>_mod.png
Usage: python 15_fig2_accpdf_mod.py --dataset dns|dm1c
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
os.makedirs(FIGS, exist_ok=True)

COLOR = {"dns": "k", "dm1c": "tab:blue", "dm3c": "tab:orange"}
LABEL = {"dns": "DNS", "dm1c": "DM-1c", "dm3c": "DM-3c"}


def parse_src(fn, blk):
    cur, rows = None, []
    for l in open(os.path.join(SRC, fn), errors="replace"):
        if l.startswith("#"):
            cur = l[1:].strip()
        elif l.strip() and cur == blk:
            rows.append([float(x) for x in l.split()])
    return np.array(rows)


def trim_ends(xc, hc, min_count=2):
    """Keep only the contiguous non-empty run that contains the peak, then retreat at each end to the
    outermost bin whose count is >= min_count.

    Rationale (checked 2026-10-03/04):
      - the published curve (MOESM4) has **no zero-valued point** inside its range, and every y value
        is an integer multiple of 1/(N*d) => it is the raw count histogram, drawn only over the
        contiguous non-empty run that holds the peak;
      - the ranges of the paper, +-29.59/+-18.88/+-12.28/+-5.43 sigma for the four lags, are what
        this rule produces;
      - a plain count threshold leaves isolated islands separated by zeros at the ends, which draw
        as broken lines, whereas this rule has no holes by construction.
    """
    if len(hc) < 3:
        return xc, hc
    pk = int(np.argmax(hc))
    lo = pk
    while lo > 0 and hc[lo - 1] > 0:
        lo -= 1
    hi = pk
    while hi < len(hc) - 1 and hc[hi + 1] > 0:
        hi += 1
    keep = np.nonzero(hc[lo:hi + 1] >= min_count)[0]
    if len(keep) < 3:
        return xc[lo:hi + 1], hc[lo:hi + 1]
    a, b = lo + keep[0], lo + keep[-1]
    return xc[a:b + 1], hc[a:b + 1]


def bridge_gaps(xc, hc, max_gap=1):
    """Fill runs of at most max_gap zero-count bins by log-linear interpolation between the
    neighbouring non-zero bins.

    Rationale (2026-10-04): a few places in our aggregated bins hold exactly one zero-count bin
    (e.g. tau/tau_eta = 5, right tail near 12.0 sigma). It truncates the run around the peak and
    shortens the curve by about 0.4 sigma against the paper, whose bin edges differ and which has
    no gap there. Bridging a single hole restores the extension seen in the paper, while a run of
    more than max_gap empty bins is a genuine statistical void and stays at zero (the curve
    breaks). The filled values are a visual interpolation only; the raw counts remain in
    figstats_*.npz. Returns (xc, hc_float, number of filled bins).
    """
    hcf = np.asarray(hc, dtype=float).copy()
    z = np.nonzero(hcf == 0)[0]
    if len(z) == 0:
        return xc, hcf, 0
    runs, start = [], int(z[0])
    for i in range(1, len(z)):
        if z[i] != z[i - 1] + 1:
            runs.append((start, int(z[i - 1])))
            start = int(z[i])
    runs.append((start, int(z[-1])))
    n_fill = 0
    for a, b in runs:
        if (b - a + 1) > max_gap or a == 0 or b == len(hcf) - 1:
            continue
        if hcf[a - 1] <= 0 or hcf[b + 1] <= 0:
            continue
        la, lb = np.log(hcf[a - 1]), np.log(hcf[b + 1])
        for k in range(a, b + 1):
            hcf[k] = np.exp(la + (k - a + 1) / (b + 1 - a + 1) * (lb - la))
            n_fill += 1
    return xc, hcf, n_fill


def fold_hist(h, edges):
    """Fold the fine histogram onto the |x| half-axis: hf[r] = h(+r) + h(-r).

    Statistically legitimate (the normalized-increment PDF is left-right symmetric; the measured
    skewness is -0.0096..-0.044). The gain:
      - each |x| bin holds twice the counts, so the noise drops by sqrt(2) and the >= nmin
        criterion reaches further out;
      - the left and right ranges become symmetric by construction, so the independent-side
        convention cannot turn Poisson fluctuations at the 0/1-count tail into asymmetric ranges
        (e.g. DNS at tau/tau_eta = 1: -27.9 sigma on the left against +31.6 sigma on the right).
    Cost: the real skewness is erased (here |S3/sigma^3| <= 0.044, negligible), and the
    presentation differs from the paper, which draws each side separately.
    """
    n = len(h)
    half = n // 2
    return h[half:] + h[half - 1::-1], edges[half:]


def adaptive_curve(h, edges, agg0, nmin, absorb_min=0, x_cap=None):
    """Counting-rule curve, independent of the paper: bins are accumulated outward from the peak
    into groups, and **a point is drawn only when the group holds >= nmin counts**.

    - the core resolution is agg0 fine bins (by default support/--nbins, the resolution of the
      paper);
    - toward the tails the group size doubles as needed (agg0 -> 2 agg0 -> 4 agg0 ...) until the
      group holds >= nmin counts;
    - absorb_min > 0: if the end of a side cannot reach nmin, the **actual counted interval** of
      the remaining tail on that side may be absorbed into one last point (which needs
      >= absorb_min counts), which maximises the range (--absorb-tail);
    - x_cap limits the excursion to +-x_cap (used by --symmetrize range: symmetric range with
      per-side values).

    Returns (x, counts, group width in sigma) sorted by x.
    """
    n = len(h)
    wb = float(edges[1] - edges[0])
    ctr = 0.5 * (edges[:-1] + edges[1:])
    pk = int(np.argmax(h))
    if x_cap is None:
        i_pos, i_neg = n - 1, 0
    else:
        i_pos = int(np.searchsorted(ctr, x_cap))
        i_neg = int(np.searchsorted(ctr, -x_cap))
    xs, cs, ds = [], [], []
    for s in (-1, 1):
        i = pk
        lo_b, hi_b = (pk, i_pos) if s > 0 else (i_neg, pk)
        while lo_b <= i <= hi_b:
            g = agg0
            while True:
                lo, hi = (i, min(n, i + g, hi_b + 1)) if s > 0 else (max(0, i - g, lo_b), i)
                if hi <= lo:
                    break
                if float(h[lo:hi].sum()) >= nmin or lo == 0 or hi == n or \
                        hi == hi_b + 1 or lo == lo_b:
                    break
                g *= 2
                if g > n:
                    break
            lo, hi = (i, min(n, i + g, hi_b + 1)) if s > 0 else (max(0, i - g, lo_b), i)
            if hi <= lo:
                break
            c = float(h[lo:hi].sum())
            if c < nmin:          # the tail cannot reach nmin
                at_edge = (lo == 0 or hi == n or hi == hi_b + 1 or lo == lo_b)
                if not (absorb_min > 0 and c >= absorb_min and at_edge):
                    break
                nz = np.nonzero(h[lo:hi])[0]      # shrink the group to the interval that actually holds counts
                if len(nz):                       # (empty bins must not enter the width, so that it reflects the real data span)
                    lo, hi = lo + int(nz[0]), lo + int(nz[-1]) + 1
            xc = 0.5 * (edges[lo:hi] + edges[lo + 1:hi + 1])
            xs.append(float((xc * h[lo:hi]).sum() / c))
            cs.append(c)
            ds.append((hi - lo) * wb)
            i = i + s * g
    xs = np.array(xs); cs = np.array(cs); ds = np.array(ds)
    o = np.argsort(xs)
    return xs[o], cs[o], ds[o]


def coarse_bins(h, edges, nbins=200):
    """Aggregate the fine histogram into about nbins equal coarse bins inside the statistical
    support.

    Motivation (checked 2026-10-03): on our fine grid (Delta = 0.02 sigma) the tail bins hold only
    0 or 1 counts (1 count => PDF = 2.6e-8), so the histogram is jagged in the tail, whereas the
    published Fig. 1a curves carry about 200 points inside each curve's own support (Delta x ~
    support/200: 0.325/0.189/0.123/0.054 sigma), a much coarser effective resolution. Aggregating
    the fine bins by an integer factor is equivalent to re-binning with the width of the paper,
    which allows a like-for-like comparison without re-running script 07. The fine histograms
    stay in figstats_*.npz.

    Returns (agg, coarse bin centres, coarse counts); PDF = counts / (total counts * agg * fine
    width).
    """
    nz = np.nonzero(h)[0]
    if len(nz) < 3:
        return 1, 0.5 * (edges[:-1] + edges[1:]), h
    lo, hi = int(nz[0]), int(nz[-1])
    agg = max(1, int(round((hi - lo + 1) / float(nbins))))
    m = max(1, (hi - lo + 1) // agg)
    hc = h[lo:lo + m * agg].reshape(m, agg).sum(axis=1)
    e = edges[lo:lo + m * agg + 1:agg]
    return agg, 0.5 * (e[:-1] + e[1:]), hc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["dns", "dm3c", "dm1c"])
    ap.add_argument("--out-suffix", default="_mod",
                    help="output-suffix (default _mod; the initial version writes A8_fig2_<ds>.png)")
    ap.add_argument("--nbins", type=int, default=200,
                    help="number of core bins for the counting rule (support/nbins, the resolution of the paper; default 200)")
    ap.add_argument("--min-count-point", type=int, default=0,
                    help="> 0 switches to the independent counting rule (adaptive grouping, every drawn point holding at least this many counts, e.g. 20), "
                         "replacing the coarse grid of the paper as the main convention (default 0 = that grid)")
    ap.add_argument("--symmetrize", choices=["none", "fold", "range"], default="none",
                    help="with --min-count-point: none = per side; fold = fold before the excursion (symmetric); "
                         "range = symmetric range (half-width from the folded rule) with per-side values, the convention of the paper")
    ap.add_argument("--with-retrained", action="store_true",
                    help="additionally draw our retrained DM-1c (figstats_dm1c_repro.npz) in the "
                         "same style; used for the merged comparison panel of the document")
    ap.add_argument("--absorb-tail", type=int, default=0,
                    help="with the counting rule: > 0 absorbs the remaining tail of an end that cannot reach the threshold into the last point (maximises the range)")
    args = ap.parse_args()
    ds = args.dataset

    z = np.load(os.path.join(RESULTS, "figstats_%s.npz" % ds))
    z2 = (np.load(os.path.join(RESULTS, "figstats_dm1c_repro.npz"))
          if args.with_retrained else None)

    # ============ Fig. 2 (revision: bin width and axes of the paper) ============
    ea = z["acc_edges"]
    ca_ = 0.5 * (ea[:-1] + ea[1:])
    Ha = z["hist_acc"]
    wb = ea[1] - ea[0]
    pa = Ha / (Ha.sum() * wb)                     # fine-grid PDF (Delta = 0.05 sigma), used for the quantitative checks only
    Cv = z["Cv"]                                  # ⟨vi(t)vi(t+τ)⟩ per comp, τ=0..1025
    # a(t)=v(t+1)-v(t) → Ca(τ)=⟨a(t)a(t+τ)⟩=2C(τ)-C(τ-1)-C(τ+1), τ=0.1..102.3
    Ca = 2 * Cv[1:-2] - Cv[:-3] - Cv[2:-1]
    tau_axis = np.arange(1, len(Ca) + 1) * 0.1

    ref = parse_src("42256_2024_810_MOESM5_ESM.txt",
                    {"dns": "Figure2_DNS.out", "dm1c": "Figure2_DM-1c.out",
                     "dm3c": "Figure2_DNS.out"}[ds])
    # change (2026-10-04: the initial version showed a sawtooth, so it was revised in _mod style):
    #   on our fine grid (Delta = 0.05 sigma) 15% (DNS) / 13% (DM-1c) of the non-empty tail bins hold <= 1 count
    #   (1 count => PDF ~ 2.6e-8), so joining the points on a log axis gives a sawtooth; the source data of Fig. 2 use
    #   coarse bins of about 1 sigma (DNS Delta ~ 0.989 sigma -> 143 points, DM-1c Delta ~ 0.975 sigma -> 170 points).
    #   => same convention as Fig. 1a: re-bin the fine histogram onto the coarse grid of the paper (point by point),
    #     keeping the tail rule of trim_ends (the contiguous non-empty interval around the peak, no bridging inside).
    #   the axes follow the paper as well: x linear over +-100, y logarithmic 10^-11..10^1 with a tick every two decades.
    xp = ref[:, 0]
    Wp = float(np.median(np.diff(xp)))
    lo = xp - Wp / 2.0
    hi = xp + Wp / 2.0
    j = np.searchsorted(lo, ca_, side="right") - 1
    valid = (j >= 0) & (j < len(xp)) & (ca_ >= lo[0]) & (ca_ < hi[-1])
    cnt = np.bincount(j[valid], weights=Ha[valid], minlength=len(xp))
    pdf_coarse = cnt / (Ha.sum() * Wp)
    xc_p, hc_p = trim_ends(xp, cnt, min_count=1)
    keep = np.isin(xp, xc_p)

    # ---- main-plot data: the coarse grid of the paper by default; --min-count-point > 0 = the independent counting rule (as in Fig. 1a) ----
    if args.min_count_point > 0:
        nza = np.nonzero(Ha)[0]
        agga = max(1, int(round((nza[-1] - nza[0] + 1) / float(args.nbins))))
        if args.symmetrize == "range":
            _hf, _ef = fold_hist(Ha, ea)
            _xh, _ch, _dh = adaptive_curve(_hf, _ef, agga, args.min_count_point)
            _xsym = float(_xh.max())
            xc_plot, hc_plot, dgp = adaptive_curve(Ha, ea, agga, args.min_count_point,
                                                   absorb_min=max(1, args.absorb_tail),
                                                   x_cap=_xsym)
            lab_plot = ("%s (this work, adaptive ≥%d cnt/pt, %.0f..%.0f$\\sigma$)"
                        % (LABEL[ds], args.min_count_point, xc_plot.min(), xc_plot.max()))
        elif args.symmetrize == "fold":
            _hf, _ef = fold_hist(Ha, ea)
            _xh, _ch, _dh = adaptive_curve(_hf, _ef, agga, args.min_count_point)
            xc_plot = np.concatenate([-_xh[::-1], _xh])
            hc_plot = np.concatenate([_ch[::-1], _ch])
            dgp = np.concatenate([_dh[::-1], _dh])
            lab_plot = "%s (this work, adaptive ≥%d cnt/pt, folded)" % (LABEL[ds], args.min_count_point)
        else:
            xc_plot, hc_plot, dgp = adaptive_curve(Ha, ea, agga, args.min_count_point,
                                                   absorb_min=args.absorb_tail)
            lab_plot = "%s (this work, adaptive ≥%d cnt/pt)" % (LABEL[ds], args.min_count_point)
        pdf_plot = hc_plot / (Ha.sum() * dgp)
        print("[Fig.2] counting rule (%s): core %dx%.2f sigma = %.2f sigma; %d points; range %.2f..%.2f sigma; last point %d/%d counts"
              % (args.symmetrize, agga, wb, agga * wb, len(xc_plot), xc_plot.min(), xc_plot.max(),
                 int(hc_plot[0]), int(hc_plot[-1])), flush=True)
    else:
        ok = keep & (pdf_coarse > 0)
        xc_plot, pdf_plot = xp[ok], pdf_coarse[ok]
        lab_plot = "%s (this work, $\\Delta x$=%.3f$\\sigma$)" % (LABEL[ds], Wp)

    fig = plt.figure(figsize=(7.6, 5.6))
    axm = fig.add_axes([0.115, 0.11, 0.865, 0.83])
    # merged mode: short labels, so the third curve does not crowd the legend
    lab_self = "DM-1c (paper samples)" if args.with_retrained else lab_plot
    lab_ref = ("paper" if args.with_retrained
               else "paper %s ($\\Delta x$=%.3f$\\sigma$)" % (LABEL[ds], Wp))
    # 2026-10-04: the reproduction curve uses a plain line instead of symbols (the paper keeps a dotted line)
    axm.semilogy(xc_plot, pdf_plot, "-", color=COLOR[ds], lw=1.3, label=lab_self)
    if args.with_retrained:
        # same coarse grid (the paper's xp/Wp) and the same trim rule as the main curve
        ea2 = z2["acc_edges"]
        ca2 = 0.5 * (ea2[:-1] + ea2[1:])
        Ha2 = z2["hist_acc"]
        j2 = np.searchsorted(lo, ca2, side="right") - 1
        v2 = (j2 >= 0) & (j2 < len(xp)) & (ca2 >= lo[0]) & (ca2 < hi[-1])
        cnt2 = np.bincount(j2[v2], weights=Ha2[v2], minlength=len(xp))
        pdf2 = cnt2 / (Ha2.sum() * Wp)
        ok2 = keep & (pdf2 > 0)
        axm.semilogy(xp[ok2], pdf2[ok2], "-", color="tab:red", lw=1.3,
                     label="DM-1c (ours)")
    mr = ref[:, 1] > 0
    axm.semilogy(ref[mr, 0], ref[mr, 1], ":", color="0.45", lw=1.3, label=lab_ref)
    axm.set_xlim(-100, 100)
    axm.set_ylim(1e-11, 1e1)
    axm.set_yticks(10.0 ** np.arange(1, -12, -2))
    axm.set_xlabel("$a_i/\\sigma(a_i)$")
    axm.set_ylabel("$\\mathrm{PDF}(a_i)\\times\\sigma(a_i)$")
    axm.grid(alpha=0.22, which="both", lw=0.4)
    axm.legend(fontsize=8, loc="lower center")
    axm.set_title("Fig.2 reproduction: acceleration PDF (%s%s)"
                  % (LABEL[ds], ", released and retrained" if args.with_retrained else ""),
                  fontsize=10)

    # inset: the paper places it in the upper right corner, log-x (10^0..10^2) and linear y (ticks 0/0.001/0.002)
    # upper right corner, about 3/8 of the axes (0.375 x 0.375); bottom at y ~ 8e-5, left at x ~ 22 sigma,
    # so it no longer covers the curve
    axi = axm.inset_axes([0.61, 0.575, 0.375, 0.375])
    cam = Ca.mean(axis=1)
    axi.semilogx(tau_axis, cam, color=COLOR[ds], lw=1.1,
                 label=("DM-1c (paper samples)" if args.with_retrained else LABEL[ds]))
    if args.with_retrained:
        Cv2 = z2["Cv"]
        Ca2 = 2 * Cv2[1:-2] - Cv2[:-3] - Cv2[2:-1]
        axi.semilogx(np.arange(1, len(Ca2) + 1) * 0.1, Ca2.mean(axis=1),
                     color="tab:red", lw=1.1, label="DM-1c (ours)")
    refi = parse_src("42256_2024_810_MOESM5_ESM.txt",
                     {"dns": "Figure2_inset_DNS.out", "dm1c": "Figure2_inset_DM-1c.out",
                      "dm3c": "Figure2_inset_DNS.out"}[ds])
    axi.semilogx(refi[:, 0], refi[:, 1], ":", color="0.45", lw=1.1, label="paper")
    axi.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    axi.set_xlim(0.07, 1.15e2)   # the left end reaches slightly below the smallest lag (tau = 0.1), the right end slightly beyond 102.4
    axi.set_ylim(-2.5e-4, 2.1e-3)
    axi.set_yticks([0.0, 1e-3, 2e-3])
    axi.set_xlabel("$\\tau/\\tau_\\eta$", fontsize=8)
    axi.set_ylabel("$C_\\tau$", fontsize=8)       # paper: Ctau = <a_i(t+tau) a_i(t)>, i = x, y, z
    axi.tick_params(labelsize=7)
    axi.legend(fontsize=7, loc="upper right")
    p2 = os.path.join(FIGS, "A8_fig2_%s%s.png" % (ds, args.out_suffix))
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print("saved", p2)

    # ---- quantitative comparison (point by point on the shared grid, coverage, tail probability, inset) ----
    mq = (ref[:, 1] > 1e-7) & (pdf_coarse > 0)
    if mq.sum() > 3:
        rel = np.abs(pdf_coarse[mq] - ref[mq, 1]) / ref[mq, 1]
        q = "n=%d median %.2f%%  90th percentile %.2f%%" % (mq.sum(), np.median(rel) * 100,
                                                np.quantile(rel, 0.9) * 100)
    else:
        q = "too few comparison points"
    n0 = min(len(cam), len(refi))
    ci = np.corrcoef(cam[:n0], refi[:n0, 1])[0, 1]
    zc_us = tau_axis[cam < 0][0] if (cam < 0).any() else float("nan")
    zc_pap = refi[refi[:, 1] < 0, 0][0]
    tail10 = pa[np.abs(ca_) > 10].sum() * wb
    tail_pap = ref[np.abs(ref[:, 0]) > 10, 1].sum() * Wp
    print("Fig.2 quantitative: point-by-point deviation on the shared grid %s | peak ours (coarse) %.3f vs paper %.3f | "
          "|a| > 10 sigma %.4g vs paper %.4g | coverage %.1f..%.1f sigma (paper %.1f..%.1f sigma) | "
          "points %d/%d (single-count %d) | last point %d/%d counts"
          % (q, pdf_coarse.max(), ref[:, 1].max(), tail10, tail_pap,
             xp[keep].min(), xp[keep].max(), xp.min(), xp.max(),
             int(keep.sum()), len(xp), int((cnt[keep] <= 1).sum()),
             int(cnt[keep][0]), int(cnt[keep][-1])))
    print("  inset: corr=%.6f  zero crossing tau/tau_eta ours %.1f vs paper %.1f" % (ci, zc_us, zc_pap))


if __name__ == "__main__":
    main()
