"""14_fig1a_pdf_mod.py -- one figure, one script: reproduction of Fig. 1a of the paper
(velocity-increment PDFs at tau/tau_eta = 1, 2, 5, 100); revision with the tail rule, the
adaptive counting convention and the symmetrization switches.

Split out of 08_fig1_fig2_figs_mod.py; the merged original is frozen as a historical version
and must not be edited.
Every figure has its own pair of scripts (initial version + _mod revision) and they do not
affect each other.
Output: results/figs/A7_fig1a_<ds>_mod.png
Usage: python 14_fig1a_pdf_mod.py --dataset dns|dm1c
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
    ap.add_argument("--tail-min-count", type=int, default=1,
                    help="end rule for Fig. 1a: take the contiguous non-empty run around the peak, then retreat to the outermost bin holding at least this many counts."
                         "0 = no trimming (keeps isolated islands and broken lines); 1 = draw to the end of the run, as the paper does, "
                         "the final setting chosen for this figure on 2026-10-04; >= 2 = retreat to that count")
    ap.add_argument("--nbins", type=int, default=200,
                    help="number of coarse bins for Fig. 1a (the paper resolves about support/200; default 200)")
    ap.add_argument("--out-suffix", default="_mod",
                    help="output-suffix for Fig. 1a (default _mod), used for parameter comparisons, e.g. "
                         "--tail-min-count 1 --out-suffix _mod_tailmin1; the Fig. 2 output name is unaffected")
    ap.add_argument("--absorb-tail", type=int, default=0,
                    help="maximise the range: > 0 lets each end that cannot reach --min-count-point fold "
                         "its whole remaining tail into one last point (which needs at least this many counts, e.g. 2); left and right stay independent")
    ap.add_argument("--symmetrize", choices=["none", "fold", "range"], default="none",
                    help="none = per-side excursion (default); fold = fold onto |x| first (symmetric range and values, "
                         "noise down by sqrt(2)); range = symmetric range (half-width from the folded rule) with **per-side values**"
                         "(the presentation of the paper, whose own asymmetry is only 0.00-1.08 sigma and which does not fold)")
    ap.add_argument("--min-count-point", type=int, default=0,
                    help="independent counting rule: > 0 switches to adaptive grouping, where every drawn point holds at least this many counts (e.g. 20 or 30) and "
                         "the core bins stay at support/--nbins, so the comparison does not depend on the paper")
    ap.add_argument("--bridge-gaps", type=int, default=0,
                    help="bridge zero-count gaps of at most this length by log-linear interpolation (0 = no bridging, the default); "
                         "1 = bridge single holes only, which reproduces the tail extension of the paper (e.g. 12.28 sigma at tau/tau_eta = 5)")
    args = ap.parse_args()
    ds = args.dataset

    z = np.load(os.path.join(RESULTS, "figstats_%s.npz" % ds))
    edges = z["inc_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    H = z["hist_inc"]
    pdf_lags = z["pdf_lags"]

    # ============ Fig.1a ============
    # the paper's curves are shifted by a decade^-1.3 each (the first one is not); ours uses the same shift
    fig, ax = plt.subplots(figsize=(7.5, 8))   # same canvas as the DM-1c panel (48_mod)
    src_fn = "42256_2024_810_MOESM4_ESM.txt"
    ref_blocks = {}
    if ds in ("dns", "dm1c"):
        pre = {"dns": "Figure1a_DNS_tau", "dm1c": "Figure1a_DM-1c_tau"}[ds]
        for tau in pdf_lags:
            try:
                ref_blocks[tau] = parse_src(src_fn, pre + str(tau // 10))
            except KeyError:
                pass
    for k, tau in enumerate(pdf_lags):
        wb = edges[1] - edges[0]
        agg, xc, hc = coarse_bins(H[k], edges, nbins=args.nbins)
        n_fill = 0
        if args.bridge_gaps > 0:      # bridge single empty bins first, then trim the ends (the run is longer after bridging)
            xc, hc, n_fill = bridge_gaps(xc, hc, args.bridge_gaps)
        if args.tail_min_count > 0:   # 0 = no trimming; 1 = draw to the end of the run around the peak (as the paper does)
            xc, hc = trim_ends(xc, hc, max(1, args.tail_min_count))
        if args.min_count_point > 0:
            # independent counting rule: a core of agg fine bins, the group doubling toward the tails until every point reaches nmin
            if args.symmetrize == "fold":
                hf, ef = fold_hist(H[k], edges)
                x_h, c_h, d_h = adaptive_curve(hf, ef, agg, args.min_count_point)  # do not use xs/cs/ds here (ds is the dataset name)
                xc = np.concatenate([-x_h[::-1], x_h])
                hc = np.concatenate([c_h[::-1], c_h])
                dg = np.concatenate([d_h[::-1], d_h])
                pdf = hc / (2.0 * H[k].sum() * dg)     # folded counts are the sum of both sides, so dividing by 2 restores the one-sided PDF
            elif args.symmetrize == "range":
                # symmetric range (half-width from the folded rule) with per-side values: the presentation of the paper
                hf, ef = fold_hist(H[k], edges)
                x_h, c_h, d_h = adaptive_curve(hf, ef, agg, args.min_count_point)
                x_sym = float(x_h.max())
                xc, hc, dg = adaptive_curve(H[k], edges, agg, args.min_count_point,
                                            absorb_min=max(1, args.absorb_tail),
                                            x_cap=x_sym)
                pdf = hc / (H[k].sum() * dg)
            else:
                xc, hc, dg = adaptive_curve(H[k], edges, agg, args.min_count_point,
                                            absorb_min=args.absorb_tail)
                pdf = hc / (H[k].sum() * dg)
            if k == 0:
                i_end = int(np.argmax(np.abs(xc)))
                print("[Fig.1a] counting rule (adaptive%s): every point holds >= %d counts (log10 error <= %.3f decade); %d points; "
                      "range +-%.2f sigma; outermost point %d counts, Delta = %.2f sigma"
                      % ((("|folded" if args.symmetrize == "fold" else "") +
                          ("|symmetric range" if args.symmetrize == "range" else "") +
                          ("|absorbed tail >= %d" % args.absorb_tail if args.absorb_tail > 0 else "")),
                         args.min_count_point,
                         0.434 / np.sqrt(args.min_count_point), len(hc),
                         float(np.max(np.abs(xc))), int(hc[i_end]), float(dg[i_end])), flush=True)
        else:
            pdf = np.where(hc > 0, hc / (H[k].sum() * agg * wb), np.nan)   # zero bins break the curve instead of being joined
            if k == 0:
                inner_z = int(np.sum(hc == 0))
                print("[Fig.1a] fine bins Delta = %.3f sigma; aggregated %dx -> Delta = %.4f sigma; curve range +-%.2f sigma (end count >= %d); "
                      "%d empty bins inside; %d gaps bridged (of at most %d bins); 1 count => PDF = %.2e"
                      % (wb, agg, agg * wb, float(np.nanmax(np.abs(xc))), args.tail_min_count,
                         inner_z, args.bridge_gaps, n_fill, 1.0 / (H[k].sum() * agg * wb)), flush=True)
        sh = 10.0 ** (-1.3 * k)
        # one entry for the whole family, listing all four lags -- same style as the
        # DM-1c panel (user, 2026-10-22)
        ax.semilogy(xc, pdf * sh, color=COLOR[ds], lw=1.2,
                    label=("%s, $\\tau/\\tau_\\eta$ = 1, 2, 5, 100" % LABEL[ds]) if k == 0 else None)
        if tau in ref_blocks:
            ref = ref_blocks[tau]
            mr = ref[:, 1] > 0
            if k == 0:
                ax.semilogy(ref[mr, 0], ref[mr, 1], ":", color="0.4", lw=1.0, label="paper")
            else:
                ax.semilogy(ref[mr, 0], ref[mr, 1], ":", color="0.4", lw=1.0)
    ax.set_xlim(-40, 40)          # the axes of Fig. 1a of the paper
    ax.set_ylim(1e-13, 1e1)       # the axes of Fig. 1a of the paper
    ax.set_xlabel("$\\delta_\\tau V_i / \\sigma(\\delta_\\tau V_i)$")
    ax.set_ylabel("PDF$(\\delta_\\tau V_i)$")
    # two lines, like the DM-1c panel's title, so the two panels end up the same height
    ax.set_title("Fig.1a reproduction (%s solid, paper dotted;\nPDFs shifted by $10^{-1.3k}$)"
                 % LABEL[ds])
    h_, l_ = ax.get_legend_handles_labels()
    # identical canvas and margins to the DM-1c panel (48_mod) -> same panel size
    fig.subplots_adjust(left=0.155, right=0.975, top=0.905, bottom=0.175)
    fig.legend(h_, l_, loc="lower center", ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.055))
    p1 = os.path.join(FIGS, "A7_fig1a_%s%s.png" % (ds, args.out_suffix))
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print("saved", p1)



if __name__ == "__main__":
    main()
