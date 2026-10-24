"""71_fig2_accpdf_diff_data.py -- quantify the DM-1c acceleration-PDF comparison of Fig.3
(paper Fig.2) from the same arrays that the figure is drawn from.

The figure rebins our fine acceleration histograms onto the paper's own coarse grid and
keeps the contiguous non-empty run around the peak (`trim_ends`).  Any statement about
"where our curve stops" or "how much lower the retrained curve is in the tail" must be
computed on exactly that grid, not read off the rendered panel.

Outputs (printed + results/fig2_accpdf_diff.npz):
  * plotted range and sample count of each of the three curves;
  * probability mass in |a| bands, for each curve;
  * relative deviation of our released samples and of the retrained model against the
    paper curve, and of the retrained model against the released samples, per band;
  * total-variation distance over the common range.
"""
import os
import numpy as np

# Root of the analysis tree: the directory that holds results/ and Data/.
# Override with the LTS_ROOT environment variable; the default is the parent
# of this analysis/ directory.
ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SRC = os.path.join(ROOT, "Data", "Li_et_al_2024_source_data")
NB = "42256_2024_810_MOESM5_ESM.txt"


def parse_src(fn, blk):
    cur, rows = None, []
    for l in open(os.path.join(SRC, fn), errors="replace"):
        if l.startswith("#"):
            cur = l[1:].strip()
        elif l.strip() and cur == blk:
            rows.append([float(x) for x in l.split()])
    return np.array(rows)


def trim_ends(xc, hc, min_count=1):
    pk = int(np.argmax(hc))
    lo = pk
    while lo > 0 and hc[lo - 1] > 0:
        lo -= 1
    hi = pk
    while hi < len(hc) - 1 and hc[hi + 1] > 0:
        hi += 1
    keep = np.nonzero(hc[lo:hi + 1] >= min_count)[0]
    return xc[lo:hi + 1][keep], hc[lo:hi + 1][keep]


def rebin(z, xp, lo, hi):
    """our fine histogram -> the paper's coarse grid (counts, then pdf with the paper's Wp)."""
    ea = z["acc_edges"]
    ca = 0.5 * (ea[:-1] + ea[1:])
    Ha = z["hist_acc"]
    Wp = float(np.median(np.diff(xp)))
    j = np.searchsorted(lo, ca, side="right") - 1
    v = (j >= 0) & (j < len(xp)) & (ca >= lo[0]) & (ca < hi[-1])
    cnt = np.bincount(j[v], weights=Ha[v], minlength=len(xp))
    return cnt / (Ha.sum() * Wp), float(Ha.sum())


def band_stats(a, f_num, f_den, label):
    out = []
    for b in (1.0, 3.0, 5.0, 8.0, 10.0, 12.0):
        m = np.abs(a) >= b
        if m.sum() < 3:
            continue
        r = f_num[m] / f_den[m] - 1.0
        out.append((b, int(m.sum()), 100 * np.median(r), 100 * r.min(), 100 * r.max()))
        print("  %-22s |a|>=%4.1f  n=%3d  median %+6.1f%%  range %+6.1f%%..%+6.1f%%"
              % (label, b, m.sum(), 100 * np.median(r), 100 * r.min(), 100 * r.max()))
    return out


def main():
    ref = parse_src(NB, "Figure2_DM-1c.out")
    xp = ref[:, 0]
    Wp = float(np.median(np.diff(xp)))
    lo, hi = xp - Wp / 2.0, xp + Wp / 2.0

    rel = np.load(os.path.join(RESULTS, "figstats_dm1c.npz"))
    ret = np.load(os.path.join(RESULTS, "figstats_dm1c_repro.npz"))

    p_rel, n_rel = rebin(rel, xp, lo, hi)
    p_ret, n_ret = rebin(ret, xp, lo, hi)

    # the figure keeps the contiguous non-empty run of the released samples around the peak
    xk, _ = trim_ends(xp, np.round(p_rel * n_rel * Wp))
    keep = np.isin(xp, xk)
    ok_rel = keep & (p_rel > 0)
    ok_ret = keep & (p_ret > 0)

    print("[71] coarse grid: %d points, dx = %.3f sigma (paper's bin width)" % (len(xp), Wp))
    for name, ok, n in (("released samples", ok_rel, n_rel), ("retrained", ok_ret, n_ret)):
        print("  %-18s N=%8d  plotted a/sigma = %6.1f .. %6.1f  (%d points)"
              % (name, n, xp[ok].min(), xp[ok].max(), ok.sum()))
    mref = ref[:, 1] > 0
    print("  %-18s                 |a|/sigma = %6.1f .. %6.1f  (%d points)"
          % ("paper curve", ref[mref, 0].min(), ref[mref, 0].max(), int(mref.sum())))

    # probability mass in bands, each curve normalised to unit total probability
    print("\n[71] probability mass in |a|/sigma bands")
    for name, p in (("released", p_rel), ("retrained", p_ret)):
        row = ["%s: total %.3f" % (name, p.sum() * Wp)]
        for b in (1.0, 3.0, 5.0, 8.0):
            row.append("|a|>=%g: %.2e" % (b, p[np.abs(xp) >= b].sum() * Wp))
        print("  " + " | ".join(row))
    mp = ref[:, 1]
    print("  paper:    total %.3f | " % (mp[mref].sum() * Wp) + " | ".join(
        "|a|>=%g: %.2e" % (b, mp[mref][np.abs(ref[mref, 0]) >= b].sum() * Wp) for b in (1.0, 3.0, 5.0, 8.0)))

    # relative deviations, band by band
    print("\n[71] released vs paper (same grid, common points)")
    cm = ok_rel & mref
    band_stats(xp[cm], p_rel[cm], ref[cm, 1], "released/paper")
    print("\n[71] retrained vs released")
    cm2 = ok_rel & ok_ret
    band_stats(xp[cm2], p_ret[cm2], p_rel[cm2], "retrained/released")
    print("\n[71] retrained vs paper")
    cm3 = ok_ret & mref
    band_stats(xp[cm3], p_ret[cm3], ref[cm3, 1], "retrained/paper")

    # mass ratios in bands (robust; per-bin medians are dominated by empty far-tail bins)
    print("\n[71] probability-mass ratios in |a|/sigma bands")
    for b in (1.0, 3.0, 5.0, 8.0):
        m = np.abs(xp) >= b
        mr_, mt_, mp_ = (p_rel[m].sum() * Wp, p_ret[m].sum() * Wp, ref[m, 1].sum() * Wp)
        print("  |a|>=%4.1f  released %.3e  retrained %.3e  paper %.3e | ret/rel %+6.1f%%  "
              "rel/paper %+6.1f%%  ret/paper %+6.1f%%"
              % (b, mr_, mt_, mp_, 100 * (mt_ / mr_ - 1), 100 * (mr_ / mp_ - 1), 100 * (mt_ / mp_ - 1)))

    # total-variation distance on ONE common mask, so the three pairs are comparable
    print("\n[71] total-variation distance (half the L1 norm) on the common mask")
    mcom = ok_rel & ok_ret & mref
    for lbl, p1, p2 in (("released vs paper", p_rel, ref[:, 1]),
                        ("retrained vs released", p_ret, p_rel),
                        ("retrained vs paper", p_ret, ref[:, 1])):
        print("  %-22s %.4f  (%d points)" % (lbl, 0.5 * np.abs(p1[mcom] - p2[mcom]).sum() * Wp, int(mcom.sum())))

    # per-bin spread where both curves are well resolved
    print("\n[71] per-bin relative deviation, |a|/sigma <= 20")
    for lbl, p1, p2 in (("released vs paper", p_rel, ref[:, 1]),
                        ("retrained vs released", p_ret, p_rel)):
        m = mcom & (np.abs(xp) <= 20.0)
        r = p1[m] / p2[m] - 1.0
        print("  %-22s n=%3d  median %+5.1f%%  mean %+5.1f%%  max|.| %5.1f%%"
              % (lbl, m.sum(), 100 * np.median(r), 100 * r.mean(), 100 * np.abs(r).max()))

    np.savez(os.path.join(RESULTS, "fig2_accpdf_diff.npz"),
             xp=xp, Wp=Wp, keep=keep, p_released=p_rel, p_retrained=p_ret, p_paper=ref[:, 1])


if __name__ == "__main__":
    main()
