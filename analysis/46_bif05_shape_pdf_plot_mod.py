"""46_bif05_shape_pdf_plot_mod.py -- FIGURE stage, MODIFIED version  (figure B04_mod).

Differences vs the initial 46_bif05_shape_pdf_plot.py (which stays frozen):
  * LINEAR y axis by default -- Biferale et al. (2005) Fig. 3 and Van Aartrijk et al. (2008) Fig. 14 both
    draw the shape-index pdfs on linear axes (VA08's measured ticks: y 0..16 and 0..35, x 0.3..1.1 for the
    I1 panel), so the initial log-y choice was a deviation that is now removed.
  * per-panel y limits (`--ymax auto` = 1.08 x the largest drawn value), because the three pdfs differ by
    orders of magnitude in peak height; `--ymax a,b,c` sets them explicitly.
  * I3 panel gets a wider x range (default 0.30) and the I3 histograms come from the *_mod* npz, which bins
    I3 on [0, 0.36] with the same 0.005 bin width as I1/I2 -- the initial npz stopped at 0.2 and clipped the
    rest into its last bin while the data actually reach I3 = 0.297.
  * `--binw-i3` coarsens the stored I3 bins by an integer factor (density magnitude scales as 1/binwidth,
    so a coarser binning is what makes a LINEAR-y I3 panel readable).
  * a clipping self-check is printed for every drawn layer.

Usage: 46_bif05_shape_pdf_plot_mod.py [--times ...] [--sel] [--contrast] [--yscale linear] [--out NAME]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tetrad_lib as tl

SUB = ["1", "2", "3"]


def rebin(h, k):
    """Coarsen a (possibly multi-dimensional) uniform histogram by an integer factor (density-preserving)."""
    if k == 1:
        return h, 1.0
    n = h.shape[-1]
    if n % k:
        raise ValueError(f"bin count {n} is not divisible by {k}")
    return h.reshape(*h.shape[:-1], n // k, k).mean(axis=-1), float(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(tl.RESULTS, "bif05_shape_pdf_mod.npz"))
    ap.add_argument("--out", default="B04_bif05_shape_pdf_mod.png")
    ap.add_argument("--times", default="0.10,2.05,15.30,35.00,63.00")
    ap.add_argument("--sel", action="store_true")
    ap.add_argument("--contrast", action="store_true")
    ap.add_argument("--ref", choices=["gauss", "none"], default="gauss")
    # per-panel x windows (I1, I2, I3).  I1's support is [1/3, 1] and the paper (VA08 Fig.14) draws it as
    # 0.3-1.1 -- keep xmax(I1) = 1.0: at 0.95 more than HALF the mass of the most distorted time would fall
    # outside (measured P(I1 > 0.95) = 53.4% at t = 15.3 tau_eta, 34.7% at 35, 19.3% at 63).
    ap.add_argument("--xmax", default="1.1,0.55,0.30")             # per panel (I1, I2, I3)
    # "linear" | "log" | per-panel "linear,linear,log".  B05's own shape-index figure uses a LINEAR y for the
    # P(I2) main panel but a LOGARITHMIC y for the P(I3) inset (read off the rendered page: its inset y ticks
    # are 0.01 / 1 / 100) -- so the per-panel form is the faithful way to mirror the paper when asked.
    ap.add_argument("--yscale", default="linear")
    ap.add_argument("--ymax", default="auto")                      # auto | a,b,c
    # per panel, "auto" = 0 (linear) / data-driven floor (log).  B05's own P(I3) inset is drawn on
    # y in [0.01, 100]; the bins below density 0.01 carry only 0.002-0.037% of the mass and are the
    # low-count (0-2 samples/bin) sawtooth region, so that window is the honest way to clean the tail.
    ap.add_argument("--ymin", default="auto,auto,auto")
    # VA08 Fig.14 draws the I1 panel over 0.3-1.1 (its tick labels: 0.3 ... 1.1); I1 >= 1/3 is a hard bound,
    # so the 0.3 left edge costs nothing (audited: mass outside = 0.000%) and buys horizontal resolution.
    ap.add_argument("--xmin", default="0.3,0,0")                   # per panel (I1, I2, I3)
    # I3 default 0.02 (= 4 x the stored 0.005 grid): the density peak of a cuspy pdf scales as ~1/binwidth,
    # and I3 concentrates at small values, so at 0.005 the linear-y I3 panel is one spike of density ~170
    # that flattens every other curve; 0.02 brings the peaks to 21-48, i.e. comparable with I1/I2 (3-20).
    # Measured peaks (t = 0.10/2.05/15.3/35/63 tau_eta): w=0.005 -> 35/33/171/138/106, w=0.02 -> 30/22/48/44/40.
    ap.add_argument("--binw-i3", type=float, default=0.02)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--figsize", default="10.6,4.3")
    # "below" (default) puts ONE shared legend under the three panels, clear of all curves; any matplotlib
    # loc string (e.g. "upper left") puts it back inside the I1 axes.
    ap.add_argument("--legend", default="below")
    ap.add_argument("--title", default="Shape-index PDFs -- TURB-Lagr tetrads")
    a = ap.parse_args()

    want = [float(x) for x in a.times.split(",")]
    xmax = [float(x) for x in a.xmax.split(",")]
    xmin = [float(x) for x in a.xmin.split(",")]
    def per3(s):
        """Accept "auto" / a single value / a 3-entry list."""
        v = str(s).split(",")
        return v * 3 if len(v) == 1 else v
    ymx, ymn = per3(a.ymax), per3(a.ymin)
    d = np.load(a.inp)

    def get3(base):
        """Per-panel arrays: the _mod npz stores hist1..3/edges1..3 (different bin counts)."""
        if f"{base}1" in d.files:
            return [d[f"{base}{c+1}"] for c in range(3)]
        return list(d[base])

    times, edges, binw = d["times_tau"], get3("edges"), float(d["binw"])
    ok = np.array([np.abs(times - w).argmin() for w in want])
    bad = [f"{w} (closest {times[i]:.2f})" for w, i in zip(want, ok) if abs(times[i] - w) > 0.02]

    ysc = a.yscale.split(",")
    if len(ysc) == 1:
        ysc = ysc * 3
    assert len(ysc) == 3 and all(v in ("linear", "log") for v in ysc), a.yscale
    k = int(round(a.binw_i3 / binw))
    k = max(k, 1)
    hist, hist_sel, hist_ref = get3("hist"), get3("hist_sel"), get3("hist_ref")

    plt = tl.setup_style()
    fig, axes = plt.subplots(1, 3, figsize=tuple(float(x) for x in a.figsize.split(",")), sharey=False)
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(ok)))
    ylims = np.zeros((3, 2))
    report = []
    for c, ax in enumerate(axes):
        e = edges[c]
        kk = k if c == 2 else 1
        cc, _ = rebin(0.5 * (e[:-1] + e[1:]), kk)
        drawn = []
        esc = []          # probability mass left outside the drawn x window, per curve
        for col, i in zip(cols, ok):
            # per-panel lists now (hist[c] is that panel's (n_times, n_bins) array)
            h, _ = rebin(hist[c][i] if not a.sel else hist_sel[c][i], kk)
            ax.plot(cc, h, color=col, lw=1.6, label=rf"$t = {times[i]:.2f}\,\tau_\eta$")
            drawn.append(h)
            esc.append(float((h[(cc < xmin[c]) | (cc > xmax[c])] * (binw * kk)).sum()))
        if a.contrast and a.sel:
            for i in ok:
                h, _ = rebin(hist[c][i], kk)
                ax.plot(cc, h, color="0.45", lw=0.8, ls="--", alpha=0.9,
                        label="unconditional (same times)" if i == ok[0] else None)
                esc.append(float((h[(cc < xmin[c]) | (cc > xmax[c])] * (binw * kk)).sum()))
        if a.ref == "gauss":
            hr, _ = rebin(hist_ref[c], kk)
            ax.plot(cc, hr, color="0.15", lw=1.5, ls=":",
                    label="isotropic Gaussian" if c == 0 else None)
            drawn.append(hr)
            esc.append(float((hr[(cc < xmin[c]) | (cc > xmax[c])] * (binw * kk)).sum()))
        top = max(float(np.nanmax(x)) for x in drawn)
        ymax_c = ymx[c].strip()
        ymax = 1.08 * top if ymax_c == "auto" else float(ymax_c)
        ymin_c = ymn[c].strip()
        if ymin_c != "auto":
            lo = float(ymin_c)
        elif ysc[c] == "linear":
            lo = 0.0
        else:
            # log panel, auto floor: just under the smallest non-zero drawn density, so the decades that
            # actually carry signal are visible (zero-count bins are left out by matplotlib on a log axis).
            pos = [float(x[x > 0].min()) for x in drawn if np.any(x > 0)]
            lo = max(0.8 * min(pos), 1e-6) if pos else 1e-6
        ylims[c] = (lo, ymax)
        ax.set_xlim(xmin[c], xmax[c])
        ax.set_yscale(ysc[c])
        ax.set_ylim(lo, ymax)
        ax.set_xlabel(rf"$I_{SUB[c]}$")
        ax.set_title(rf"$I_{SUB[c]}$")
        # clipping self-check: every drawn layer against the view
        outside = sum(int(np.sum((x > ymax) | ((x > 0) & (x < lo)))) for x in drawn)
        ymass = max(float((x[(x < lo) | (x > ymax)] * (binw * kk)).sum()) for x in drawn) * 100
        report.append((c, top, ymax, lo, outside, ymass, 100 * max(esc) if esc else 0.0, xmin[c], xmax[c], f"{ysc[c]}-y; "
                   + ("I3 bins %.4g" % (binw * kk) if c == 2 else "bins %.4g" % binw)))
    axes[0].set_ylabel(r"$P(I_i)$")
    if a.legend == "below":
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4,
                   fontsize=7.8, frameon=False)
        rect = (0, 0.135, 1, 0.93)
    else:
        axes[0].legend(fontsize=7.2, loc=a.legend, ncol=2)
        rect = (0, 0, 1, 0.93)
    note = "B05-selected" if a.sel else "all tetrads"
    # pass fractions as percentages with one decimal: the old array2string printed bare "0." for exact
    # zeros, which read as a display glitch (reported by the user) instead of "0.0 %".
    fig.suptitle(a.title + f"   ({note}; {','.join(ysc)} y; bin width {binw:g}"
                 + (f", I3 {binw*k:g}" if k > 1 else "") + "; pass "
                 + "/".join(f"{100*v:.1f}" for v in d["pass_frac"][ok]) + " %)", fontsize=10)
    fig.tight_layout(rect=rect)
    tl.savefig(fig, a.out, dpi=a.dpi)

    print(f"source npz: {a.inp}  calibre={note}  yscale={a.yscale}  times={[f'{times[i]:.2f}' for i in ok]}")
    print("  panel  max drawn  ymax     ymin      pts>ywin  mass>ywin   mass out of x window   window")
    for c, top, ymax, lo, outside, ymass, mout, xlo, xhi, note2 in report:
        print(f"    I{c+1}   {top:9.3f}  {ymax:8.3f} {lo:9.5f}  {outside:8d}  {ymass:7.3f}%      {mout:8.3f}%          "
              f"[{xlo:g}, {xhi:g}]   ({note2})")
    print("  'mass>ywin' = probability mass of a drawn curve outside the y window (only via an explicit --ymin/--ymax)")
    print("  'mass out of x window' = largest probability mass of any drawn curve outside [xmin, xmax]")
    if bad:
        print("  WARNING: requested times not in npz:", bad)
    for i in ok:
        print(f"  t = {times[i]:7.2f} tau_eta: <I> = {np.array2string(d['stats'][i,:,0], precision=4)}"
              f"  pass = {100*d['pass_frac'][i]:5.1f}%")


if __name__ == "__main__":
    main()
