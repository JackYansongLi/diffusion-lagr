"""73_fig4b_mean_diff.py -- mean flatness of the retrained DM-1c against the paper's DM-1c curve.

Feeds the V4 revision of the Figure 9 paragraph (annotation C10): the qualitative
statement "in the small-tau region the retrained model lies below the paper" is turned
into numbers taken from the plotted means, not from the rendering.

Inputs (all arrays the figure is drawn from):
  results/flatness_{dm1c,dm1c_repro}_full.npz   mean curves per order (p = 2,4,6,8)
  Data/Li_et_al_2024_source_data/42256_2024_810_MOESM7_ESM.txt
      Figure4b_DM-1c_p{4,6,8}.out (tau, mean, -, +)
Output: results/fig4b_mean_numbers.npz  + printed report.
"""
import os

import numpy as np

# Root of the analysis tree: the directory that holds results/ and Data/.
# Override with the LTS_ROOT environment variable; the default is the parent
# of this analysis/ directory.
ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SRC = os.path.join(ROOT, "Data", "Li_et_al_2024_source_data", "42256_2024_810_MOESM7_ESM.txt")


def paper(block):
    cur, rows = None, []
    for line in open(SRC, errors="replace"):
        if line.startswith("#"):
            cur = line[1:].strip()
        elif line.strip() and cur and cur.startswith(block):
            rows.append([float(x) for x in line.replace(",", " ").split()])
    return np.array(rows)


def ours(fn, order):
    z = np.load(os.path.join(RESULTS, fn))
    orders = [int(o) for o in z["orders"]]
    i2 = orders.index(2)
    i = orders.index(order)
    S2 = z["Sbc"][:, :, i2, 0]                       # (nb, nl)
    Sp = z["Sbc"][:, :, i, 0]
    F = Sp / S2 ** (order / 2.0)                      # per-batch F, then mean (figure convention)
    return z["lags"], F.mean(axis=0)


def bands(t, r):
    out = []
    for lo, hi in ((0, 0.1), (0.1, 1), (1, 3), (3, 10), (10, 30), (30, 1e9)):
        m = (t >= lo) & (t < hi) if lo else (t <= hi)
        out.append(100 * np.median(r[m] - 1) if m.any() else np.nan)
    return out


print("mean flatness, retrained DM-1c vs the paper's DM-1c curve")
print("  %-4s %9s %9s %9s %9s %9s %9s" % ("p", "<=0.1", "0.1-1", "1-3", "3-10", "10-30", ">30"))
store = {}
for p in (4, 6, 8):
    pap = paper("Figure4b_DM-1c_p%d.out" % p)
    tp, mp = pap[:, 0], pap[:, 1]
    lags, m = ours("flatness_dm1c_repro_full.npz", p)
    t = lags / 10.0
    mi = np.interp(t, tp, mp)
    r = m / mi
    print("  %-4d %8.1f%% %8.1f%% %8.1f%% %8.1f%% %8.1f%% %8.1f%%"
          % (p, *bands(t, r)))
    store["p%d_ratio" % p] = r
    store["tau"] = t
    # where does it cross 1?
    s = np.sign(r - 1.0)
    idx = np.nonzero(np.diff(s) != 0)[0]
    print("       first crossing at tau/tau_eta = %.1f | min ratio %.3f at tau=%.1f | at tau=0.1: %.1f%%"
          % (t[idx[0]] if idx.size else np.nan, r.min(), t[np.argmin(r)],
             100 * (r[np.argmin(np.abs(t - 0.1))] - 1)))

# the same comparison for the released samples, as the reference the paper plotted
print("\n  released samples vs the same paper curve (should be ~0 by construction)")
for p in (4, 6, 8):
    pap = paper("Figure4b_DM-1c_p%d.out" % p)
    lags, m = ours("flatness_dm1c_full.npz", p)
    t = lags / 10.0
    r = m / np.interp(t, pap[:, 0], pap[:, 1])
    print("  p=%d  %8.1f%% %8.1f%% %8.1f%% %8.1f%% %8.1f%% %8.1f%%"
          % (p, *bands(t, r)))
np.savez(os.path.join(RESULTS, "fig4b_mean_numbers.npz"), **store)
