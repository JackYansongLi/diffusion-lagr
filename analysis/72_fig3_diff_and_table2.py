"""72_fig3_diff_and_table2.py -- numbers behind the V3 revisions of the discussion document.

Three blocks, all computed from the arrays the figures are drawn from:

  (1) Table 2 of the paper (single-time second-order correlations) reproduced for
      DNS / DM-1c released samples / retrained DM-1c / DM-3c, with the relative
      differences against the published values.
  (2) C_tau inset of Fig. 3 (acceleration autocorrelation), DNS vs released DM-1c vs
      retrained DM-1c vs the paper's inset curve -- where and by how much they differ.
  (3) Fig. 3e / 3d: flatness and second-order structure function per denoising stage
      against the paper, restricted to the small-lag range.

Usage: run with the project venv on Windows paths (DATA/RESULTS are Windows paths).
Outputs results/fig3_v3_numbers.npz and a printed report.
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

# published Table 2 (Nature Machine Intelligence 6, 393-403, 2024)
PAPER_T2 = {
    "E":  {"DNS": 3.0, "DM-1c": 3.0, "DM-3c": 2.9},
    "A":  {"DNS": 1.7e-3, "DM-1c": 1.8e-3, "DM-3c": 1.6e-3},
    "SV": {"DNS": -0.41, "DM-1c": np.nan, "DM-3c": -0.39},
    "SA": {"DNS": 4.4e-5, "DM-1c": np.nan, "DM-3c": 2.4e-5},
}
DS = {"dns": ("DNS", "figstats_dns.npz"),
      "dm1c": ("DM-1c", "figstats_dm1c.npz"),
      "retrain": ("DM-1c re-trained", "figstats_dm1c_repro.npz"),
      "dm3c": ("DM-3c", "figstats_dm3c.npz")}


def parse_src(fn, blk):
    cur, rows = None, []
    for l in open(os.path.join(SRC, fn), errors="replace"):
        if l.startswith("#"):
            cur = l[1:].strip()
        elif l.strip() and cur is not None and cur.startswith(blk):
            rows.append([float(x) for x in l.split()])
    return np.array(rows)


def block_table2():
    print("=" * 78)
    print("[72] Table 2 of the paper, reproduced")
    print("  %-20s %10s %10s %10s %10s" % ("dataset", "E", "A", "SigmaV", "SigmaA"))
    out = {}
    for key, (name, fn) in DS.items():
        z = np.load(os.path.join(RESULTS, fn))
        E, A = float(np.mean(z["E"])), float(np.mean(z["A_comp"]))
        SV = float(z["SV"]) if "SV" in z.files else np.nan
        SA = float(z["SA"]) if "SA" in z.files else np.nan
        out[key] = (E, A, SV, SA)
        print("  %-20s %10.4g %10.4g %10.4g %10.4g" % (name, E, A, SV, SA))
    print("  published: DNS 3.0 / 1.7e-3 / -0.41 / 4.4e-5 ; DM-1c 3.0 / 1.8e-3 / - / - ;"
          " DM-3c 2.9 / 1.6e-3 / -0.39 / 2.4e-5")
    print("\n  relative difference to the published value")
    for key, paper_col in (("dns", "DNS"), ("dm1c", "DM-1c"), ("retrain", "DM-1c"), ("dm3c", "DM-3c")):
        E, A, SV, SA = out[key]
        row = []
        for q, val in (("E", E), ("A", A), ("SV", SV), ("SA", SA)):
            p = PAPER_T2[q][paper_col]
            row.append("%s %+6.1f%%" % (q, 100 * (val / p - 1)) if np.isfinite(p) else "%s  n/a" % q)
        print("  %-20s %s" % (DS[key][0], "  ".join(row)))
    return out


def block_ctau():
    print("=" * 78)
    print("[72] Fig. 3 inset: C_tau = <a_i(t+tau) a_i(t)> (same construction as the figure)")
    curves = {}
    for key, fn in (("dns", "figstats_dns.npz"), ("dm1c", "figstats_dm1c.npz"),
                    ("retrain", "figstats_dm1c_repro.npz")):
        z = np.load(os.path.join(RESULTS, fn))
        Cv = np.asarray(z["Cv"])
        if Cv.ndim == 2:                                  # (lag, component)
            Cv = Cv.mean(axis=1)
        Ca = 2 * Cv[1:-2] - Cv[:-3] - Cv[2:-1]          # Ca(lag 1 .. len-2)
        curves[key] = Ca
    lag = np.arange(1, len(curves["dns"]) + 1)
    tau_eta = lag / 10.0                                 # 10 storage steps = tau_eta
    pap = parse_src("42256_2024_810_MOESM5_ESM.txt", "Figure2_inset_DM-1c.out")
    print("  published inset curve: %d points, tau/tau_eta %.2f .. %.2f"
          % (len(pap), pap[:, 0].min(), pap[:, 0].max()))

    def at(arr, t):
        return float(np.interp(t, tau_eta, arr))

    print("  %-8s %10s %10s %10s %10s" % ("tau/eta", "DNS", "released", "retrained", "paper"))
    for t in (0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0):
        print("  %-8.1f %10.3e %10.3e %10.3e %10.3e" % (
            t, at(curves["dns"], t), at(curves["dm1c"], t), at(curves["retrain"], t),
            float(np.interp(t, pap[:, 0], pap[:, 1]))))

    def crossing(arr, level=0.0):
        s = np.sign(arr - level)
        idx = np.nonzero(np.diff(s) != 0)[0]
        return float(tau_eta[idx[0]]) if idx.size else np.nan

    print("  zero crossing (tau/tau_eta): DNS %.2f | released %.2f | retrained %.2f | paper %.2f"
          % (crossing(curves["dns"]), crossing(curves["dm1c"]), crossing(curves["retrain"]),
             crossing(pap[:, 1])))
    m = tau_eta <= 2.0
    for lbl, a in (("released", curves["dm1c"]), ("retrained", curves["retrain"])):
        r = a[m] / curves["dns"][m] - 1
        print("  %s vs DNS, tau/eta<=2: median %+6.1f%%  range %+6.1f..%+6.1f  (at tau/eta=0.3: %+.1f%%, 1.0: %+.1f%%)"
              % (lbl, 100 * np.median(r), 100 * r.min(), 100 * r.max(),
                 100 * (at(a, 0.3) / at(curves["dns"], 0.3) - 1),
                 100 * (at(a, 1.0) / at(curves["dns"], 1.0) - 1)))
    r = curves["retrain"][m] / curves["dm1c"][m] - 1
    print("  retrained vs released, tau/eta<=2: median %+6.1f%%  range %+6.1f..%+6.1f" %
          (100 * np.median(r), 100 * r.min(), 100 * r.max()))
    return curves, tau_eta


def block_stages():
    print("=" * 78)
    print("[72] Fig. 3e (flatness) and 3d (S2) per denoising stage, against the paper")
    z = np.load(os.path.join(RESULTS, "fig3e_f4_dm1c_repro.npz"))
    lags, order = z["lags"], [str(x) for x in z["order"]]
    S2b, S4b = z["S2b"], z["S4b"]
    tau_eta = lags / 10.0
    print("  flatness F (mean over the 10 trajectory batches), ratio to the paper's own stage curve")
    print("  %-8s %12s %12s %12s" % ("stage", "tau<=1", "1<tau<=5", "all"))
    for st in ("A", "B", "C", "D"):
        F = np.array([S4b[order.index("%s_%d" % (st, b))] / S2b[order.index("%s_%d" % (st, b))] ** 2
                      for b in range(10)])
        cur = F.mean(axis=0)
        pap = parse_src("42256_2024_810_MOESM6_ESM.txt", "Figure3e_%s.out" % st)
        pap_i = np.interp(tau_eta, pap[:, 0], pap[:, 1])
        out = []
        for m in (tau_eta <= 1.0, (tau_eta > 1.0) & (tau_eta <= 5.0), np.ones_like(tau_eta, bool)):
            r = cur[m] / pap_i[m] - 1
            out.append(100 * np.median(r))
        print("  %-8s %11.1f%% %11.1f%% %11.1f%%" % (st, *out))

    z2 = np.load(os.path.join(RESULTS, "fig3d_s2_dm1c_repro.npz"))
    lags2 = z2["lags"]
    pap2 = None
    t2 = lags2 / 10.0
    print("\n  S2, ratio to the published curve (released samples: figstats_dm1c vs Figure4a_DM-1c_p2)")
    print("  %-8s %12s %12s %12s" % ("stage", "tau<=1", "1<tau<=5", "all"))
    for st in ("A", "B", "C", "D"):
        cur = z2["%s_S2b" % st]
        pap2 = parse_src("42256_2024_810_MOESM6_ESM.txt", "Figure3d_%s.out" % st)
        pap_i = np.interp(t2, pap2[:, 0], pap2[:, 1])
        out = []
        for m in (t2 <= 1.0, (t2 > 1.0) & (t2 <= 5.0), np.ones_like(t2, bool)):
            r = cur[m] / pap_i[m] - 1
            out.append(100 * np.median(r))
        print("  %-8s %11.1f%% %11.1f%% %11.1f%%" % (st, *out))


def block_s2_curves():
    """S2 from the structure-function files, compared with the paper's own curve."""
    print("=" * 78)
    print("[72] S2 against the paper (Fig. 3d): released samples, retrained final sample, DNS")
    print("  %-24s %10s %12s %10s %10s" % ("curve", "tau<=1", "1<tau<=5", "all", "n_lags"))
    for lbl, fn, blk in (
            ("released DM-1c", "structfuncs_dm1c_full.npz", "Figure3d_A.out"),
            ("retrained DM-1c (A)", "structfuncs_dm1c_repro_full.npz", "Figure3d_A.out"),
            ("DNS", "structfuncs_dns_full.npz", "Figure3d_DNS.out")):
        z = np.load(os.path.join(RESULTS, fn))
        lag = np.asarray(z["lags"], float)
        S2 = np.asarray(z["S2"], float)[:len(lag)]
        t = lag / 10.0
        pap = parse_src("42256_2024_810_MOESM6_ESM.txt", blk)
        pi = np.interp(t, pap[:, 0], pap[:, 1])
        row = []
        for m in (t <= 1.0, (t > 1.0) & (t <= 5.0), np.ones_like(t, bool)):
            row.append(100 * np.median(S2[m] / pi[m] - 1))
        print("  %-24s %9.1f%% %11.1f%% %9.1f%% %10d" % (lbl, row[0], row[1], row[2], len(t)))


if __name__ == "__main__":
    t2 = block_table2()
    cl, te = block_ctau()
    block_stages()
    block_s2_curves()
    np.savez(os.path.join(RESULTS, "fig3_v3_numbers.npz"),
             table2_keys=np.array(list(t2)),
             table2_vals=np.array([t2[k] for k in t2]), ctau_dns=cl["dns"],
             ctau_dm1c=cl["dm1c"], ctau_retrain=cl["retrain"], tau_eta=te)
