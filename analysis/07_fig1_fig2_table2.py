# -*- coding: utf-8 -*-
"""
07_fig1_fig2_table2.py -- part A3-5: one-pass streaming accumulation of the statistics
behind Fig. 1a, Fig. 2 and Table 2 of Li et al. 2024 (Nat Mach Intell).

A single scan per dataset accumulates:
  HIST_INC  histograms of the normalized increment dtauVi/sigma(tau), lags {10,20,50,1000} (Fig. 1a, tau/tau_eta = 1,2,5,100)
  HIST_ACC  histograms of the normalized acceleration ai/sigma(ai), a = dv per storage step (forward difference, lag 1) (Fig. 2)
            Table 2 of the paper gives A = 1.7e-3, close to S2(lag 1) = 1.72e-3, which confirms this definition
  CV        velocity autocorrelation C_v(tau) = <vi(t) vi(t+tau)>, tau = 0..1025 (Fig. 2 inset:
            Ca(τ)=C(τ+1)-2C(τ)+C(τ-1), Ca(0)=S2(1))
  E_i       single-time <vi^2> (Table 2, E)
  VV_ij     ⟨vi²vj²⟩ ij=xy,xz,yz (Table 2 ΣV)
  AA_ij     ⟨ai²aj²⟩ (Table 2 ΣA)

sigma(tau) is read from S2 of the existing results/structfuncs_<ds>_full.npz, which saves a second pass.
DM-1c-10%, Fig. 3 and Fig. 6 need model weights or training and are outside this script.

Usage: python 07_fig1_fig2_table2.py --dataset dns  [--n-traj N] [--offset 0] [--tag xxx]
Output: results/figstats_<tag>.npz
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

# Root of the analysis tree: the directory that holds results/ and Data/.
# Override with the LTS_ROOT environment variable; the default is the parent
# of this analysis/ directory.
ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

PDF_LAGS = np.array([10, 20, 50, 1000])       # τ/τη = 1, 2, 5, 100
ACC_LAG = 1
CV_LAGS = np.arange(0, 1026)                   # 0..1025 (Ca needs tau+1)
INC_EDGES = np.linspace(-40.0, 40.0, 4001)     # d/sigma(tau), +-40 sigma, width 0.02 sigma
ACC_EDGES = np.linspace(-120.0, 120.0, 4801)   # a/sigma(a), +-120 sigma (the paper's tail runs beyond 60 sigma)
BLOCK = 512                                    # 32 GB with no pagefile leaves little commit headroom, so the blocks stay small (10 workers x ~60 MB)


def get_dataset(name):
    if name == "dns":
        import h5py
        p = os.path.join(DATA, "Lagr_u3c_diffusion.h5")
        with h5py.File(p, "r") as f:
            mn = f["min"][()]
            mx = f["max"][()]
            n = f["train"].shape[0]
        return dict(kind="h5", path=p, min=mn, max=mx, n_traj=n, C=3)
    if name == "dm3c":
        import h5py
        p = os.path.join(DATA, "tmp", "dm3c_358400x2000x3_f32.npy")
        with h5py.File(os.path.join(DATA, "Lagr_u3c_diffusion.h5"), "r") as f:
            mn = f["min"][()]
            mx = f["max"][()]
        return dict(kind="npy", path=p, min=mn, max=mx, n_traj=358400, C=3)
    if name == "dm1c":
        import h5py
        p = os.path.join(DATA, "tmp", "dm1c_358400x2000x1_f32.npy")
        with h5py.File(os.path.join(DATA, "Lagr_u1c_diffusion.h5"), "r") as f:
            mn = float(f["min"][()])
            mx = float(f["max"][()])
        return dict(kind="npy", path=p, min=mn, max=mx, n_traj=358400, C=1)
    if name == "dm1c_repro":
        # Our own DM-1c sampling (self-trained DM-1c, ema_0.9999_250000.pt, 28 runs x 12800).
        # Same u1c physical domain as the official dm1c -> reuse its min/max scalars.
        p = os.path.join(DATA, "tmp", "dm1c_repro_358400x2000x1_f32.npy")
        try:
            import h5py
            with h5py.File(os.path.join(DATA, "Lagr_u1c_diffusion.h5"), "r") as f:
                mn = float(f["min"][()])
                mx = float(f["max"][()])
        except ImportError:
            with np.load(os.path.join(RESULTS, "structfuncs_dm1c_full.npz")) as z:
                mn = float(np.asarray(z["min"]).ravel()[0])
                mx = float(np.asarray(z["max"]).ravel()[0])
        return dict(kind="npy", path=p, min=mn, max=mx, n_traj=358400, C=1)
    raise ValueError(name)


def load_sigma2(ds, tag):
    """Read S2(lag) (component average) from the existing structfuncs npz -> sigma^2 per lag."""
    p = os.path.join(RESULTS, "structfuncs_%s_full.npz" % tag)
    z = np.load(p, allow_pickle=True)
    return z["S2"].astype(np.float64)          # len 1024, lag 1..1024


def worker(task):
    ds, i0, i1, s2 = task
    mn = np.asarray(ds["min"], dtype=np.float32).reshape(1, 1, -1)
    mx = np.asarray(ds["max"], dtype=np.float32).reshape(1, 1, -1)
    scale = (mx - mn) / 2.0
    if ds["kind"] == "h5":
        import h5py
        with h5py.File(ds["path"], "r") as f:
            u = f["train"][i0:i1].astype(np.float32)
    else:
        a = np.load(ds["path"], mmap_mode="r")
        u = np.asarray(a[i0:i1], dtype=np.float32)
    v = (u + 1.0) * scale + mn
    del u
    B, K, C = v.shape
    # in-place f32 arithmetic is promoted to f64 by float64 scalars, so it is pinned to f32
    v = v.astype(np.float32, copy=False)
    n_inc = len(INC_EDGES) - 1
    n_acc = len(ACC_EDGES) - 1
    n_pdf = len(PDF_LAGS)
    n_cv = len(CV_LAGS)

    hist_inc = np.zeros((n_pdf, n_inc), dtype=np.float64)
    hist_acc = np.zeros(n_acc, dtype=np.float64)
    cv = np.zeros((n_cv, C), dtype=np.float64)     # Σ v_i(t) v_i(t+τ)
    E = np.zeros(C, dtype=np.float64)              # sum v_i^2 (single time, averaged over both ends)
    vv = np.zeros((3,), dtype=np.float64) if C == 3 else None   # Σ vi²vj²
    aa = np.zeros((3,), dtype=np.float64) if C == 3 else None   # Σ ai²aj²
    a2sum = np.zeros(C, dtype=np.float64)          # sum ai^2 (per component, physical units)
    cnt_cv = np.zeros((n_cv,), dtype=np.float64)
    cnt_e = 0.0
    cnt_a = 0.0                                    # number of acceleration samples (K-1 per trajectory)

    # --- single-time quantities: E, SigmaV (both ends averaged, to use more data) ---
    E += np.sum(v * v, dtype=np.float64, axis=(0, 1))
    cnt_e += B * K
    if C == 3:
        x2 = v[:, :, 0] ** 2
        y2 = v[:, :, 1] ** 2
        z2 = v[:, :, 2] ** 2
        # paper SigmaV: <vi^2 vj^2> - <vi^2><vj^2>; vi^2 and vj^2 are different components, so the product is elementwise and then averaged
        vv[0] += np.sum(x2 * y2, dtype=np.float64)
        vv[1] += np.sum(x2 * z2, dtype=np.float64)
        vv[2] += np.sum(y2 * z2, dtype=np.float64)
        del x2, y2, z2

    # --- velocity autocorrelation C_v(tau), tau = 0..1025 ---
    for j, tau in enumerate(CV_LAGS):
        if tau == 0:
            cv[j] += np.sum(v * v, dtype=np.float64, axis=(0, 1))
            cnt_cv[j] += B * K
        else:
            cv[j] += np.sum(v[:, tau:, :] * v[:, :K - tau, :], dtype=np.float64, axis=(0, 1))
            cnt_cv[j] += B * (K - tau)

    # --- acceleration (lag-1 increment) moments and histograms; every moment is taken in PHYSICAL units, then normalized for the histograms ---
    d1 = v[:, 1:, :] - v[:, :K - 1, :]             # a = increment over one storage step (physical units)
    a2sum += np.sum(d1 ** 2, dtype=np.float64, axis=(0, 1))    # Σ ai² per comp
    cnt_a += B * (K - 1)
    if C == 3:
        ax2 = d1[:, :, 0] ** 2
        ay2 = d1[:, :, 1] ** 2
        az2 = d1[:, :, 2] ** 2
        # physical units: <ai^2 aj^2>; computing it after normalization would give a dimensionless number, which is not the Table 2 definition
        aa[0] += np.sum(ax2 * ay2, dtype=np.float64)
        aa[1] += np.sum(ax2 * az2, dtype=np.float64)
        aa[2] += np.sum(ay2 * az2, dtype=np.float64)
        del ax2, ay2, az2
    s_a = np.sqrt(s2[0])                           # sigma(a) = sqrt(S2(lag 1)), component average
    d1 *= np.float32(1.0 / s_a)                    # in-place normalization (must come after aa)
    hist_acc += np.histogram(d1, bins=ACC_EDGES)[0]
    del d1

    # --- Fig. 1a increment histograms, lags {10,20,50,1000} ---
    for k, tau in enumerate(PDF_LAGS):
        d = v[:, tau:, :] - v[:, :K - tau, :]
        s = np.sqrt(s2[tau - 1])
        d *= np.float32(1.0 / s)                   # in-place normalization, saves one copy of 2x the data
        hist_inc[k] += np.histogram(d, bins=INC_EDGES)[0]
        del d

    return hist_inc, hist_acc, cv, cnt_cv, E, cnt_e, vv, aa, a2sum, cnt_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["dns", "dm3c", "dm1c", "dm1c_repro"])
    ap.add_argument("--n-traj", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    ds = get_dataset(args.dataset)
    n = ds["n_traj"] if args.n_traj == 0 else min(args.n_traj, ds["n_traj"] - args.offset)
    tag = args.tag or args.dataset
    s2 = load_sigma2(ds, args.dataset)             # structfuncs_<dataset>_full

    tasks = [(ds, i0, min(i0 + BLOCK, args.offset + n), s2)
             for i0 in range(args.offset, args.offset + n, BLOCK)]
    print(f"[{tag}] dataset={args.dataset} n_traj={n} blocks={len(tasks)} "
          f"workers={args.workers} C={ds['C']}", flush=True)

    C = ds["C"]
    n_inc = len(INC_EDGES) - 1
    n_acc = len(ACC_EDGES) - 1
    HIST_INC = np.zeros((len(PDF_LAGS), n_inc))
    HIST_ACC = np.zeros(n_acc)
    CV = np.zeros((len(CV_LAGS), C))
    CNT_CV = np.zeros(len(CV_LAGS))
    E = np.zeros(C)
    CNT_E = 0.0
    VV = np.zeros(3) if C == 3 else None
    AA = np.zeros(3) if C == 3 else None
    A2 = np.zeros(C)
    CNT_A = 0.0

    t0 = time.time()
    done = 0
    with Pool(args.workers) as pool:
        for hi, ha, cv, ccv, e, ce, vv, aa, a2s, ca in pool.imap_unordered(worker, tasks, chunksize=1):
            HIST_INC += hi
            HIST_ACC += ha
            CV += cv
            CNT_CV += ccv
            E += e
            CNT_E += ce
            A2 += a2s
            CNT_A += ca
            if C == 3:
                VV += vv
                AA += aa
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} blocks ({time.time()-t0:.0f}s)", flush=True)

    Cv = CV / CNT_CV[:, None]                      # ⟨vi(t)vi(t+τ)⟩ per comp
    E_mean = E / CNT_E                             # ⟨vi²⟩ per comp
    # <ai^2> per component: second moment of the lag-1 increment taken directly (physical units).
    # 2(E - Cv(1)) must not be used: E and Cv(1) are both about 3.0 and differ by ~1e-3, so catastrophic cancellation in f32 would bias the result by more than 10%.
    A_comp = A2 / CNT_A

    # Table 2: SigmaV / SigmaA (physical units, averaged over the three pairs xy/xz/yz)
    SV = SA = None
    SVp = SAp = None
    if C == 3:
        pair_E = np.array([E_mean[0] * E_mean[1], E_mean[0] * E_mean[2], E_mean[1] * E_mean[2]])
        pair_A = np.array([A_comp[0] * A_comp[1], A_comp[0] * A_comp[2], A_comp[1] * A_comp[2]])
        SVp = VV / CNT_E - pair_E                  # VV is already summed per pair, so the denominator is the sample count (do not divide by 3 again)
        SAp = AA / CNT_A - pair_A
        SV = float(SVp.mean())
        SA = float(SAp.mean())

    out = os.path.join(RESULTS, "figstats_%s.npz" % tag)
    np.savez_compressed(
        out, pdf_lags=PDF_LAGS, inc_edges=INC_EDGES, acc_edges=ACC_EDGES,
        hist_inc=HIST_INC, hist_acc=HIST_ACC, cv_lags=CV_LAGS, Cv=Cv,
        E=E_mean, A_comp=A_comp,
        VV=VV if C == 3 else np.zeros(1), AA=AA if C == 3 else np.zeros(1),
        SV=SV if SV is not None else np.nan, SA=SA if SA is not None else np.nan,
        SV_pairs=SVp if SVp is not None else np.zeros(1),
        SA_pairs=SAp if SAp is not None else np.zeros(1),
        n_traj=n, C=C)
    print("saved", out, "(%.0fs)" % (time.time() - t0), flush=True)

    # Table 2 sanity check (physical units, component average)
    print("E/3 =", round(float(E_mean.mean()), 4), " (paper DNS: 3.0)")
    print("A   =", round(float(A_comp.mean()), 8), " (paper DNS: 1.7e-3)")
    if C == 3:
        print("SigmaV = %.4f  (per pair %s)  paper DNS -0.41 / DM-3c -0.39"
              % (SV, " ".join("%.4f" % x for x in SVp)))
        print("SigmaA = %.4e  (per pair %s)  paper DNS 4.4e-5 / DM-3c 2.4e-5"
              % (SA, " ".join("%.2e" % x for x in SAp)))


if __name__ == "__main__":
    main()
