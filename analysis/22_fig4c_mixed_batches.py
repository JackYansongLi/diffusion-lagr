#!/usr/bin/env python3
"""
22_fig4c_mixed_batches.py -- statistics of the **mixed-component fourth-order flatness** of
Fig. 4c (the DNS and DM-3c).

Definition in the paper:
    F(4,ij)_tau = <(dtauVi)^2 (dtauVj)^2> / [S(2)_tau]^2, averaged over ij = xy, xz, yz.
    the denominator is **one** S(2) (isotropy, a common denominator), not the per-component product
    S2_i*S2_j.
    Checked on 500 trajectories: the two forms differ by less than 0.5%, as isotropy requires (see
    RESULTS_ANALYSIS section 3.13).

Bar conventions (the same lineage as Figs. 4a/4b; both are computed and the final choice is made
against the paper):
    convention A (10 blocks): the dataset is cut into 10 independent blocks, the three component
    pairs are averaged inside each block, and the ten resulting curves give the min/max;
    convention B (30 component blocks): 10 blocks x 3 component pairs = 30 curves -> min/max (verified
    to be what the paper uses, Figs. 4a/4b).
    Caption of the paper: "ten different independent batches of smaller size".

Output: results/fig4c_mixed_<ds>.npz
    lags(1024), tau(1024), F_mean(1024), X_mean(1024,3), S2_mean(1024),
    Fb(10,1024) per block with the pairs averaged, Fb_pairs(10,1024,3) per block and component pair,
    A_lo/hi(1024) the min/max of convention A, B_lo/hi(1024) that of convention B,
    n_traj, n_batch, batch_size, dataset, pairs
Usage: python3 22_fig4c_mixed_batches.py --dataset dns|dm3c [--batches 10] [--workers 8]
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

LAGS = np.arange(1, 1025)          # stored step; tau/tau_eta = 0.1*lag (the first column of the source data of the paper)
BLOCK = 2048                       # trajectories per worker block (about 49 MB each, f32 x 2000 x 3)
PAIRS = ((0, 1), (0, 2), (1, 2))   # xy, xz, yz


def _denorm_params(name):
    """The min/max used to de-normalise: read from the HDF5 attributes when possible, otherwise from
    structfuncs_<name>_full.npz."""
    try:
        import h5py
    except ImportError:
        z = np.load(os.path.join(RESULTS, "structfuncs_%s_full.npz" % name))
        return z["min"], z["max"]
    fn = "Lagr_u3c_diffusion.h5" if name in ("dns", "dm3c") else "Lagr_u1c_diffusion.h5"
    import h5py
    with h5py.File(os.path.join(DATA, fn), "r") as f:
        return f["min"][()], f["max"][()]


def get_dataset(name):
    if name == "dns":
        import h5py
        p = os.path.join(DATA, "Lagr_u3c_diffusion.h5")
        with h5py.File(p, "r") as f:
            mn, mx = f["min"][()], f["max"][()]
            n = f["train"].shape[0]
        return dict(kind="h5", path=p, min=mn, max=mx, n_traj=n, C=3)
    if name == "dm3c":
        mn, mx = _denorm_params("dm3c")
        return dict(kind="npy", path=os.path.join(DATA, "tmp", "dm3c_358400x2000x3_f32.npy"),
                    min=mn, max=mx, n_traj=358400, C=3)
    raise ValueError("Fig. 4c needs three-component data: only dns and dm3c are supported, since DM-1c is\n                     single-component and the mixed statistic is undefined there")


def worker(task):
    """(batch_id, ds, i0, i1) → (batch_id, S(nl,C), X(nl,3), cnt(nl))

    S holds sum di^2, X the sum di^2 dj^2 of the three component pairs, and cnt the number of sample
    pairs per lag (one count serves both).
    """
    bid, ds, i0, i1 = task
    mn = np.asarray(ds["min"], dtype=np.float32).reshape(1, 1, -1)
    mx = np.asarray(ds["max"], dtype=np.float32).reshape(1, 1, -1)
    if ds["kind"] == "h5":
        import h5py
        with h5py.File(ds["path"], "r") as f:
            u = f["train"][i0:i1].astype(np.float32)
    else:
        a = np.load(ds["path"], mmap_mode="r")
        u = np.asarray(a[i0:i1], dtype=np.float32)
    scale = (mx - mn) / 2.0
    v = (u + 1.0) * scale + mn                      # back to physical units
    del u, scale
    B, K, C = v.shape
    S = np.zeros((len(LAGS), C), dtype=np.float64)
    X = np.zeros((len(LAGS), len(PAIRS)), dtype=np.float64)
    cnt = np.zeros(len(LAGS), dtype=np.float64)
    for li, tau in enumerate(LAGS):
        d = v[:, tau:, :] - v[:, :K - tau, :]
        m2 = d * d                                  # delta^2 (a float32 product, halving memory and bandwidth)
        del d
        S[li] += m2.sum(axis=(0, 1), dtype=np.float64)          # the accumulation stays in float64
        for pi, (i, j) in enumerate(PAIRS):
            X[li, pi] += (m2[:, :, i] * m2[:, :, j]).sum(dtype=np.float64)
        cnt[li] += B * (K - tau)
        del m2
    del v
    return bid, S, X, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["dns", "dm3c"])
    ap.add_argument("--batches", type=int, default=10, help="number of independent blocks (the paper uses 10)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--n-traj", type=int, default=0, help="use only the first N trajectories (0 = all; for smoke tests)")
    ap.add_argument("--out-tag", default="", help="output-name suffix, so a smoke test does not overwrite the real npz")
    args = ap.parse_args()

    ds = get_dataset(args.dataset)
    n = args.n_traj if args.n_traj else ds["n_traj"]
    n = min(n, ds["n_traj"])
    bs = n // args.batches
    print(f"[22_fig4c_mixed] dataset={args.dataset} n_traj={n} batches={args.batches} "
          f"batch_size={bs} workers={args.workers} lags={LAGS[0]}..{LAGS[-1]}", flush=True)

    tasks = []
    for b in range(args.batches):
        j0 = b * bs
        j1 = (b + 1) * bs if b < args.batches - 1 else n
        for i0 in range(j0, j1, args.block):
            tasks.append((b, ds, i0, min(i0 + args.block, j1)))
    print(f"  tasks={len(tasks)} blocks", flush=True)

    Ssum = np.zeros((args.batches, len(LAGS), ds["C"]))
    Xsum = np.zeros((args.batches, len(LAGS), len(PAIRS)))
    cnt = np.zeros((args.batches, len(LAGS)))
    t0 = time.time()
    done = 0
    with Pool(args.workers) as pool:
        for bid, S, X, c in pool.imap_unordered(worker, tasks):
            Ssum[bid] += S
            Xsum[bid] += X
            cnt[bid] += c
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} blocks ({time.time()-t0:.0f}s)", flush=True)

    # S2 and the mixed moment of every block (the sample count per lag is the same, so they divide
    # directly)
    S2b = Ssum / cnt[:, :, None]                                  # (nb, nl, C)
    Xb = Xsum / cnt[:, :, None]                                   # (nb, nl, 3)
    S2b_avg = S2b.mean(axis=2)                                    # the common denominator (isotropy)
    Fb_pairs = Xb / (S2b_avg ** 2)[:, :, None]                    # (nb, nl, 3) per block and component pair
    Fb = Fb_pairs.mean(axis=2)                                    # (nb, nl) per block, pairs averaged

    Xtot = Xsum.sum(axis=0) / cnt.sum(axis=0)[:, None]
    S2tot = (Ssum.sum(axis=0) / cnt.sum(axis=0)[:, None]).mean(axis=1)
    F_mean = Xtot.mean(axis=1) / S2tot ** 2

    A_lo, A_hi = Fb.min(axis=0), Fb.max(axis=0)
    P = Fb_pairs.reshape(-1, len(LAGS))
    B_lo, B_hi = P.min(axis=0), P.max(axis=0)

    out = os.path.join(RESULTS, f"fig4c_mixed_{args.dataset}{args.out_tag}.npz")
    np.savez_compressed(out, lags=LAGS, tau=LAGS * 0.1, F_mean=F_mean,
                        X_mean=Xtot, S2_mean=S2tot,
                        Fb=Fb, Fb_pairs=Fb_pairs,
                        A_lo=A_lo, A_hi=A_hi, B_lo=B_lo, B_hi=B_hi,
                        n_traj=n, n_batch=args.batches, batch_size=bs,
                        dataset=args.dataset, pairs=np.array(PAIRS))
    print(f"  wrote {out}", flush=True)

    # point-by-point check against the source data of the paper (printed only; the plotting script sets
    # the convention)
    try:
        lines = open(os.path.join(DATA, "Li_et_al_2024_source_data",
                                  "42256_2024_810_MOESM7_ESM.txt"), errors="replace").read().split("\n")
        tag = {"dns": "DNS", "dm3c": "DM-3c"}[args.dataset]
        i = [k for k, l in enumerate(lines) if l.startswith("# Figure4c_%s.out" % tag)][0]
        pa = np.array([[float(x) for x in lines[i + 1 + j].split()] for j in range(1024)])
        print("  tau/tau_eta | paper mean | our F_mean (ratio) | paper +/- | A (10 blocks) width | B (30 blocks) width")
        for k in (0, 9, 99, 499, 999, 1023):
            pl = pa[k, 3] - pa[k, 2]
            print("  %6.1f | %9.4f | %9.4f (%5.3f) | %8.4f | %8.4f | %8.4f"
                  % (pa[k, 0], pa[k, 1], F_mean[k], F_mean[k] / pa[k, 1], pl,
                     A_hi[k] - A_lo[k], B_hi[k] - B_lo[k]), flush=True)
    except Exception as e:                                        # noqa: BLE001
        print("  (check against the source data of the paper skipped: %s)" % e, flush=True)
    print(f"  done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
