"""Streaming statistics of the Lagrangian structure functions S(p), the generalized
flatness F(p) and the mixed flatness F(4,ij); reproduces the quantities of Figs. 4 and 5a of Li et al. 2024 (Nat Mach Intell).

Design:
- block streaming (2048 trajectories per block by default), never a full load; one interface for the DNS h5 and the DM npy (mmap) inputs;
- the DM data are de-normalized to physical units first: v = (u+1)*(max-min)/2 + min (per component for u3c, scalar for u1c);
- lag grid 1..1024 storage steps (tau/tau_eta = 0.1..102.4), the grid the paper uses for zeta;
- multiprocessing (spawn-safe: workers live at module level and main is guarded);
- output results/structfuncs_<tag>.npz: lags, S2,S4,S6,S8, F4, F6, F8, F4ij (three pairs), cnt, meta.

Usage:
  python 02_structure_functions.py --dataset dns  [--n-traj N] [--offset 0] [--tag xxx]
  python 02_structure_functions.py --dataset dm3c ...
  python 02_structure_functions.py --dataset dm1c ...
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

LAGS = np.arange(1, 1025)  # storage-step index
BLOCK = 2048


def get_dataset(name):
    """Return dict(kind, path, min, max, n_traj)。min/max are the constants used to de-normalize to physical units."""
    if name == "dns":
        import h5py
        p = os.path.join(DATA, "Lagr_u3c_diffusion.h5")
        with h5py.File(p, "r") as f:
            mn = f["min"][()]
            mx = f["max"][()]
            n = f["train"].shape[0]
        return dict(kind="h5", path=p, min=mn, max=mx, n_traj=n)
    if name == "dm3c":
        import h5py
        p = os.path.join(DATA, "tmp", "dm3c_358400x2000x3_f32.npy")
        with h5py.File(os.path.join(DATA, "Lagr_u3c_diffusion.h5"), "r") as f:
            mn = f["min"][()]
            mx = f["max"][()]
        return dict(kind="npy", path=p, min=mn, max=mx, n_traj=358400)
    if name == "dm1c":
        import h5py
        p = os.path.join(DATA, "tmp", "dm1c_358400x2000x1_f32.npy")
        with h5py.File(os.path.join(DATA, "Lagr_u1c_diffusion.h5"), "r") as f:
            mn = float(f["min"][()])
            mx = float(f["max"][()])
        return dict(kind="npy", path=p, min=mn, max=mx, n_traj=358400)
    if name == "dm1c_repro":
        # Our own DM-1c sampling (self-trained DM-1c, ema_0.9999_250000.pt, 28 runs x 12800).
        # Same u1c physical domain as the official dm1c -> reuse its min/max scalars
        # (HDF5 if h5py is available, otherwise the scalars stored in structfuncs_dm1c_full.npz).
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
        return dict(kind="npy", path=p, min=mn, max=mx, n_traj=358400)
    raise ValueError(name)


def load_block(ds, i0, i1):
    if ds["kind"] == "h5":
        import h5py
        with h5py.File(ds["path"], "r") as f:
            v = f["train"][i0:i1]
        return v.astype(np.float32)
    a = np.load(ds["path"], mmap_mode="r")
    return np.asarray(a[i0:i1], dtype=np.float32)


def worker(task):
    """Accumulate the moments of every lag over one trajectory block; returns (S(nl,4), P(nl,3), cnt(nl)).
    Memory: the block is read once (f32, about 12 MB at B=512), then each lag is accumulated and its temporaries are released in place.
    """
    ds, i0, i1 = task
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
    v = (u + 1.0) * scale + mn                    # de-normalize -> physical units
    del u
    B, K, C = v.shape
    nl = len(LAGS)
    S = np.zeros((nl, 4), dtype=np.float64)        # p = 2,4,6,8
    P = np.zeros((nl, 3), dtype=np.float64)        # xy, xz, yz
    cnt = np.zeros(nl, dtype=np.float64)
    for li, tau in enumerate(LAGS):
        d = v[:, tau:, :] - v[:, :K - tau, :]      # (B, K-tau, C)
        m2 = d * d
        del d
        S[li, 0] += np.sum(m2, dtype=np.float64)
        m4 = m2 * m2
        S[li, 1] += np.sum(m4, dtype=np.float64)
        np.multiply(m2, m4, out=m2)                # in place: m2 <- m2*m4 (p=6)
        S[li, 2] += np.sum(m2, dtype=np.float64)
        del m2
        np.multiply(m4, m4, out=m4)                # in place: m4 <- m4**2 (p=8)
        S[li, 3] += np.sum(m4, dtype=np.float64)
        cnt[li] += B * (K - tau) * C
        del m4
    # the mixed-flatness numerator needs <di^2 dj^2>, but the in-place products above destroyed m2/m4, so it is computed again
    for li, tau in enumerate(LAGS):
        if C != 3:
            break
        d = v[:, tau:, :] - v[:, :K - tau, :]
        m2 = d * d
        del d
        x, y, z = m2[:, :, 0], m2[:, :, 1], m2[:, :, 2]
        P[li, 0] += np.sum(x * y, dtype=np.float64)
        P[li, 1] += np.sum(x * z, dtype=np.float64)
        P[li, 2] += np.sum(y * z, dtype=np.float64)
        del m2, x, y, z
    return S, P, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["dns", "dm3c", "dm1c", "dm1c_repro"])
    ap.add_argument("--n-traj", type=int, default=0, help="0 = all trajectories")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    ds = get_dataset(args.dataset)
    n = ds["n_traj"] if args.n_traj == 0 else min(args.n_traj, ds["n_traj"] - args.offset)
    tag = args.tag or f"{args.dataset}_n{n}"

    tasks = [(ds, i0, min(i0 + BLOCK, args.offset + n))
             for i0 in range(args.offset, args.offset + n, BLOCK)]
    print(f"[{tag}] dataset={args.dataset} n_traj={n} blocks={len(tasks)} "
          f"workers={args.workers} lags={LAGS[0]}..{LAGS[-1]}", flush=True)

    t0 = time.time()
    S = np.zeros((len(LAGS), 4)); P = np.zeros((len(LAGS), 3)); cnt = np.zeros(len(LAGS))
    done = 0
    with Pool(args.workers) as pool:
        for s, p, c in pool.imap_unordered(worker, tasks, chunksize=1):
            S += s; P += p; cnt += c
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} blocks ({time.time()-t0:.0f}s)", flush=True)

    C = 3 if args.dataset in ("dns", "dm3c") else 1
    Sp = S / cnt[:, None]                      # S(p): averaged over components, time and trajectories
    S2 = Sp[:, 0]
    F = Sp / (S2[:, None] ** (np.array([1.0, 2.0, 3.0, 4.0])[None, :]))  # F(2),F(4),F(6),F(8)
    Pm = P / (cnt[:, None] / C)                # per pair <di^2 dj^2>
    F4ij = Pm / (S2[:, None] ** 2) if C == 3 else None

    out = os.path.join(RESULTS, f"structfuncs_{tag}.npz")
    np.savez_compressed(
        out, lags=LAGS, S2=Sp[:, 0], S4=Sp[:, 1], S6=Sp[:, 2], S8=Sp[:, 3],
        F2=F[:, 0], F4=F[:, 1], F6=F[:, 2], F8=F[:, 3],
        F4ij_xy=F4ij[:, 0] if C == 3 else np.zeros(1),
        F4ij_xz=F4ij[:, 1] if C == 3 else np.zeros(1),
        F4ij_yz=F4ij[:, 2] if C == 3 else np.zeros(1),
        cnt=cnt, dataset=args.dataset, n_traj=n, offset=args.offset,
        min=np.atleast_1d(ds["min"]), max=np.atleast_1d(ds["max"]),
    )
    dt = time.time() - t0
    print(f"[{tag}] saved {out} ({dt:.0f}s)", flush=True)
    # quick check: anchor against the first row of MOESM7 Figure4a_DNS_p2, S2(tau = 0.1 tau_eta) = 1.724e-3
    print(f"  S2(lag1)={Sp[0,0]:.4e}  F4(lag1)={F[0,1]:.1f}  F4(lag1024)={F[-1,1]:.2f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
