"""tetrad_lib.py -- shared helpers for the tetrad (four-particle) analysis scripts.

NOTE: run scripts/tetrad_lib_selftest.py after editing this library.

Conventions (details and provenance: TETRAD_DM.md sections 1-4)
----------------------------------------------------------------
* quartet k        = raw tracer rows 4k+0 .. 4k+3; particle 1 = origin of the orthogonal "corner"
* 12-channel order = [p1x,p1y,p1z, p2x,p2y,p2z, p3x,p3y,p3z, p4x,p4y,p4z]   (same as `train` and `pos`)
* geometry         rho_1=(x2-x1)/sqrt2, rho_2=(2x3-x2-x1)/sqrt6, rho_3=(3x4-x3-x2-x1)/sqrt12
                   I = rho rho^T, eigenvalues g1>=g2>=g3, r = sqrt(tr I), I_i = g_i/r^2
* identity         r^2 = tr(I) = (1/8) sum_{i,j}|xi-xj|^2      <-- ORDERED pairs
                              = (1/4) sum_{i<j}|xi-xj|^2 = sum_i |xi-X_cm|^2
* units (OUR DNS, TURB-Lagr / Biferale et al. 2024 Tab. 1; NOT Biferale 2005's eta=5e-3)
                   eta = 0.0042, tau_eta = 0.023, dx = 2*pi/1024, dts = 0.00225, box L = 2*pi
* minimal image    applies to DISPLACEMENTS ONLY -- never to velocity differences
* initial condition closed form (orthogonal corner, leg h = dx/2):
                   g = (h^2, h^2, h^2/4) = (dx^2/4, dx^2/4, dx^2/16), I_i = (4/9,4/9,1/9), r(0) = 1.5 h = 0.75 dx
                   in eta units: legs 0.73047 eta, diagonals sqrt2 h = 1.03304 eta, r(0) = 1.09570 eta
                   measured over all 81920 quartets (script 44b, see TETRAD_DM.md section 3.5):
                   0.73128 / 1.03428 / 1.09725 eta, per-quartet scatter ~2.3%, V(0)/(h^3/6) = 0.999981
                   (NOT a regular tetrahedron -- VA08's initial separations are regular-tet edge lengths)

All readers return per-chunk arrays; iterate with the generators so that a full 81920-quartet
pass never needs more than ~1 GB of RAM.
"""
import os
import numpy as np
import h5py

# Root of the analysis tree: the directory that holds results/ and Data/.
# Override with the LTS_ROOT environment variable; the default is the parent
# of this analysis/ directory.
ROOT = os.environ.get("LTS_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "Data", "full_traj_tracers.h5")
TET = os.path.join(ROOT, "Data", "Lagr_tetrad_u12c.h5")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "results", "figs")   # publish_to_onedrive.sh publishes only the *.png written here

ETA = 0.0042          # Kolmogorov length of OUR dataset
ETA_ERR = 0.0001
TAU_ETA = 0.023       # Kolmogorov time
DTS = 0.00225         # storage interval (15 dt)
DX = 2.0 * np.pi / 1024
LBOX = 2.0 * np.pi
NT = 2000             # time steps per trajectory
NQ = 81920            # quartets in the dataset
DX2_2 = DX * DX / 2.0   # 1.8825e-5, the unit used in Biferale 2005's Fig. 1 caption
NQUARTET = 4
NCH = 12


# ---------------------------------------------------------------- constants / normalisation
# Official DM-3c training file: the authoritative source of the calibre-A constants.  The tetrad
# dataset records their provenance in attrs but does NOT carry the datasets itself (script 41
# reads them from here and only stores the attrs).
DM3C = os.path.join(ROOT, "Data", "Lagr_u3c_diffusion.h5")
MIN_REF = np.array([-9.97037474, -11.24481512, -9.49838034])
MAX_REF = np.array([10.27369886, 10.31416093, 10.19539748])


def calibre_a():
    """Per-component min/max of DM-3c (calibre A), float64.

    Prefers the tetrad file (if it ever carries them), otherwise falls back to the official
    DM-3c training file -- both are checked by tetrad_lib_selftest.py against MIN_REF/MAX_REF.
    """
    for path in (TET, DM3C):
        with h5py.File(path, "r") as f:
            if "min" in f and "max" in f:
                return np.asarray(f["min"][:], np.float64), np.asarray(f["max"][:], np.float64)
    raise RuntimeError(f"no min/max found in either {TET} or {DM3C}")


def normalise(v):
    """Physical 12-channel velocities -> normalised tensor (inverse of denormalise)."""
    mn, mx = calibre_a()
    half = np.tile((mx - mn) / 2.0, NQUARTET)
    off = np.tile(mn, NQUARTET)
    return 2.0 * (v - off) / (2.0 * half) - 1.0


def denormalise(v_norm):
    """Normalised 12-channel tensor -> physical velocities (channels cycle c mod 3)."""
    mn, mx = calibre_a()
    half = np.tile((mx - mn) / 2.0, NQUARTET)
    off = np.tile(mn, NQUARTET)
    return (v_norm + 1.0) * half + off


# ---------------------------------------------------------------- readers (chunked)
def _chunks(k0, k1, chunk):
    for a in range(k0, k1, chunk):
        yield a, min(a + chunk, k1)


def iter_positions(k0=0, k1=NQ, chunk=64, source="pos", image=None):
    """Yield (k0, x) with x (nq, NT, 4, 3) physical positions.

    source="pos"  : the stored position companion dataset (unwrapped, exact)
    source="raw"  : straight from full_traj_tracers.h5 (slower: full-file scan)
    image="min"   : replace by positions relative to particle 1's minimal image
                    (2*pi periodic); image=None keeps the stored (unwrapped) geometry.
    """
    for a, b in _chunks(k0, k1, chunk):
        if source == "pos":
            with h5py.File(TET, "r") as f:
                x = f["pos"][a:b].astype(np.float64).reshape(b - a, NT, NQUARTET, 3)
        elif source == "raw":
            with h5py.File(RAW, "r") as f:
                raw = f["traj3d"][4 * a:4 * b, :, 0:3].astype(np.float64)
            x = raw.reshape(b - a, NQUARTET, NT, 3).transpose(0, 2, 1, 3)
        else:
            raise ValueError(source)
        if image == "min":
            x = x - minimal_image(x - x[:, :, :1, :])
        yield a, x


def iter_positions_idx(idx, chunk=64, source="pos", image=None):
    """Same as iter_positions but for an explicit (already balanced) array of quartet indices."""
    idx = np.asarray(idx, dtype=np.int64)
    for a in range(0, idx.size, chunk):
        sl = idx[a:a + chunk]
        if source == "pos":
            with h5py.File(TET, "r") as f:
                x = f["pos"][sl].astype(np.float64).reshape(sl.size, NT, NQUARTET, 3)
        elif source == "raw":
            rows = np.concatenate([4 * sl + r for r in range(NQUARTET)])
            with h5py.File(RAW, "r") as f:
                raw = f["traj3d"][rows, :, 0:3].astype(np.float64)
            x = raw.reshape(sl.size, NQUARTET, NT, 3)[np.argsort(rows % NQUARTET)].transpose(0, 2, 1, 3)
        else:
            raise ValueError(source)
        if image == "min":
            x = x - minimal_image(x - x[:, :, :1, :])
        yield sl, x


def integrate_rho(rho0, W, dts=None):
    """rho(t) = rho0 + trapezoidal time integral of W, with rho(t_0) == rho0 EXACTLY.

    W is (n, T, 3, 3), rho0 (n, 3, 3).  The first increment must be zero: a naive
    incr[0] = W[0]/2 adds a half-step (dt*W(0)/2 ~ 2e-4, i.e. ~4% of the initial |rho|)
    to the initial condition -- caught by the dns/integrated route comparison.
    """
    dts = DTS if dts is None else dts
    W = np.asarray(W, dtype=np.float64)
    incr = np.empty_like(W)
    incr[:, 0] = 0.0
    if W.shape[1] > 1:
        incr[:, 1:] = 0.5 * (W[:, 1:] + W[:, :-1])
    return rho0[:, None] + dts * np.cumsum(incr, axis=1)


def iter_rhoW(idx, chunk=64, source="dns"):
    """Yield (idx_chunk, rho, W) where rho holds the Jacobi RELATIVE positions and W the Jacobi
    relative velocities -- everything downstream is translation invariant by construction.

    source="dns"        : rho from the stored (exact) positions, W from the stored velocities.
    source="integrated" : Pantea-style reconstruction -- rho(t) = rho(0) + trapezoid integral of W,
                          with rho(0) taken from the stored t=0 configuration (the DM has no
                          positions at all, so this is the only route for generated samples).
                          Comparing the two on DNS data quantifies every error the integration
                          route introduces (mean drift, quadrature, low-frequency fidelity).
    """
    idx = np.asarray(idx, dtype=np.int64)
    for a in range(0, idx.size, chunk):
        sl = idx[a:a + chunk]
        with h5py.File(TET, "r") as f:
            vn = f["train"][sl].astype(np.float64)
            if source != "integrated":
                x = f["pos"][sl].astype(np.float64).reshape(sl.size, NT, NQUARTET, 3)
        v = denormalise(vn).reshape(sl.size, NT, NQUARTET, 3)
        W = jacobi_W(v)
        if source == "dns":
            rho = jacobi_rho(x)
        elif source == "integrated":
            with h5py.File(TET, "r") as f:
                x0 = f["pos"][sl, 0].astype(np.float64).reshape(sl.size, NQUARTET, 3)
            rho0 = jacobi_rho(x0[:, None, :, :])[:, 0]                  # (n, 3, 3)
            rho = integrate_rho(rho0, W)
        else:
            raise ValueError(source)
        yield sl, rho, W


def iter_velocities(k0=0, k1=NQ, chunk=64):
    """Yield (k0, v) with v (nq, NT, 4, 3) physical velocities (de-normalised, calibre A)."""
    for a, b in _chunks(k0, k1, chunk):
        with h5py.File(TET, "r") as f:
            v = denormalise(f["train"][a:b].astype(np.float64)).reshape(b - a, NT, NQUARTET, 3)
        yield a, v


# ---------------------------------------------------------------- geometry
def minimal_image(d):
    """Minimum-image of displacements d (in-place-free). NEVER apply to velocities."""
    return d - LBOX * np.floor(d / LBOX + 0.5)


def jacobi_rho(x):
    """x (nq, NT, 4, 3) -> rho (nq, NT, 3, 3) whose COLUMNS are rho_1, rho_2, rho_3."""
    x1, x2, x3, x4 = (x[:, :, i, :] for i in range(4))
    r1 = (x2 - x1) / np.sqrt(2.0)
    r2 = (2.0 * x3 - x2 - x1) / np.sqrt(6.0)
    r3 = (3.0 * x4 - x3 - x2 - x1) / np.sqrt(12.0)
    return np.stack([r1, r2, r3], axis=-1)          # (..., 3 components, 3 vectors)


def jacobi_W(v):
    """Same combination applied to velocities -> W (nq, NT, 3, 3) (columns W_1..W_3)."""
    return jacobi_rho(v)


def eig3_sym(A):
    """Vectorised eigenvalues of symmetric 3x3 matrices, sorted DESCENDING (..., 3).

    Closed-form trigonometric solution -- ~100x faster than np.linalg.eigvalsh, which loops in
    Python over the batch.  Validated against numpy (see scripts/tetrad_lib_selftest.py).
    """
    A = np.asarray(A, dtype=np.float64)
    a00, a01, a02 = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    a11, a12, a22 = A[..., 1, 1], A[..., 1, 2], A[..., 2, 2]
    p1 = a01 * a01 + a02 * a02 + a12 * a12
    q = (a00 + a11 + a22) / 3.0
    p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2.0 * p1
    p = np.sqrt(np.maximum(p2, 0.0) / 6.0)
    safe = p > 0
    ps = np.where(safe, p, 1.0)
    b00, b11, b22 = (a00 - q) / ps, (a11 - q) / ps, (a22 - q) / ps
    b01, b02, b12 = a01 / ps, a02 / ps, a12 / ps
    detB = (b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
            + b02 * (b01 * b12 - b11 * b02))
    r = np.clip(detB / 2.0, -1.0, 1.0)
    phi = np.arccos(r) / 3.0
    e1 = q + 2.0 * ps * np.cos(phi)
    e3 = q + 2.0 * ps * np.cos(phi + 2.0 * np.pi / 3.0)
    e2 = 3.0 * q - e1 - e3
    g = np.stack([e1, e2, e3], axis=-1)
    g = np.where(safe[..., None], g, np.stack([q, q, q], axis=-1))   # degenerate: p == 0
    return -np.sort(-g, axis=-1)


def inertia(rho):
    """rho (..., 3, 3) -> (g desc, r^2, I_i) evaluated at each (quartet, time)."""
    I = rho @ np.swapaxes(rho, -1, -2)                     # (..., 3, 3)
    g = eig3_sym(I)                                        # descending
    r2 = g.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        Ii = np.where(r2[..., None] > 0, g / np.maximum(r2, 1e-300)[..., None], np.nan)
    return g, r2, Ii


def K_tensor(rho, W):
    """Exact identity K = 1/2 (W rho^T + rho W^T) (paper eq. 1); no numerical differentiation."""
    return 0.5 * (W @ np.swapaxes(rho, -1, -2) + rho @ np.swapaxes(W, -1, -2))


def eval_K(K):
    """Eigenvalues of K, sorted descending (kappa_3 is typically negative)."""
    return eig3_sym(K)


def block_stats(values, blocks, axis=0):
    """Per-block mean and min/max: values (n, ...) with blocks (n,) -> mean/min/max over blocks."""
    vals = np.asarray(values, dtype=np.float64)
    labels = np.unique(blocks)
    means = np.stack([vals[blocks == b].mean(axis=axis) for b in labels])
    return means.mean(0), means.min(0), means.max(0), labels.size


# --- large-scale parameters of the TURB-Lagr DNS -----------------------------------------------
# Needed for the van Aartrijk (2008) volume normalisation V/(u_rms^3 T_L^3) = V/L_int^3.
# Tab.1 of Biferale et al. (2024) gives N, L, dx, dt, nu, eps, tau_eta, eta, Re_lambda -- but NOT
# u_rms or T_L.  u_rms is measured from the data (Table 2 of Li et al. 2024: E = 3.0131 => 1.74),
# and eps differs between the two source papers (1.4 +- 0.1 vs 1.8 +- 0.1); 1.8 is the value
# self-consistent with the tabulated lambda = 0.14 and Re_lambda = 310, so it is the default.
U_RMS = 1.74
EPS_LI = 1.8
EPS_B24 = 1.4
L_INT = U_RMS ** 3 / EPS_LI          # 2.93 = u_rms^3/eps  (integral length scale)
L_INT_B24 = U_RMS ** 3 / EPS_B24     # 3.76 (if eps = 1.4 is used)
T_L = L_INT / U_RMS                  # 1.68 (large-eddy turnover, integral-scale based)


def balanced(nq, start=0, stop=NQ):
    """Balanced quartet indices: spread evenly across the 32x32 pencil columns.

    Taking the first nq quartets would cover only nq/80 pencil columns (a spatially clustered
    sample -> unrepresentative error bands).  This sampler walks the whole index range instead.
    """
    stop = min(stop, NQ)
    if nq >= stop - start:
        return np.arange(start, stop)
    step = (stop - start) / float(nq)
    return np.unique(np.minimum((start + step * np.arange(nq) + step / 2).astype(np.int64), stop - 1))


def pencil_column(index):
    """Spatial block label of a tracer row (32x32 pencils, 320 rows each): index = x*10240+y*320+local."""
    return index // 320


# ---------------------------------------------------------------- plotting
def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
        "axes.titlesize": 12, "figure.autolayout": True,
    })
    return plt


def savefig(fig, name, verbose=True, dpi=None):
    os.makedirs(FIGURES, exist_ok=True)
    path = os.path.join(FIGURES, name)
    # bbox_inches="tight": without it anything wider than the axes (e.g. a long title) is clipped
    # at the figure edge -- caught by the user on B01 (title truncated on the right).
    kw = {"bbox_inches": "tight"}
    if dpi is not None:
        kw["dpi"] = dpi
    fig.savefig(path, **kw)
    if verbose:
        print("figure ->", path)
    return path
