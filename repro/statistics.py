"""Float64 trajectory statistics, keeping temporal and component dependence.

Input axes are (physical trajectory, time, velocity component). Finite trajectory
bootstrap intervals are conditional on exchangeability of the observed traces;
they do not quantify training-seed variation or shared-flow DNS correlations.
"""
import numpy as np


def check_trajectories(u):
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 3 or u.shape[0] < 1 or u.shape[1] < 2 or u.shape[2] not in (1, 3):
        raise ValueError("expected (trajectory, time, component=1 or 3)")
    if not np.isfinite(u).all():
        raise ValueError("input contains nonfinite values")
    return u


def denormalize(u, minimum, maximum):
    minimum, maximum = np.asarray(minimum), np.asarray(maximum)
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise ValueError("normalization bounds must be finite")
    if np.any(maximum <= minimum):
        raise ValueError("normalization range must be positive")
    return (np.asarray(u, dtype=np.float64) + 1) * (maximum - minimum) / 2 + minimum


def validate_lags(lags, length):
    original = np.asarray(lags)
    if original.ndim != 1 or not len(original) or not np.all(np.isfinite(original)):
        raise ValueError("lags must be a finite nonempty vector")
    lags = original.astype(int)
    if not np.array_equal(original, lags) or np.any(lags < 1) or np.any(lags >= length):
        raise ValueError("lags must be integers in [1, time length)")
    if np.any(np.diff(lags) <= 0):
        raise ValueError("lags must be strictly increasing")
    return lags


def trajectory_moments(u, lags, orders=(2, 4, 6, 8)):
    """Per-trace, per-component means, shape (N, lag, component, order).

    Compute chunk by chunk for large datasets and save these sufficient
    statistics to a memmap; never allocate all trajectory increments at once.
    """
    u = check_trajectories(u)
    lags = validate_lags(lags, u.shape[1])
    if any(p < 2 or p % 2 for p in orders):
        raise ValueError("this study uses positive even moments")
    out = np.empty((len(u), len(lags), u.shape[2], len(orders)))
    for j, lag in enumerate(lags):
        delta = u[:, lag:, :] - u[:, :-lag, :]
        for k, order in enumerate(orders):
            out[:, j, :, k] = np.mean(delta ** order, axis=1, dtype=np.float64)
    return out


def mixed_trajectory_moments(u, lags):
    u = check_trajectories(u)
    if u.shape[2] != 3:
        raise ValueError("mixed moments require three simultaneous components")
    lags = validate_lags(lags, u.shape[1])
    out = np.empty((len(u), len(lags), 3))
    for j, lag in enumerate(lags):
        squared = (u[:, lag:, :] - u[:, :-lag, :]) ** 2
        for k, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
            out[:, j, k] = np.mean(squared[:, :, a] * squared[:, :, b], axis=1)
    return out


def flatness(structure, orders=(2, 4, 6, 8)):
    s = np.asarray(structure, dtype=np.float64)
    if orders[0] != 2 or s.shape[-1] != len(orders) or np.any(s[..., 0] <= 0):
        raise ValueError("positive second moment must be first")
    return s / s[..., :1] ** (np.asarray(orders) / 2)


def local_exponents(structure, lags):
    """Second-order differences in log lag, followed by derivative ratio.

    Paper comparisons must use all integer lags 1..1024. Sparse grids are only
    suitable for numerical validation, never silently interpreted as paper F5.
    """
    s, lag = np.asarray(structure), np.asarray(lags)
    if s.ndim != 2 or len(lag) < 3 or s.shape[0] != len(lag):
        raise ValueError("need (lag, order) and at least three lags")
    validate_lags(lag, int(max(lag)) + 1)
    if np.any(s <= 0):
        raise ValueError("logarithmic derivatives require positive moments")
    derivative = np.gradient(np.log(s), np.log(lag), axis=0, edge_order=2)
    if np.any(np.abs(derivative[:, 0]) < 1e-12):
        raise ValueError("second-moment logarithmic derivative is near zero")
    return derivative / derivative[:, :1]


def paper_batch_range(moments, seed=230708529, batches=10):
    """Min/max envelopes over trajectory partitions and individual components.

    These reproduce the paper's *type* of error bar, not its unavailable original
    random partition. A min/max envelope is not a confidence interval. The 30
    component-batches of a three-component dataset are not independent.
    """
    moments = np.asarray(moments)
    if batches < 2 or len(moments) < batches:
        raise ValueError("at least one physical trajectory per batch is needed")
    ids = np.random.default_rng(seed).permutation(len(moments))
    blocks = np.array_split(ids, batches)
    by_group = np.stack([moments[block].mean(axis=0) for block in blocks])
    # (group, lag, component, order) -> (group*component, lag, order)
    curves = by_group.transpose(0, 2, 1, 3).reshape(-1, *by_group.shape[1::2])
    fcurves = flatness(curves)
    return {"structure_min": curves.min(axis=0), "structure_max": curves.max(axis=0),
            "flatness_min": fcurves.min(axis=0), "flatness_max": fcurves.max(axis=0),
            "group_sizes": [len(block) for block in blocks], "curves": curves}


def trajectory_bootstrap(moments, seed=230708529, replicates=400):
    """Resample physical trajectories, preserving their time/component clusters."""
    moments = np.asarray(moments)
    if len(moments) < 2 or replicates < 2:
        raise ValueError("bootstrap needs >=2 trajectories and replicates")
    per_trace = moments.mean(axis=2)
    rng = np.random.default_rng(seed)
    estimates = np.empty((replicates, *per_trace.shape[1:]))
    for b in range(replicates):
        estimates[b] = per_trace[rng.integers(len(per_trace), size=len(per_trace))].mean(axis=0)
    return {"structure_ci95": np.quantile(estimates, [0.025, 0.975], axis=0),
            "flatness_ci95": np.quantile(flatness(estimates), [0.025, 0.975], axis=0)}


def acceleration(u, dt, method="central"):
    """Interior central difference (or forward difference for sensitivity)."""
    u = check_trajectories(u)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive")
    if method == "central":
        if u.shape[1] < 3:
            raise ValueError("central differences need three times")
        return (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
    if method == "forward":
        return np.diff(u, axis=1) / dt
    if method == "gradient":
        return np.gradient(u, dt, axis=1, edge_order=1)
    raise ValueError("method must be central, forward, or gradient")


def single_time_statistics(u):
    """Table II component-averaged square and pairwise square covariance."""
    u = check_trajectories(u)
    squares = u**2
    component = squares.mean(axis=(0, 1))
    sigma = None
    if u.shape[2] == 3:
        sigma = float(np.mean([np.mean(squares[:, :, i] * squares[:, :, j])
                               - component[i] * component[j]
                               for i, j in ((0, 1), (0, 2), (1, 2))]))
    return {"component_mean_square": component.tolist(),
            "mean_square": float(component.mean()), "mixed_squared_covariance": sigma}


def correlation(u, lags):
    """Uncentered finite-lag correlation, including the exactly defined zero lag."""
    u = check_trajectories(u)
    lags = np.asarray(lags)
    if not np.array_equal(lags, lags.astype(int)) or np.any(lags < 0) or np.any(lags >= u.shape[1]):
        raise ValueError("invalid correlation lag")
    return np.array([np.mean(u * u) if k == 0 else
                     np.mean(u[:, :-k, :] * u[:, k:, :]) for k in lags.astype(int)])


def standardized_histogram(values, edges):
    """Unconditional histogram density, explicitly retaining excluded tail mass."""
    x, edges = np.asarray(values, dtype=float), np.asarray(edges, dtype=float)
    if not x.size or not np.isfinite(x).all() or not np.all(np.diff(edges) > 0):
        raise ValueError("invalid values or histogram edges")
    mean, std = float(x.mean()), float(x.std())
    if std <= 0:
        raise ValueError("zero standard deviation")
    z = (x.ravel() - mean) / std
    counts = np.histogram(z, bins=edges)[0]
    return {"edges": edges, "counts": counts,
            "density": counts / (x.size * np.diff(edges)),
            "total_count": x.size, "below": int((z < edges[0]).sum()),
            "above": int((z > edges[-1]).sum()), "mean": mean, "std": std}
