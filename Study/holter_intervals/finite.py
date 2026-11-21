"""Fixed-design 1-alpha intervals for a stratified, bounded finite-population mean.

Each row of sample_sums contains the sum of sampled minute AF fractions for
each stratum; counts and sample_sizes describe a fixed stratified SRSWOR plan.
Targets and ascertainment must be fixed before sampling. No distributional
assumption about patients, clocks, within-record correlations, or outcomes.
"""
import numpy as np
from scipy.special import xlogy


def _prepare(counts, sample_sizes, sample_sums, alpha):
    counts, sizes = np.asarray(counts), np.asarray(sample_sizes)
    if (not 0 < alpha < 1 or counts.ndim != 1 or sizes.shape != counts.shape
            or counts.dtype.kind not in "iu" or sizes.dtype.kind not in "iu"):
        raise ValueError("Expected integer stratum counts/sizes and alpha in (0,1)")
    if np.any(counts < sizes) or np.any(sizes < 0) or np.any((counts > 0) & (sizes == 0)):
        raise ValueError("Every nonempty stratum must have a valid nonzero sample")
    n_total = counts.sum()
    if n_total <= 0:
        raise ValueError("The target population is empty")
    sums = np.asarray(sample_sums, dtype=float)
    if sums.ndim not in (1, 2) or sums.shape[-1] != len(counts):
        raise ValueError("Expected one vector or a matrix of stratum sample sums")
    if not np.isfinite(sums).all() or np.any((sums < -1e-12) | (sums > sizes + 1e-12)):
        raise ValueError("Sample sums must come from outcomes in [0,1]")
    sums = np.clip(np.atleast_2d(sums), 0, sizes)
    active = sizes < counts
    exact = sums[:, ~active].sum(axis=1) / n_total
    mass = counts[active].sum() / n_total
    if not np.any(active):
        return active, exact, mass, None, None, None
    weights = counts[active] / counts[active].sum()
    sizes = sizes[active]
    data = sums[:, active] / sizes
    return active, exact, mass, weights, sizes, data


def _offset(x):
    return xlogy(x, x) + xlogy(1-x, 1-x)


def _scalar_upper(x, radius):
    """Outward endpoint where kl(x||p) first exceeds radius for p>=x."""
    left = x.copy()
    right = np.ones_like(x)
    constant = _offset(x)
    for _ in range(44):
        mid = (left + right) * .5
        kl = constant - xlogy(x, mid) - xlogy(1-x, 1-mid)
        outside = kl >= radius
        right = np.where(outside, mid, right)
        left = np.where(outside, left, mid)
    return right


def bounded_intervals(counts, sample_sizes, sample_sums, alpha=.1):
    """Two-sided KL-Chernoff design bound for the direct stratified mean.

    Returns two one-dimensional arrays, including for a single sample row.
    """
    active, exact, mass, weights, sizes, data = _prepare(
        counts, sample_sizes, sample_sums, alpha)
    if not np.any(active):
        return exact, exact.copy()
    yhat = np.einsum('ij,j->i', data, weights)
    yhat = np.clip(yhat, 0, 1)
    radius = np.max(weights/sizes) * np.log(2/alpha)
    lo = 1-_scalar_upper(1-yhat, radius)
    hi = _scalar_upper(yhat, radius)
    return np.maximum(0, exact+mass*lo-2e-15), np.minimum(1, exact+mass*hi+2e-15)


def _profile_upper(x, cutoff, weights, sizes):
    """Project the convex KL-level set with its separable Lagrangian."""
    const = _offset(x)
    slope = weights / sizes
    def region(multiplier):
        z = multiplier[:, None] * slope
        root = np.sqrt(np.square(1-z) + 4*z*x)
        # Stable complementary formula even when the maximizer tends to one.
        complement = (2*(1-x)) / (1+z+root)
        # Stable direct formula near zero, with special handling at x=0.
        denom = 1-z+root
        prob = np.divide(2*x, denom, out=np.zeros_like(x), where=denom>0)
        # Where z >= 1, the analogous direct quadratic avoids cancellation.
        prob = np.where(z >= 1, ((z-1)+root)/(2*np.maximum(z,1)), prob)
        cost = np.einsum('ij,j->i', const-xlogy(x,prob)-xlogy(1-x,complement),sizes)
        target = np.einsum('ij,j->i',prob,weights)
        return cost, target

    # Doubling is only needed while an attainable maximum is meaningfully <1.
    # Do not use size of the whole array for convergence: each row obtains its
    # own bracket, preserving accuracy for rare and heterogeneous samples.
    left = np.zeros(len(x))
    right = np.full(len(x), 1./np.max(slope))
    ceiling = np.zeros(len(x),dtype=bool)
    for _ in range(64):
        cost, mean = region(right)
        grow = (cost < cutoff) & (mean < 1-2e-13)
        ceiling |= (cost < cutoff) & ~grow
        if not np.any(grow):
            break
        left[grow] = right[grow]
        right[grow] *= 2.
    else:
        ceiling |= grow
    for _ in range(39):
        mid = (right+left)*.5
        cost, _ = region(mid)
        outside = cost >= cutoff
        right = np.where(outside, mid, right)
        left = np.where(outside, left, mid)
    _, mean = region(right)
    return np.where(ceiling, 1., np.minimum(1,mean+2e-13))


def profile_intervals(counts, sample_sizes, sample_sums, q, alpha=.1):
    """Two-sided projected confidence set by inversion of a stratum product bet.

    q is the vector of *presampling* mean predictions within strata. Its
    dependence on reviewed outcomes would invalidate the confidence bound.
    Populated stratum predictions are clipped to [.01, .99]. Empty-stratum
    entries may be NaN. Returns (lower, upper) arrays on the prevalence scale.
    """
    active, exact, mass, weights, sizes, x = _prepare(
        counts, sample_sizes, sample_sums, alpha)
    q = np.asarray(q, dtype=float)
    if q.shape != active.shape or not np.all(np.isfinite(q[active])) or np.any((q[active]<0)|(q[active]>1)):
        raise ValueError("Supply a presampling probability for each noncensus stratum")
    if not np.any(active):
        return exact, exact.copy()
    q = np.clip(q[active], .01, .99)
    penalty = _offset(x) - xlogy(x,q) - xlogy(1-x,1-q)
    cutoff = np.log(1/alpha) + np.einsum('ij,j->i',penalty,sizes)
    hi = _profile_upper(x,cutoff,weights,sizes)
    lo = 1-_profile_upper(1-x,cutoff,weights,sizes)
    return np.maximum(0,exact+mass*lo-2e-15), np.minimum(1,exact+mass*hi+2e-15)
