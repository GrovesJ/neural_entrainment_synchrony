"""Single-neuron SNR from a point-process GLM."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "bin_spikes",
    "stimulus_covariates",
    "history_covariates",
    "deviance",
    "single_neuron_snr",
    "population_snr",
]


def bin_spikes(spike_times, duration_s: float, bin_ms: float) -> np.ndarray:
    """Bin spike times into counts."""
    spike_times = np.asarray(spike_times, dtype=float)
    n_bins = max(1, int(np.ceil(duration_s * 1000.0 / bin_ms)))
    counts, _ = np.histogram(spike_times, bins=n_bins, range=(0.0, duration_s))
    return counts.astype(np.int64)


def stimulus_covariates(t, freq: float, harmonics=(1, 2)) -> np.ndarray:
    """Build sine/cosine stimulus-phase covariates."""
    t = np.asarray(t, dtype=float)
    columns = []
    for h in harmonics:
        phase = 2.0 * np.pi * h * freq * t
        columns.append(np.cos(phase))
        columns.append(np.sin(phase))
    if not columns:
        return np.empty((t.size, 0))
    return np.column_stack(columns)


def history_covariates(spike_counts, n_lags: int) -> np.ndarray:
    """Build lagged spike-history covariates."""
    spike_counts = np.asarray(spike_counts, dtype=np.int64)
    n_bins = spike_counts.size
    history = np.zeros((n_bins, n_lags), dtype=float)
    for lag in range(1, n_lags + 1):
        history[lag:, lag - 1] = spike_counts[:-lag]
    return history


def _poisson_glm(X, y):
    """Fit a Poisson GLM and return parameters and log-likelihood."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n_bins = X.shape[0]
    X_design = np.column_stack([np.ones(n_bins), X])

    def neg_log_likelihood(params):
        eta = X_design @ params
        eta = np.clip(eta, -40.0, 40.0)
        rate = np.exp(eta)
        return -np.sum(y * eta - rate)

    def neg_gradient(params):
        eta = X_design @ params
        eta = np.clip(eta, -40.0, 40.0)
        rate = np.exp(eta)
        return X_design.T @ (rate - y)

    result = minimize(
        neg_log_likelihood,
        np.zeros(X_design.shape[1]),
        jac=neg_gradient,
        method="BFGS",
    )
    params = result.x
    eta = np.clip(X_design @ params, -40.0, 40.0)
    rate = np.exp(eta)
    log_likelihood = float(np.sum(y * eta - rate))
    return params, log_likelihood


def deviance(
    spike_counts,
    stim_covariates,
    history,
    min_mean_rate: float = 1e-9,
) -> dict:
    """Decompose Poisson-GLM deviance into SNR components."""
    y = np.asarray(spike_counts, dtype=float)
    mean_rate = max(float(y.mean()), min_mean_rate)

    ll_null = float(np.sum(y * np.log(mean_rate) - mean_rate))

    _, ll_stim = _poisson_glm(stim_covariates, y)
    _, ll_hist = _poisson_glm(history, y)
    full = np.hstack([stim_covariates, history])
    _, ll_full = _poisson_glm(full, y)

    D_stim = 2.0 * (ll_stim - ll_null)
    D_hist = 2.0 * (ll_hist - ll_null)
    D_full = 2.0 * (ll_full - ll_null)
    snr_db = float("nan") if D_hist <= 0.0 else 10.0 * np.log10(D_stim / D_hist)
    return {
        "D_stim": float(D_stim),
        "D_hist": float(D_hist),
        "D_full": float(D_full),
        "snr_db": snr_db,
    }


def single_neuron_snr(
    spike_times,
    duration_s: float,
    freq: float,
    bin_ms: float = 5.0,
    n_lags: int = 20,
    harmonics=(1, 2),
) -> dict:
    """Estimate SNR for one neuron's spike train."""
    spike_times = np.asarray(spike_times, dtype=float)
    counts = bin_spikes(spike_times, duration_s, bin_ms)
    t = np.arange(counts.size) * (bin_ms / 1000.0)
    stim = stimulus_covariates(t, freq, harmonics=harmonics)
    hist = history_covariates(counts, n_lags)
    result = deviance(counts, stim, hist)
    result["n_spikes"] = int(spike_times.size)
    result["n_bins"] = int(counts.size)
    return result


def population_snr(
    spike_times,
    spike_indices,
    n_neurons: int,
    duration_s: float,
    freq: float,
    bin_ms: float = 5.0,
    n_lags: int = 20,
    harmonics=(1, 2),
    min_spikes: int = 20,
    neuron_ids=None,
) -> list[dict]:
    """Estimate SNR for selected neurons."""
    spike_times = np.asarray(spike_times, dtype=float)
    spike_indices = np.asarray(spike_indices, dtype=np.int64)
    if spike_times.size != spike_indices.size:
        return []

    keep = (spike_indices >= 0) & (spike_indices < n_neurons)
    spike_times = spike_times[keep]
    spike_indices = spike_indices[keep]

    if neuron_ids is None:
        neuron_ids = np.arange(n_neurons)
    neuron_ids = np.asarray(neuron_ids, dtype=np.int64)

    rows = []
    for neuron in neuron_ids:
        mask = spike_indices == neuron
        times = spike_times[mask]
        if times.size < min_spikes:
            continue
        row = single_neuron_snr(
            times, duration_s, freq, bin_ms=bin_ms, n_lags=n_lags, harmonics=harmonics
        )
        row["neuron"] = int(neuron)
        rows.append(row)
    return rows
