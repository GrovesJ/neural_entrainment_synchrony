"""Synchrony measures for auditory entrainment.

The phase-locking value (PLV) quantifies how consistently the phase of
a signal tracks a reference rhythm, in the range [0, 1].
"""

from __future__ import annotations

import numpy as np

__all__ = ["hilbert", "instantaneous_phase", "plv", "plv_pair", "golomb_hansel", "spike_pairwise_corr", "vector_strength", "mean_vector_strength"]


def hilbert(signal) -> np.ndarray:
    """Analytic signal of a real 1-D signal via an FFT Hilbert transform."""
    signal = np.asarray(signal, dtype=float)
    n = signal.size
    if n == 0:
        return np.empty(0, dtype=complex)
    spec = np.fft.fft(signal)
    h = np.zeros(n, dtype=float)
    h[0] = 1.0
    if n % 2 == 0:
        h[1:n // 2] = 2.0
        h[n // 2] = 1.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(spec * h)


def _bandpass(signal, fs: float, freq: float, bandwidth: float | None = None) -> np.ndarray:
    """Real-valued signal band-passed around ``freq`` (FFT zero-phase filter)."""
    signal = np.asarray(signal, dtype=float)
    n = signal.size
    if n == 0:
        return signal
    if bandwidth is None:
        bandwidth = freq
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    spec = np.fft.fft(signal)
    keep = np.abs(np.abs(freqs) - freq) <= bandwidth / 2.0
    spec[~keep] = 0.0
    return np.fft.ifft(spec).real


def instantaneous_phase(signal, fs: float, freq: float, bandwidth: float | None = None) -> np.ndarray:
    """Instantaneous phase (radians) of a signal band-passed around ``freq``.

    Parameters
    ----------
    signal : array_like
        Real-valued signal.
    fs : float
        Sampling rate in Hz.
    freq : float
        Centre frequency of the band in Hz.
    bandwidth : float, optional
        Width of the band-pass window in Hz (default ``freq``).

    Returns
    -------
    ndarray
        Phase in radians, same length as ``signal``.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return np.empty(0)
    return np.angle(hilbert(_bandpass(signal, fs, freq, bandwidth)))


def plv(signal, fs: float, freq: float, t0: float = 0.0) -> float:
    """Phase-locking value of ``signal`` to a fixed-frequency reference.

    PLV = |<exp(i * (phi(t) - 2*pi*freq*(t0 + t)))>|, in [0, 1].
    1 means the signal is perfectly phase-locked to the reference rhythm
    at ``freq``; 0 means no locking. ``t0`` is the reference start time
    in seconds (e.g. the first beat onset of the stimulus).
    """
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return float("nan")
    phi = instantaneous_phase(signal, fs, freq)
    t = t0 + np.arange(phi.size) / fs
    return float(np.abs(np.mean(np.exp(1j * (phi - 2.0 * np.pi * freq * t)))))


def plv_pair(x, y, fs: float) -> float:
    """PLV between the instantaneous phases of two broadband signals."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or x.size != y.size:
        return float("nan")
    phi_x = np.angle(hilbert(x))
    phi_y = np.angle(hilbert(y))
    return float(np.abs(np.mean(np.exp(1j * (phi_x - phi_y)))))


def golomb_hansel(V, fs=None, freq=None, bandwidth=None) -> float:
    """Golomb-Hansel synchrony measure chi of population membrane potentials.

    chi = sigma2(mean_i V_i) / mean_i sigma2(V_i), where the mean is
    across neurons at each time step and sigma2 is the temporal
    variance. chi is in [0, 1]: 1 means every neuron shares the same
    fluctuation (perfect synchrony), while chi ~= 1/N for N independent
    neurons. The chance level is therefore the reciprocal of the number
    of recorded neurons, not zero.

    Parameters
    ----------
    V : array_like
        Membrane potentials, shape ``(n_neurons, n_time)``. A single
        1-D trace is treated as one neuron and returns NaN.
    fs : float, optional
        Sampling rate in Hz. If given together with ``freq``, each
        trace is band-passed around ``freq`` first, so chi quantifies
        synchrony within that frequency band only.
    freq : float, optional
        Centre frequency of the band in Hz.
    bandwidth : float, optional
        Width of the band-pass window in Hz (default ``freq``).

    Returns
    -------
    float
        chi in [0, 1], or NaN if it cannot be computed.
    """
    V = np.asarray(V, dtype=float)
    if V.ndim == 1:
        V = V[None, :]
    if V.ndim != 2 or V.shape[0] < 2 or V.shape[1] < 2:
        return float("nan")
    if fs is not None and freq is not None:
        V = np.stack([_bandpass(v, fs, freq, bandwidth) for v in V])
    var_pop = float(np.var(V.mean(axis=0)))
    var_avg = float(np.mean(np.var(V, axis=1)))
    if var_avg <= 0.0:
        return float("nan")
    return var_pop / var_avg


def spike_pairwise_corr(spike_times, spike_indices, n_neurons, duration, bin_size=0.005) -> float:
    """Mean pairwise Pearson correlation of binned spike trains.

    Each neuron's spikes are binned into ``bin_size``-second bins over
    ``duration`` seconds, and the Pearson correlation between every pair
    of neurons' spike-count vectors is averaged over all pairs (i < j).

    The result is in [-1, 1]: 0 indicates independent firing, positive
    values indicate spike synchrony. Neurons that never spike have zero
    variance and are excluded.

    Note: the raw correlation depends on the bin size and firing rate,
    and independent neurons can give a small negative value. For a
    chance level, shuffle or jitter spike times and recompute.

    Parameters
    ----------
    spike_times : array_like
        Spike times in seconds.
    spike_indices : array_like of int
        Neuron index of each spike, 0-based and < ``n_neurons``.
    n_neurons : int
        Number of neurons in the population.
    duration : float
        Recording window in seconds.
    bin_size : float
        Bin width in seconds (default 5 ms).

    Returns
    -------
    float
        Mean pairwise correlation in [-1, 1], or NaN if it cannot be
        computed (fewer than two active neurons).
    """
    spike_times = np.asarray(spike_times, dtype=float)
    spike_indices = np.asarray(spike_indices, dtype=np.int64)
    if spike_times.size == 0 or spike_times.size != spike_indices.size:
        return float("nan")
    if duration <= 0 or bin_size <= 0:
        return float("nan")
    n_bins = max(1, int(np.ceil(duration / bin_size)))
    bin_idx = np.floor(spike_times / bin_size).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    keep = (spike_indices >= 0) & (spike_indices < n_neurons)
    spike_indices = spike_indices[keep]
    bin_idx = bin_idx[keep]
    if spike_indices.size == 0:
        return float("nan")
    flat = spike_indices * n_bins + bin_idx
    counts = np.bincount(flat, minlength=int(n_neurons) * n_bins).reshape(n_neurons, n_bins)
    active = counts.sum(axis=1) > 0
    if active.sum() < 2:
        return float("nan")
    corr = np.corrcoef(counts[active])
    iu = np.triu_indices(corr.shape[0], k=1)
    return float(np.nanmean(corr[iu]))


def vector_strength(spike_times, freq: float, t0: float = 0.0) -> float:
    """Vector strength (circular resultant) of spike phases relative to a
    fixed-frequency reference rhythm.

    VS = |<exp(i * 2*pi*freq*(t - t0))>| over spikes, in [0, 1]. 1 means
    every spike lands at the same phase of the reference rhythm; 0 means
    spikes are uniformly spread over the cycle. Unlike PLV computed on a
    population rate trace, VS is computed on spike times and does not
    increase with firing rate. The magnitude is independent of ``t0``
    (a constant phase offset only rotates the circular mean).

    Returns NaN if there are no spikes.
    """
    spike_times = np.asarray(spike_times, dtype=float)
    if spike_times.size == 0 or freq <= 0:
        return float("nan")
    phase = 2.0 * np.pi * freq * (spike_times - t0)
    return float(np.abs(np.mean(np.exp(1j * phase))))


def mean_vector_strength(spike_times, spike_indices, n_neurons, freq: float, t0: float = 0.0, min_spikes: int = 5) -> float:
    """Mean vector strength across neurons, each neuron weighted equally.

    Each neuron's vector strength is computed from its own spikes and the
    results are averaged over neurons with at least ``min_spikes`` spikes.
    Equal weighting keeps the score from being driven by high-rate
    neurons, and ``min_spikes`` avoids the trivial VS = 1 of a neuron
    with a single spike.
    """
    spike_times = np.asarray(spike_times, dtype=float)
    spike_indices = np.asarray(spike_indices, dtype=np.int64)
    if spike_times.size == 0 or spike_times.size != spike_indices.size or freq <= 0:
        return float("nan")
    keep = (spike_indices >= 0) & (spike_indices < n_neurons)
    spike_times = spike_times[keep]
    spike_indices = spike_indices[keep]
    if spike_indices.size == 0:
        return float("nan")
    z = np.exp(1j * 2.0 * np.pi * freq * (spike_times - t0))
    counts = np.bincount(spike_indices, minlength=int(n_neurons)).astype(float)
    sums = np.bincount(spike_indices, weights=z.real, minlength=int(n_neurons)) \
        + 1j * np.bincount(spike_indices, weights=z.imag, minlength=int(n_neurons))
    active = counts >= min_spikes
    if active.sum() == 0:
        return float("nan")
    per = np.abs(sums[active]) / counts[active]
    return float(np.mean(per))
