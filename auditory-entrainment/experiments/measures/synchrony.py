"""
Synchrony measures for auditory entrainment.

The phase phase locking value (PLV) measures how consistently the phase of
a signal follows a rhythm, in the range [0, 1].
"""

from __future__ import annotations

import numpy as np

__all__ = ["hilbert", "instantaneous_phase", "plv", "plv_pair", "golomb_hansel", "spike_pairwise_corr", "vector_strength", "mean_vector_strength"]


def hilbert(signal) -> np.ndarray:
    """Hilbert transform of a signal"""
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
    """Band pass a signal around a target frequency with a rectangular window"""
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
    """
    Instantaneous phase in rads of a signal signal band passed around a certain frequency
    """
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return np.empty(0)
    return np.angle(hilbert(_bandpass(signal, fs, freq, bandwidth)))


def plv(signal, fs: float, freq: float, t0: float = 0.0) -> float:
    """
    Phase locking value of a signal to a reference rhythm
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
    """
    Golomb Hansel synchrony measure chi of population membrane potentials
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
    """
    Pairwise Pearson correlation of binned spike trains:

    1. Each neuron's spikes are binned into "duration" seconds
    2. Count spikes of each neuron in each bin
    3. Compute Pearson corelation between every pairs spike count vectors
    4. Sum each correllation and divide by number of pairs
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
    """
    Vector strength of spike times to a reference rhythm:

    1. Convert spike times to phases of the reference rhythm
    2. Make a unit vector for each phase
    3. Take the mean of those unit vectors
    4. The length of the mean vector is the vector strength

    """
    spike_times = np.asarray(spike_times, dtype=float)
    if spike_times.size == 0 or freq <= 0:
        return float("nan")
    phase = 2.0 * np.pi * freq * (spike_times - t0)
    return float(np.abs(np.mean(np.exp(1j * phase))))


def mean_vector_strength(spike_times, spike_indices, n_neurons, freq: float, t0: float = 0.0, min_spikes: int = 5) -> float:
    """
    Vector strength averaged over neurons instead of over all spikes.

    1. Remove spikes from neuron indices outside [0, n_neurons]
    2. Give each spike a unit vector at its phase
    3. Sum the vectors separately for each neuron (real and imaginary parts)
    4. For each neuron with at least min_spikes spikes, VS = |sum| / count
    5. Return the mean across those neurons
    """
    spike_times = np.asarray(spike_times, dtype=float)
    spike_indices = np.asarray(spike_indices, dtype=np.int64)
    if spike_times.size == 0 or spike_times.size != spike_indices.size or freq <= 0:
        return float("nan")
    valid = (spike_indices >= 0) & (spike_indices < n_neurons)
    spike_times = spike_times[valid]
    spike_indices = spike_indices[valid]
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
