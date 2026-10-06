"""
Brian2 model of neural entrainment.

Excitatory/inhibitory LIF network with:

- spike-frequency adaptation in the inhibitory neurons
- PING-style gamma oscillations, 
- rhythmic thalamic drive 

"""

from dataclasses import dataclass as _dataclass

import brian2 as _b2 # type: ignore
import numpy as _np # type: ignore

from . import stimuli as _stimuli

__all__ = ["Params", "Result", "simulate"]


class Params:
    """Parameters of the auditory entrainment model. """

    def __init__(
        self,
        *,
        
        # Simulation Parameters
        duration: float = 1000.0,
        dt: float = 0.1,
        seed: int = 42,
        v_dt: float = 1.0, # Membrane-potential recording interval (ms)
        
        # Population Sizes
        n_exc: int = 400,
        n_inh: int = 100,

        # Network Dynamics
        C_m: float = 200.0, # Membrane Capacitance
        tau_exc: float = 20.0, # Excitatory cell membrane constant
        tau_inh: float = 10.0, # Inhibitory cell membrane constant
        E_l: float = -65.0, # Leak Reversal Potential
        E_e: float = 0.0, # Excitatory Reversal Potential
        E_i: float = -80.0, # Inhibtaory Reversal Potential
        v_thresh: float = -50.0, # Spike Threshold
        v_reset: float = -65.0, # Post spike potential
        tau_ref_exc: float = 2.0, # Excitatory refractory period
        tau_ref_inh: float = 1.0, # Inhibitory refractory period

        # Synaptic Time Constants
        tau_e: float = 5.0, # Excitatory Conductance Decay
        tau_i: float = 10.0, # Inhibitory Conductance Decay
        delay_ei: float = 1.0, # E-to-I
        delay_ie: float = 1.0, # I-to-E
        delay_ei_std: float = 0.0, # E-to-I across-neuron delay spread (ms)
        delay_ie_std: float = 0.0, # I-to-E across-neuron delay spread (ms)

        # Recurrent weights
        w_ee: float = 1.5, # Recurrent E-E Conductance Increment
        w_ei: float = 1.0, # E-I weight
        w_ie: float = 3.0, # I-E weight 
        w_ii: float = 4.0, # I-I weight
        
        # Connection Probabilities
        p_ee: float = 0.02, # E-E connection
        p_ei: float = 0.4, # E-I connectioon
        p_ie: float = 0.5, # I-E connection
        p_ii: float = 0.5, # I-I connection
        
        # Spike Frequency Adaption
        tau_adapt: float = 150.0, # adaption decay constant
        w_adapt: float = 100.0, # adaptaiton current amplitude
        delta_adapt: float = 0.1, # incremept to adaptation per spike
        
        # Rhythmic Thalamic Input
        n_thal: int = 20, # Number of Thalamic Input Channels
        w_thal: float = 0.05, # Synaptic weight from Thalamus to Excitatory Neurons
        p_thal: float = 0.5, # Connection Probability

        # Background Noise
        bg_noise: float = 0.0, # Std (mV) of Gaussian white-noise voltage injection; 0 = off

        # Tonic Drive
        I_bias: float = 100.0, # Mean tonic depolarizing current (pA)
        I_bias_std: float = 0.0, # Across-neuron standard deviation (pA)
        weight_lognormal_sigma: float = 0.0, # Shared per-neuron synaptic weight spread
    ):

        # Simulation Parameters
        self.duration = duration
        self.dt = dt
        self.seed = seed
        self.v_dt = v_dt
       
        # Population Sizes
        self.n_exc = n_exc
        self.n_inh = n_inh

        # Network Dynamics
        self.C_m = C_m
        self.tau_exc = tau_exc
        self.tau_inh = tau_inh
        self.E_l = E_l
        self.E_e = E_e
        self.E_i = E_i
        self.v_thresh = v_thresh
        self.v_reset = v_reset

        # Synaptic Time Constants
        self.tau_ref_exc = tau_ref_exc
        self.tau_ref_inh = tau_ref_inh
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.delay_ei = delay_ei
        self.delay_ie = delay_ie
        if delay_ei < 0 or delay_ie < 0:
            raise ValueError("Mean synaptic delays must be non-negative.")
        if delay_ei_std < 0 or delay_ie_std < 0:
            raise ValueError("Synaptic delay spreads must be non-negative.")
        self.delay_ei_std = delay_ei_std
        self.delay_ie_std = delay_ie_std
        
        # Recurrent Weights
        self.w_ee = w_ee
        self.w_ei = w_ei
        self.w_ie = w_ie
        self.w_ii = w_ii

        # Connection Probabilities
        self.p_ee = p_ee
        self.p_ei = p_ei
        self.p_ie = p_ie
        self.p_ii = p_ii
        
        # Spike Frequency Adaptation
        self.tau_adapt = tau_adapt
        self.w_adapt = w_adapt
        self.delta_adapt = delta_adapt
        
        # Rhythmic Thalamic Input
        self.n_thal = n_thal
        self.w_thal = w_thal
        self.p_thal = p_thal
        
        # Background Noise
        self.bg_noise = bg_noise

        # Tonic Drive
        self.I_bias = I_bias
        if I_bias_std < 0:
            raise ValueError("I_bias_std must be non-negative.")
        self.I_bias_std = I_bias_std
        if weight_lognormal_sigma < 0:
            raise ValueError("weight_lognormal_sigma must be non-negative.")
        self.weight_lognormal_sigma = weight_lognormal_sigma

    def __repr__(self) -> str:
        parts = ", ".join(f"{name}={getattr(self, name)!r}" for name in _PARAM_FIELDS)
        return f"Params({parts})"


_PARAM_FIELDS = (
    "duration", "dt", "seed", "v_dt",
    "n_exc", "n_inh",
    "C_m", "tau_exc", "tau_inh",
    "E_l", "E_e", "E_i", "v_thresh", "v_reset",
    "tau_ref_exc", "tau_ref_inh",
    "tau_e", "tau_i", "delay_ei", "delay_ie", "delay_ei_std", "delay_ie_std",
    "w_ee", "w_ei", "w_ie", "w_ii",
    "p_ee", "p_ei", "p_ie", "p_ii",
    "tau_adapt", "w_adapt", "delta_adapt",
    "n_thal", "w_thal", "p_thal",
    "bg_noise",
    "I_bias", "I_bias_std", "weight_lognormal_sigma",
)


@_dataclass
class Result:
    """Recorded output of a simulation run."""

    params: Params
    stimulus: _stimuli.Rhythm | None
    n_exc: int
    n_inh: int
    spike_times: _np.ndarray  # seconds; excitatory spikes first, then inhibitory
    spike_indices: _np.ndarray  # global index, 0..n_exc-1 excitatory, n_exc.. inhibitory
    rate_t: _np.ndarray  # seconds
    rate_exc: _np.ndarray  # Hz
    rate_inh: _np.ndarray  # Hz
    v_t: _np.ndarray  # seconds
    v_exc: _np.ndarray  # mV, shape (n_exc, n_steps)
    v_inh: _np.ndarray  # mV, shape (n_inh, n_steps)


def simulate(
    params: Params | None = None,
    stimulus: _stimuli.Rhythm | None = None,
    background: _stimuli.Rhythm | None = None,
) -> Result:
    """Build the network, run it, and return the recorded activity.

    ``stimulus`` is a :class:`auditory_entrainment.stimuli.Rhythm`
    describing the rhythmic (accent) drive; ``None`` means no rhythmic
    input. ``background`` is an optional steady metrical beat presented
    on a separate input channel, so syncopated accents are heard against
    a steady beat.
    """
    if params is None:
        params = Params()
   

    _b2.seed(params.seed)
    _b2.defaultclock.dt = params.dt * _b2.ms

    components = _build_network(params, stimulus, background)
    exc = components["exc"]
    inh = components["inh"]

    exc_spikes = _b2.SpikeMonitor(exc)
    inh_spikes = _b2.SpikeMonitor(inh)
    exc_rate = _b2.PopulationRateMonitor(exc)
    inh_rate = _b2.PopulationRateMonitor(inh)
    exc_v = _b2.StateMonitor(exc, "v", record=True, dt=params.v_dt * _b2.ms)
    inh_v = _b2.StateMonitor(inh, "v", record=True, dt=params.v_dt * _b2.ms)

    net = _b2.Network()
    net.add(*[c for c in components.values() if c is not None])
    net.add(exc_spikes, inh_spikes, exc_rate, inh_rate, exc_v, inh_v)
    net.run(params.duration * _b2.ms, report=None)

    return _collect_result(
        params, stimulus, exc_spikes, inh_spikes, exc_rate, inh_rate, exc_v, inh_v
    )

def _build_network(
    params: Params,
    stimulus: _stimuli.Rhythm | None = None,
    background: _stimuli.Rhythm | None = None,
):
    ms = _b2.ms
    mV = _b2.mV
    pF = _b2.pF
    nS = _b2.nS
    pA = _b2.pA

    C_m = params.C_m * pF
    tau_e = params.tau_e * ms
    tau_i = params.tau_i * ms
    E_l = params.E_l * mV
    E_e = params.E_e * mV
    E_i = params.E_i * mV
    v_thresh = params.v_thresh * mV
    v_reset = params.v_reset * mV
    g_leak_exc = C_m / (params.tau_exc * ms)
    g_leak_inh = C_m / (params.tau_inh * ms)
    w_adapt = params.w_adapt * pA
    tau_adapt = params.tau_adapt * ms
    sigma_noise = params.bg_noise * mV
    noise_e = sigma_noise * (2 / (params.tau_exc * ms)) ** 0.5
    noise_i = sigma_noise * (2 / (params.tau_inh * ms)) ** 0.5
    bias_rng = _np.random.default_rng(params.seed)
    weight_rng = _np.random.default_rng(params.seed + 1)
    exc_bias = _np.clip(
        bias_rng.normal(params.I_bias, params.I_bias_std, params.n_exc),
        0.0,
        None,
    ) * pA
    inh_bias = _np.clip(
        bias_rng.normal(params.I_bias, params.I_bias_std, params.n_inh),
        0.0,
        None,
    ) * pA

    exc_eqs = _b2.Equations(
        "dv/dt = (g_leak*(E_leak - v) + g_e*(E_e - v) + g_i*(E_i - v) - w_adapt*a + I_bias) / C_m + noise_e * xi : volt (unless refractory)\n"
        "da/dt = -a / tau_adapt : 1\n"
        "dg_e/dt = -g_e / tau_e : siemens\n"
        "dg_i/dt = -g_i / tau_i : siemens\n"
        "I_bias : amp\n"
    )
    inh_eqs = _b2.Equations(
        "dv/dt = (g_leak*(E_leak - v) + g_e*(E_e - v) + g_i*(E_i - v) + I_bias) / C_m + noise_i * xi : volt (unless refractory)\n"
        "dg_e/dt = -g_e / tau_e : siemens\n"
        "dg_i/dt = -g_i / tau_i : siemens\n"
        "I_bias : amp\n"
    )

    common = {
        "E_leak": E_l,
        "E_e": E_e,
        "E_i": E_i,
        "C_m": C_m,
        "tau_e": tau_e,
        "tau_i": tau_i,
        "v_thresh": v_thresh,
        "v_reset": v_reset,
    }

    exc_ns = {
        **common,
        "g_leak": g_leak_exc,
        "w_adapt": w_adapt,
        "tau_adapt": tau_adapt,
        "delta_adapt": params.delta_adapt,
        "noise_e": noise_e,
    }

    inh_ns = {**common,
               "g_leak": g_leak_inh,
                 "noise_i": noise_i,
    }

    exc = _b2.NeuronGroup(
        params.n_exc,
        exc_eqs,
        threshold="v > v_thresh",
        reset="v = v_reset; a += delta_adapt",
        refractory=params.tau_ref_exc * ms,  # type: ignore[arg-type]
        method="euler",
        namespace=exc_ns,
        name="exc",
    )
    inh = _b2.NeuronGroup(
        params.n_inh,
        inh_eqs,
        threshold="v > v_thresh",
        reset="v = v_reset",
        refractory=params.tau_ref_inh * ms,  # type: ignore[arg-type]
        method="euler",
        namespace=inh_ns,
        name="inh",
    )
    exc.v = E_l
    inh.v = E_l
    exc.I_bias = exc_bias
    inh.I_bias = inh_bias

    thal = None
    if stimulus is not None:
        thal = stimulus.spike_generator_group(
            params.duration / 1000.0, n_neurons=params.n_thal, seed=params.seed
        )
    thal_bg = None
    if background is not None:
        thal_bg = background.spike_generator_group(
            params.duration / 1000.0, n_neurons=params.n_thal, seed=params.seed + 1
        )

    s_ee = _b2.Synapses(
        exc, exc,
        on_pre="g_e_post += w",
        model="w : siemens (constant)",
        name="s_ee",
    )
    s_ee.connect(condition="i != j", p=params.p_ee)
    _assign_presynaptic_weights(s_ee, params.w_ee * nS, params.n_exc, params.weight_lognormal_sigma, weight_rng)
    s_ei = _b2.Synapses(
        exc, inh,
        on_pre="g_e_post += w",
        model="w : siemens (constant)",
        name="s_ei",
    )
    s_ei.connect(p=params.p_ei)
    _assign_presynaptic_weights(s_ei, params.w_ei * nS, params.n_exc, params.weight_lognormal_sigma, weight_rng)
    ei_delays = _np.clip(
        bias_rng.normal(params.delay_ei, params.delay_ei_std, params.n_exc),
        0.0,
        None,
    )
    s_ei.delay = ei_delays[_np.asarray(s_ei.i[:], dtype=int)] * ms
    s_ie = _b2.Synapses(
        inh, exc,
        on_pre="g_i_post += w",
        model="w : siemens (constant)",
        name="s_ie",
    )
    s_ie.connect(p=params.p_ie)
    _assign_presynaptic_weights(s_ie, params.w_ie * nS, params.n_inh, params.weight_lognormal_sigma, weight_rng)
    ie_delays = _np.clip(
        bias_rng.normal(params.delay_ie, params.delay_ie_std, params.n_inh),
        0.0,
        None,
    )
    s_ie.delay = ie_delays[_np.asarray(s_ie.i[:], dtype=int)] * ms
    s_ii = _b2.Synapses(
        inh, inh,
        on_pre="g_i_post += w",
        model="w : siemens (constant)",
        name="s_ii",
    )
    s_ii.connect(condition="i != j", p=params.p_ii)
    _assign_presynaptic_weights(s_ii, params.w_ii * nS, params.n_inh, params.weight_lognormal_sigma, weight_rng)
    s_thal = None
    if thal is not None:
        s_thal = _b2.Synapses(
            thal, exc,
            on_pre="g_e_post += w",
            model="w : siemens (constant)",
            name="s_thal",
        )
        s_thal.connect(p=params.p_thal)
        _assign_presynaptic_weights(s_thal, params.w_thal * nS, params.n_thal, params.weight_lognormal_sigma, weight_rng)
    s_thal_bg = None
    if thal_bg is not None:
        s_thal_bg = _b2.Synapses(
            thal_bg, exc,
            on_pre="g_e_post += w",
            model="w : siemens (constant)",
            name="s_thal_bg",
        )
        s_thal_bg.connect(p=params.p_thal)
        _assign_presynaptic_weights(s_thal_bg, params.w_thal * nS, params.n_thal, params.weight_lognormal_sigma, weight_rng)

    return {
        "exc": exc,
        "inh": inh,
        "thal": thal,
        "thal_bg": thal_bg,
        "s_ee": s_ee,
        "s_ei": s_ei,
        "s_ie": s_ie,
        "s_ii": s_ii,
        "s_thal": s_thal,
        "s_thal_bg": s_thal_bg,
    }


def _assign_presynaptic_weights(synapses, base_weight, n_pre, sigma, rng):
    """Assign mean-preserving lognormal weights shared by each source neuron."""
    factors = _np.exp(sigma * rng.standard_normal(n_pre) - 0.5 * sigma**2)
    source_indices = _np.asarray(synapses.i[:], dtype=int)
    synapses.w = base_weight * factors[source_indices]


def _collect_result(params, stimulus, exc_spikes, inh_spikes, exc_rate, inh_rate, exc_v, inh_v) -> Result:
    exc_times = _np.asarray(exc_spikes.t / _b2.second)
    inh_times = _np.asarray(inh_spikes.t / _b2.second)
    exc_indices = _np.asarray(exc_spikes.i, dtype=_np.int64)
    inh_indices = _np.asarray(inh_spikes.i, dtype=_np.int64) + params.n_exc

    return Result(
        params=params,
        stimulus=stimulus,
        n_exc=params.n_exc,
        n_inh=params.n_inh,
        spike_times=_np.concatenate([exc_times, inh_times]),
        spike_indices=_np.concatenate([exc_indices, inh_indices]),
        rate_t=_np.asarray(exc_rate.t / _b2.second),
        rate_exc=_np.asarray(exc_rate.rate / _b2.Hz),
        rate_inh=_np.asarray(inh_rate.rate / _b2.Hz),
        v_t=_np.asarray(exc_v.t / _b2.second),
        v_exc=_np.asarray(exc_v.v / _b2.mV),
        v_inh=_np.asarray(inh_v.v / _b2.mV),
    )