"""Strict waveform diagnostics; SI-SDR measures implementation agreement only."""

import numpy as np

ATOL = 1e-5
RTOL = 1e-4


def agreement(reference, candidate):
    """No gain/time alignment for errors or acceptance. SI-SDR is supplementary."""
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.ndim != 2 or candidate.ndim != 2 or reference.shape[0] != candidate.shape[0]:
        raise ValueError("expected matching channel-first arrays")
    length = min(reference.shape[-1], candidate.shape[-1])
    if not length or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("expected nonempty finite waveforms")
    ref = reference[..., :length].astype(np.float64)
    got = candidate[..., :length].astype(np.float64)
    error = got - ref
    absolute = np.abs(error)
    limit = ATOL + RTOL * np.abs(ref)
    failures = absolute > limit
    signal_power, error_power = float(np.mean(ref**2)), float(np.mean(error**2))
    x, y = ref.ravel() - ref.mean(), got.ravel() - got.mean()
    energy = float(x @ x)
    alpha = float(x @ y) / energy if energy else 0.0
    projection = alpha * alpha * energy
    residual = float(np.sum((y - alpha * x) ** 2))
    score = float(10 * np.log10(projection / residual)) if projection and residual else None
    return {
        "reference_samples": reference.shape[-1],
        "candidate_samples": candidate.shape[-1],
        "max_absolute_error": float(absolute.max()),
        "rmse": float(np.sqrt(error_power)),
        "reference_rms": float(np.sqrt(signal_power)),
        "candidate_rms": float(np.sqrt(np.mean(got**2))),
        "relative_l2_error": float(np.sqrt(error_power / signal_power)) if signal_power else None,
        "reference_snr_db": float(10 * np.log10(signal_power / error_power))
        if signal_power and error_power
        else None,
        "reference_si_sdr_db": score,
        "si_sdr_status": "finite"
        if score is not None
        else "undefined_silence_or_constant"
        if not energy
        else "exact_scaled_reference"
        if not residual
        else "zero_projection",
        "near_silent_reference": bool(np.sqrt(signal_power) <= 1e-5),
        "absolute_error_quantiles": dict(
            zip(
                ("p50", "p95", "p99", "p999", "max"),
                map(float, np.quantile(absolute, [0.5, 0.95, 0.99, 0.999, 1.0])),
            )
        ),
        "strict_failed_samples": int(failures.sum()),
        "compared_scalar_samples": int(ref.size),
        "strict_failure_fraction": float(failures.mean()),
        "max_tolerance_ratio": float((absolute / limit).max()),
        "allclose_atol_1e-5_rtol_1e-4": bool(
            reference.shape == candidate.shape and not failures.any()
        ),
    }
