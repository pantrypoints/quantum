"""Braket-ready engine. Same signature as core.ENGINES values: f(p, n, rng) -> dict(value, ci, queries).

Strategy (Phase 1, honest): classically compute the *annual* distribution (100 states -> 7 qubits),
load it into a Braket circuit with `state_vector`, run on LocalSimulator (or SV1 if AWS creds present),
measure, histogram bitstrings back to states, then do the 3-year fold + targets classically.
This validates the plumbing and lets you compare shot-noise error to MC at equal shots.

The 1/M advantage claim (iterative QAE) is Phase 2 and lives in `qae_estimate` below; it is NOT wired
into the app's main engine path until it has an oracle for the both-exhausted event.
"""
import numpy as np
import core

N_ANNUAL = 100
N_QUBITS = 7   # 2^7 = 128 >= 100

def _pad(dist):
    v = np.zeros(2 ** N_QUBITS)
    v[:N_ANNUAL] = dist
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def braket_sampler(p, n, rng, device_arn=None):
    """Estimate the 5 targets by sampling annual states on a Braket device."""
    try:
        from braket.circuits import Circuit
        from braket.devices import LocalSimulator
    except ImportError:
        raise RuntimeError("amazon-braket-sdk not installed. pip install amazon-braket-sdk amazon-braket-default-simulator")

    A = core.annual_dist(p)
    psi = _pad(A)

    circ = Circuit().state_vector(psi)
    circ.probability()

    if device_arn:
        from braket.aws import AwsDevice
        device = AwsDevice(device_arn)
    else:
        device = LocalSimulator()

    result = device.run(circ, shots=int(n)).result()
    counts = result.measurement_counts  # keys are bitstrings

    # map bitstrings -> annual state index 0..99
    idx = np.empty(N_ANNUAL)
    for bits, c in counts.items():
        s = int(bits, 2)
        if s < N_ANNUAL:
            idx[s] = c
    if idx.sum() == 0:
        raise RuntimeError("Braket returned no in-range samples.")
    emp = idx / idx.sum()

    # fold 3 years with the empirical annual dist (classical convolution is fine here;
    # the quantum part is the *sampling* of annual states)
    from dataclasses import replace
    p_emp = replace(p, years=p.years)
    # Build term dist with empirical annual:
    cur = np.zeros(core.NS); cur[0] = 1
    for _ in range(p.years):
        cur = np.bincount(core.ADD.ravel(), weights=np.outer(cur, emp).ravel(), minlength=core.NS)
    V = core.target_matrix(p)
    x = V[np.arange(core.NS)]           # per-state target row
    val = cur @ x
    # crude CI from shot noise: propagate via bootstrap of annual counts
    se = np.sqrt(np.maximum(val * (1 - val) / n, 1e-18))
    return dict(value=val, ci=1.96 * se, queries=int(n), backend=str(device))

def qae_estimate(prob, rel_err, device_arn=None):
    """Phase-2 placeholder: iterative QAE for a single probability. Not wired into the app yet.
    Returns (estimate, n_oracle_calls). Raises NotImplementedError until an oracle is supplied."""
    raise NotImplementedError(
        "Phase 2: build a reversible oracle for the both-exhausted event, then run iterative QAE "
        "(Grinko et al. 2021) on SV1. The textbook bound is in core.qae_vs_mc."
    )