"""Basis detector filtering for CSS memory circuits.

Ports relay_bp's ``_filter_detectors_by_basis`` (and its ``detect_data_qubits``
helper) so our harness can decode the SAME basis-filtered problem as the RelayBP
reference instead of the full DEM. For a CSS ``memory_Z`` experiment the logical
failure is driven by X errors, so the reference keeps only the X-error-sensitive
detectors via ``filter_detectors_by_basis(circuit, "Z")`` (and symmetrically
``"X"`` for ``memory_X``). Dropping the other sector's detectors yields a much
smaller decoding problem (e.g. bb144 d=12: 936 detectors / 8784 mechanisms vs
the full 1728 / 67824) at the cost of discarding cross-basis / Y correlations.

Source: IBM relay_bp (Apache-2.0), src/relay_bp/stim/sinter/runner.py and
src/relay_bp/stim/noise.py. Reproduced here verbatim (modulo names) so pbit_qec
does not take a hard dependency on relay_bp.
"""

from __future__ import annotations

import numpy as np
import stim


def detect_data_qubits(circuit: stim.Circuit) -> list[int]:
    """Detect data qubits as those that are only measured once in a circuit.

    Warning: This is hacky and likely will only work with your typical memory
    circuits (matches relay_bp.stim.noise.detect_data_qubits).
    """
    qubit_times_measured = [0 for _ in range(circuit.num_qubits)]

    for inst in circuit:
        if inst.name.startswith("M") and not inst.gate_args_copy():
            for qubit in inst.targets_copy():
                qubit_times_measured[qubit.qubit_value] += 1

    return [
        qubit
        for qubit, times_measured in enumerate(qubit_times_measured)
        if times_measured == 1
    ]


def filter_detectors_by_basis(circuit: stim.Circuit, basis: str) -> stim.Circuit:
    """Drop detectors that do not respond to the requested data-qubit basis.

    ``basis="Z"`` keeps detectors sensitive to X errors (use with memory_Z);
    ``basis="X"`` keeps detectors sensitive to Z errors (use with memory_X).
    Observable instructions are preserved; only DETECTOR instructions are pruned.
    """
    pauli_error = "Z" if basis == "X" else "X"
    circuit = circuit.flattened()

    noiseless_circuit = circuit.without_noise()
    sampler = noiseless_circuit.compile_detector_sampler()
    reference_detectors, _ = sampler.sample(1, separate_observables=True)
    reference_detectors = reference_detectors[0, :]
    num_detectors = len(reference_detectors)

    detector_is_sensitive = np.full(num_detectors, False, dtype=bool)
    data_qubits = detect_data_qubits(noiseless_circuit)
    to_test = list(data_qubits)
    to_test_set = set(to_test)

    inst_idx = 0
    while to_test:
        for qubit in to_test:
            injected_circuit = stim.Circuit()
            injected_circuit += noiseless_circuit
            injected_circuit.insert(
                inst_idx,
                stim.CircuitInstruction(pauli_error + "_ERROR", [qubit], [1.0]),
            )
            injected_sampler = injected_circuit.compile_detector_sampler()
            injected_detectors, _ = injected_sampler.sample(1, separate_observables=True)
            injected_detectors = injected_detectors[0, :]
            detectors_flipped = np.where(reference_detectors != injected_detectors)
            detector_is_sensitive[detectors_flipped] = True

        to_test = []
        for inst in noiseless_circuit[inst_idx:]:
            inst_idx += 1
            if inst.name.startswith("R") or inst.name.startswith("M"):
                to_test = list(to_test_set)
                break

    filtered_circuit = stim.Circuit()
    detector_idx = 0
    for inst in circuit:
        if inst.name == "DETECTOR":
            keep = detector_is_sensitive[detector_idx]
            detector_idx += 1
            if not keep:
                continue
        filtered_circuit.append(inst)
    return filtered_circuit
