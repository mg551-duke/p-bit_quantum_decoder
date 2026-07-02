"""Tests for the Sinter integration of the p-bit DEM decoder."""

import numpy as np
import pytest

stim = pytest.importorskip("stim")
sinter = pytest.importorskip("sinter")

from pbit_qec.sinter_decoder import CompiledPBitDecoder, PBitSinterDecoder


def _rep_circuit(p: float = 0.03) -> "stim.Circuit":
    return stim.Circuit.generated(
        "repetition_code:memory",
        rounds=5,
        distance=5,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
    )


def test_compiles_to_sinter_compiled_decoder() -> None:
    circuit = _rep_circuit()
    dem = circuit.detector_error_model(decompose_errors=False)
    decoder = PBitSinterDecoder(chains=8, sweeps=30, beta_start=0.5, beta_stop=3.0)
    compiled = decoder.compile_decoder_for_dem(dem=dem)
    assert isinstance(compiled, CompiledPBitDecoder)
    assert compiled.num_detectors == dem.num_detectors
    assert compiled.num_observables == dem.num_observables


def test_bit_packed_decode_shape_and_low_ler() -> None:
    circuit = _rep_circuit(p=0.03)
    dem = circuit.detector_error_model(decompose_errors=False)
    shots = 400
    sampler = circuit.compile_detector_sampler(seed=0)
    packed_dets, packed_obs = sampler.sample(
        shots, separate_observables=True, bit_packed=True
    )

    decoder = PBitSinterDecoder(
        chains=24, sweeps=60, beta_start=0.5, beta_stop=3.0, seed=1
    )
    compiled = decoder.compile_decoder_for_dem(dem=dem)
    packed_pred = compiled.decode_shots_bit_packed(
        bit_packed_detection_event_data=packed_dets
    )

    expected_bytes = -(-dem.num_observables // 8)  # ceil division
    assert packed_pred.shape == (shots, expected_bytes)
    assert packed_pred.dtype == np.uint8

    errors = np.count_nonzero(np.any(packed_pred != packed_obs, axis=1))
    assert errors / shots < 0.03


def test_decode_is_deterministic_for_fixed_seed() -> None:
    circuit = _rep_circuit()
    dem = circuit.detector_error_model(decompose_errors=False)
    packed_dets, _ = circuit.compile_detector_sampler(seed=2).sample(
        64, separate_observables=True, bit_packed=True
    )
    out = []
    for _ in range(2):
        compiled = PBitSinterDecoder(
            chains=8, sweeps=30, beta_start=0.5, beta_stop=3.0, seed=7
        ).compile_decoder_for_dem(dem=dem)
        out.append(
            compiled.decode_shots_bit_packed(
                bit_packed_detection_event_data=packed_dets
            )
        )
    np.testing.assert_array_equal(out[0], out[1])
