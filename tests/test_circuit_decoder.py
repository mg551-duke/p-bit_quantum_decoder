"""Tests for the sparse chromatic-block-Gibbs circuit-level decoder."""

import numpy as np
import pytest

stim = pytest.importorskip("stim")

from pbit_qec.circuit_decoder import (
    ColoredBlockGibbsDecoder,
    greedy_coloring,
)
from pbit_qec.dem import dem_to_matrices


def _rep_matrices(p: float = 0.03):
    circuit = stim.Circuit.generated(
        "repetition_code:memory",
        rounds=5,
        distance=5,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
    )
    dem = circuit.detector_error_model(decompose_errors=False)
    return circuit, dem_to_matrices(dem)


def test_greedy_coloring_classes_are_detector_disjoint() -> None:
    _, mats = _rep_matrices()
    h = mats.check_matrix.tocsc()
    for cols in greedy_coloring(h):
        # No detector (row) may be touched by two columns of the same color.
        row_counts = np.asarray(h[:, cols].sum(axis=1)).ravel()
        assert row_counts.max() <= 1


def test_csr_gather_matches_dense_violated_check_count() -> None:
    _, mats = _rep_matrices()
    eng = ColoredBlockGibbsDecoder(mats, chains=4, sweeps=1, seed=0)
    rng = np.random.default_rng(0)
    residual = (rng.random((eng.m, 8)) < 0.3).astype(np.uint8)
    for block in eng._colors:
        before = (block.gather @ residual).astype(np.int64)
        dense = (np.asarray(eng._check_matrix[:, block.cols].todense(), dtype=np.int64).T
                 @ residual.astype(np.int64))
        np.testing.assert_array_equal(before, dense)


def test_incremental_residual_invariant_holds_across_sweeps() -> None:
    _, mats = _rep_matrices()
    eng = ColoredBlockGibbsDecoder(
        mats, chains=8, sweeps=20, beta_start=0.5, beta_stop=3.0, seed=1
    )
    rng = np.random.default_rng(2)
    rows = 8
    target = (rng.random(eng.m) < 0.3).astype(np.uint8)
    states = np.zeros((eng.n, rows), dtype=np.uint8)
    residual = np.repeat(target[:, None], rows, axis=1).copy()
    prior_energy = np.zeros(rows, dtype=np.float64)
    invalid = residual.sum(axis=0, dtype=np.int32)
    h = eng._check_matrix
    for beta in eng._beta_schedule():
        eng._sweep(states, residual, prior_energy, invalid, float(beta), rng)
        expected = (np.asarray((h @ states), dtype=np.int64) % 2).astype(np.uint8)
        expected ^= target[:, None]
        np.testing.assert_array_equal(residual, expected)
        # The incrementally tracked invalid count stays exact vs residual weight.
        np.testing.assert_array_equal(invalid, residual.sum(axis=0, dtype=np.int32))


def test_detectorless_mechanisms_are_dropped() -> None:
    mats = dem_to_matrices(
        stim.DetectorErrorModel(
            """
            error(0.1) D0 D1
            error(0.2) L0
            error(0.3) D1 D2
            """
        )
    )
    # The observable-only mechanism carries no syndrome information and is dropped.
    eng = ColoredBlockGibbsDecoder(mats, chains=2, sweeps=1, seed=0)
    assert mats.num_mechanisms == 3
    assert eng.n == 2


def test_decode_repetition_code_is_valid_and_low_ler() -> None:
    circuit, mats = _rep_matrices(p=0.03)
    dets, obs = circuit.compile_detector_sampler(seed=0).sample(
        500, separate_observables=True
    )
    eng = ColoredBlockGibbsDecoder(
        mats, chains=24, sweeps=60, beta_start=0.5, beta_stop=3.0, seed=1
    )
    predictions, stats = eng.decode_with_stats(dets.astype(np.uint8))
    assert predictions.shape == (500, mats.num_observables)
    assert predictions.dtype == np.uint8
    assert stats.valid_fraction == 1.0
    ler = float(np.mean(np.any(predictions != obs.astype(np.uint8), axis=1)))
    assert ler < 0.03


def test_capture_block_captures_superset_of_sweep_valid_states() -> None:
    circuit, mats = _rep_matrices(p=0.05)
    dets, _ = circuit.compile_detector_sampler(seed=0).sample(
        200, separate_observables=True
    )
    dets = dets.astype(np.uint8)
    common = dict(chains=16, sweeps=12, beta_start=0.5, beta_stop=3.0, seed=1)
    sweep_eng = ColoredBlockGibbsDecoder(mats, capture="sweep", **common)
    block_eng = ColoredBlockGibbsDecoder(mats, capture="block", **common)
    _, sweep_stats = sweep_eng.decode_with_stats(dets)
    block_pred, block_stats = block_eng.decode_with_stats(dets)
    assert block_pred.shape == (200, mats.num_observables)
    assert block_pred.dtype == np.uint8
    # Identical RNG trajectory: capture='block' inspects a superset of states
    # (after every color commit, not only sweep boundaries), so it can only find
    # at least as many valid corrections.
    assert block_stats.valid_fraction >= sweep_stats.valid_fraction


def test_default_legs_run_the_full_sweep_budget() -> None:
    circuit, mats = _rep_matrices(p=0.03)
    dets, _ = circuit.compile_detector_sampler(seed=0).sample(
        50, separate_observables=True
    )
    # Defaults (max_legs=1, target_valid=None) must reproduce the single fixed pass.
    eng = ColoredBlockGibbsDecoder(mats, chains=8, sweeps=20, seed=0)
    _, stats = eng.decode_with_stats(dets.astype(np.uint8))
    assert stats.mean_legs == 1.0
    assert stats.mean_sweeps == 20.0


def test_adaptive_legs_early_stop_executes_no_sweeps_when_already_valid() -> None:
    _, mats = _rep_matrices()
    eng = ColoredBlockGibbsDecoder(
        mats, chains=4, sweeps=50, target_valid=1, max_legs=3, seed=0
    )
    # All-zero syndrome: the e=0 start is already syndrome-consistent, so the leg
    # collects its valid member at sweep 0 and must not burn any of the 50 sweeps.
    dets = np.zeros((8, eng.m), dtype=np.uint8)
    predictions, stats = eng.decode_with_stats(dets)
    assert stats.valid_fraction == 1.0
    assert stats.mean_legs == 1.0
    assert stats.mean_sweeps == 0.0
    assert np.all(predictions == 0)


def test_more_legs_never_reduce_valid_fraction() -> None:
    circuit, mats = _rep_matrices(p=0.06)
    dets, _ = circuit.compile_detector_sampler(seed=0).sample(
        300, separate_observables=True
    )
    dets = dets.astype(np.uint8)
    common = dict(
        chains=6, sweeps=8, beta_start=0.5, beta_stop=3.0, target_valid=1, seed=5
    )
    one = ColoredBlockGibbsDecoder(mats, max_legs=1, **common)
    many = ColoredBlockGibbsDecoder(mats, max_legs=5, **common)
    _, one_stats = one.decode_with_stats(dets)
    _, many_stats = many.decode_with_stats(dets)
    # Leg 1 is RNG-identical for both engines; extra legs only retry shots that are
    # still unsolved (fresh chains), so they can only add valid shots, never remove.
    assert many_stats.valid_fraction >= one_stats.valid_fraction
    assert many_stats.mean_legs >= one_stats.mean_legs
    assert one_stats.mean_sweeps <= 8.0


def _cuda_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="requires cupy + a CUDA device")
def test_fused_gpu_sweep_matches_reference_cupy_path() -> None:
    import cupy as cp

    circuit, mats = _rep_matrices(p=0.05)
    dets, _ = circuit.compile_detector_sampler(seed=0).sample(
        128, separate_observables=True
    )
    dets = dets.astype(np.uint8)
    common = dict(
        chains=16, sweeps=30, beta_start=0.5, beta_stop=3.0,
        rho_scale=0.6, selection="min_weight", device="cuda", seed=1,
    )
    fused = ColoredBlockGibbsDecoder(mats, **common)
    ref = ColoredBlockGibbsDecoder(mats, **common)
    ref._fused = False  # reference cupy elementwise path
    # Identical device RNG streams + exact 0/1 residual sums => the fused kernel
    # reproduces the reference dynamics bit-for-bit.
    fused._device_rng = cp.random.default_rng(7)
    ref._device_rng = cp.random.default_rng(7)
    fused_pred, fused_stats = fused.decode_with_stats(dets)
    ref_pred, ref_stats = ref.decode_with_stats(dets)
    np.testing.assert_array_equal(fused_pred, ref_pred)
    assert fused_stats.valid_fraction == ref_stats.valid_fraction
    assert fused_stats.mean_selected_weight == ref_stats.mean_selected_weight


def test_decode_rejects_wrong_detector_count() -> None:
    _, mats = _rep_matrices()
    eng = ColoredBlockGibbsDecoder(mats, chains=2, sweeps=1, seed=0)
    with pytest.raises(ValueError, match="detectors"):
        eng.decode(np.zeros((1, eng.m + 1), dtype=np.uint8))


def test_active_region_defaults_are_noop() -> None:
    _, mats = _rep_matrices()
    eng = ColoredBlockGibbsDecoder(mats, chains=2, sweeps=1, gamma=1.0, seed=0)
    # Untouched knobs must leave the sweep in its uniform (pre-gating) form.
    assert eng._idle_gate is False
    assert eng._active_on is False
    assert eng.gamma_active == eng.gamma == eng.gamma_far
    # Explicitly passing the no-op values keeps the fast path disabled too.
    eng2 = ColoredBlockGibbsDecoder(
        mats, chains=2, sweeps=1, gamma=1.0,
        idle_update_rate=1.0, gamma_active=1.0, gamma_far=1.0, seed=0,
    )
    assert eng2._idle_gate is False
    assert eng2._active_on is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"idle_update_rate": 0.0},
        {"idle_update_rate": 1.5},
        {"gamma_active": 0.0},
        {"gamma_far": -1.0},
    ],
)
def test_active_region_param_validation(kwargs) -> None:
    _, mats = _rep_matrices()
    with pytest.raises(ValueError):
        ColoredBlockGibbsDecoder(mats, chains=2, sweeps=1, seed=0, **kwargs)


def test_active_region_noop_matches_baseline_cpu() -> None:
    # A decode with explicit no-op active-region params must be bit-for-bit
    # identical to the default construction (same RNG stream, same logic path).
    circuit, mats = _rep_matrices(p=0.05)
    dets, _ = circuit.compile_detector_sampler(seed=0).sample(
        200, separate_observables=True
    )
    dets = dets.astype(np.uint8)
    common = dict(chains=16, sweeps=25, beta_start=0.5, beta_stop=3.0, seed=3)
    base = ColoredBlockGibbsDecoder(mats, **common)
    noop = ColoredBlockGibbsDecoder(
        mats, idle_update_rate=1.0, gamma_active=1.0, gamma_far=1.0, **common
    )
    base_pred, base_stats = base.decode_with_stats(dets)
    noop_pred, noop_stats = noop.decode_with_stats(dets)
    np.testing.assert_array_equal(base_pred, noop_pred)
    assert base_stats.valid_fraction == noop_stats.valid_fraction
    assert base_stats.mean_selected_weight == noop_stats.mean_selected_weight


def test_active_region_residual_invariant_holds_across_sweeps() -> None:
    # The incremental residual/invalid bookkeeping in _sweep must stay exact even
    # with soft idle gating and active/far prior scaling engaged (states, residual
    # and invalid are all driven by the same post-gate flip mask).
    _, mats = _rep_matrices()
    eng = ColoredBlockGibbsDecoder(
        mats, chains=8, sweeps=20, beta_start=0.5, beta_stop=3.0,
        idle_update_rate=0.05, gamma_active=0.5, gamma_far=1.5, seed=1,
    )
    assert eng._idle_gate is True
    assert eng._active_on is True
    rng = np.random.default_rng(2)
    rows = 8
    target = (rng.random(eng.m) < 0.3).astype(np.uint8)
    states = np.zeros((eng.n, rows), dtype=np.uint8)
    residual = np.repeat(target[:, None], rows, axis=1).copy()
    prior_energy = np.zeros(rows, dtype=np.float64)
    invalid = residual.sum(axis=0, dtype=np.int32)
    h = eng._check_matrix
    for beta in eng._beta_schedule():
        eng._sweep(states, residual, prior_energy, invalid, float(beta), rng)
        expected = (np.asarray((h @ states), dtype=np.int64) % 2).astype(np.uint8)
        expected ^= target[:, None]
        np.testing.assert_array_equal(residual, expected)
        np.testing.assert_array_equal(invalid, residual.sum(axis=0, dtype=np.int32))


def test_active_region_decode_is_valid_and_low_ler() -> None:
    circuit, mats = _rep_matrices(p=0.03)
    dets, obs = circuit.compile_detector_sampler(seed=0).sample(
        500, separate_observables=True
    )
    eng = ColoredBlockGibbsDecoder(
        mats, chains=24, sweeps=60, beta_start=0.5, beta_stop=3.0,
        idle_update_rate=0.05, gamma_active=0.5, gamma_far=1.5, seed=1,
    )
    predictions, stats = eng.decode_with_stats(dets.astype(np.uint8))
    assert predictions.shape == (500, mats.num_observables)
    assert stats.valid_fraction == 1.0
    ler = float(np.mean(np.any(predictions != obs.astype(np.uint8), axis=1)))
    assert ler < 0.05


@pytest.mark.skipif(not _cuda_available(), reason="requires cupy + a CUDA device")
def test_fused_gpu_sweep_matches_reference_with_active_region() -> None:
    import cupy as cp

    circuit, mats = _rep_matrices(p=0.05)
    dets, _ = circuit.compile_detector_sampler(seed=0).sample(
        128, separate_observables=True
    )
    dets = dets.astype(np.uint8)
    common = dict(
        chains=16, sweeps=30, beta_start=0.5, beta_stop=3.0,
        rho_scale=0.6, selection="min_weight", device="cuda", seed=1,
        idle_update_rate=0.05, gamma_active=0.5, gamma_far=1.5,
    )
    fused = ColoredBlockGibbsDecoder(mats, **common)
    ref = ColoredBlockGibbsDecoder(mats, **common)
    ref._fused = False  # reference cupy elementwise path (same active-region math)
    fused._device_rng = cp.random.default_rng(7)
    ref._device_rng = cp.random.default_rng(7)
    fused_pred, fused_stats = fused.decode_with_stats(dets)
    ref_pred, ref_stats = ref.decode_with_stats(dets)
    np.testing.assert_array_equal(fused_pred, ref_pred)
    assert fused_stats.valid_fraction == ref_stats.valid_fraction
    assert fused_stats.mean_selected_weight == ref_stats.mean_selected_weight
