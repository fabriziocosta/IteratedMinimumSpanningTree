import numpy as np
import pytest
import torch

from imste._optimizer import (
    _attraction_factor,
    _repulsion_factor,
    active_edges_for_epoch,
    expected_active_edge_count,
    optimize_embedding,
    rank_phases,
)
from imste._optimizer_torch import _complement_target_from_ordinal


def test_attraction_gradient_matches_torch_autograd():
    for coordinates, epsilon in [
        ([[0.0, 0.0], [1e-9, -1e-9]], 1e-8),
        ([[0.2, -0.7, 1.1], [2.0, 3.0, -1.0]], 1e-4),
        ([[1.0, 2.0], [100.0, -200.0]], 0.03),
    ]:
        points = torch.tensor(coordinates, dtype=torch.float64, requires_grad=True)
        delta = points[0] - points[1]
        loss = torch.log1p(torch.sum(delta * delta) + epsilon)
        gradient = torch.autograd.grad(loss, points)[0][0].numpy()
        d2 = float(np.dot(delta.detach().numpy(), delta.detach().numpy()))
        explicit = _attraction_factor(d2, epsilon) * delta.detach().numpy()
        np.testing.assert_allclose(explicit, gradient, rtol=1e-12, atol=1e-12)


def test_repulsion_gradient_matches_torch_autograd():
    for coordinates, margin, temperature in [
        ([[0.0, 0.0], [1e-8, 0.0]], 1.0, 0.2),
        ([[0.2, -0.7, 1.1], [2.0, 3.0, -1.0]], 3.0, 0.7),
        ([[1.0, 2.0], [100.0, -200.0]], 0.1, 0.05),
    ]:
        points = torch.tensor(coordinates, dtype=torch.float64, requires_grad=True)
        delta = points[0] - points[1]
        d2 = torch.sum(delta * delta)
        loss = torch.nn.functional.softplus((margin - d2) / temperature)
        gradient = torch.autograd.grad(loss, points)[0][0].numpy()
        explicit = (
            _repulsion_factor(float(d2.detach()), margin, temperature)
            * -delta.detach().numpy()
        )
        np.testing.assert_allclose(explicit, gradient, rtol=1e-12, atol=1e-12)


def test_rank_scheduler_has_exact_periodic_counts():
    ranks = np.repeat(np.arange(1, 11, dtype=np.int32), 3)
    phases = rank_phases(ranks, random_state=71)
    counts = np.zeros(len(ranks), dtype=int)
    for epoch in range(100):
        counts[active_edges_for_epoch(ranks, epoch, phases)] += 1
    for rank in range(1, 11):
        rank_counts = counts[ranks == rank]
        assert np.unique(rank_counts).size == 1
        assert rank_counts[0] in {100 // rank, (100 + rank - 1) // rank}
    np.testing.assert_array_equal(phases, rank_phases(ranks, random_state=71))


def test_expected_active_count_matches_inverse_rank_objective_normalizer():
    ranks = np.repeat(np.arange(1, 6, dtype=np.int32), 9)
    expected = 9 * np.sum(1.0 / np.arange(1, 6))
    assert np.isclose(expected_active_edge_count(ranks), expected)


def test_rank_scheduled_multipliers_use_weighted_inverse_rank_normalizer(monkeypatch):
    import imste._optimizer as optimizer_module

    observed = {}

    def capture_updates(*args):
        observed["expected_active_edges"] = args[7]
        observed["edge_weights"] = args[15]
        observed["use_edge_weights"] = args[16]

    monkeypatch.setattr(optimizer_module, "_apply_epoch_updates", capture_updates)
    ranks = np.array([1, 2, 3], dtype=np.int32)
    multipliers = np.array([1.0, 2.0, 0.5])
    optimize_embedding(
        np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
        np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
        ranks,
        n_epochs=1,
        learning_rate=0.1,
        negative_ratio=0,
        lambda_rep=0.0,
        epsilon=1e-4,
        margin=1.0,
        temperature=0.5,
        random_state=3,
        schedule_by_rank=True,
        edge_weights=multipliers,
    )

    assert observed["expected_active_edges"] == pytest.approx(
        np.sum(multipliers / ranks)
    )
    np.testing.assert_array_equal(observed["edge_weights"], multipliers)
    assert observed["use_edge_weights"] is True


def test_dense_neighbor_complement_ordinals_skip_self_and_blocked_nodes():
    n_samples = 7
    source = 3
    blocked = np.array([0, 1, 4, 6], dtype=np.int64)
    adjacency_keys = np.sort(
        np.concatenate(
            [
                np.array([source * n_samples + target for target in blocked]),
                np.array([target * n_samples + source for target in blocked]),
            ]
        )
    )
    assert _complement_target_from_ordinal(source, adjacency_keys, n_samples, 0) == 2
    assert _complement_target_from_ordinal(source, adjacency_keys, n_samples, 1) == 5


def test_single_sgd_update_matches_explicit_gradient_and_recenter():
    embedding = np.array([[0.2, -0.4], [1.3, 0.5]], dtype=np.float32)
    initial = embedding.astype(np.float64)
    delta = initial[0] - initial[1]
    d2 = float(np.dot(delta, delta))
    learning_rate = 0.1
    lambda_rep = 0.25
    epsilon = 0.01
    factor = (
        learning_rate
        * 5
        * len(embedding)
        * (1.0 - lambda_rep)
        * _attraction_factor(d2, epsilon)
    )
    expected = initial.copy()
    expected[0] -= factor * delta
    expected[1] += factor * delta
    expected -= expected.mean(axis=0)

    optimize_embedding(
        embedding,
        np.array([[0, 1]], dtype=np.intp),
        np.array([1], dtype=np.int32),
        n_epochs=1,
        learning_rate=learning_rate,
        negative_ratio=5,
        lambda_rep=lambda_rep,
        epsilon=epsilon,
        margin=1.0,
        temperature=0.5,
        random_state=4,
    )
    np.testing.assert_allclose(embedding, expected, rtol=1e-6, atol=1e-7)


def test_exact_prim_uses_all_edges_with_explicit_inverse_rank_weights():
    embedding = np.array(
        [[0.0, 0.0], [1.0, 0.5], [2.0, -1.0], [2.5, 1.0]], dtype=np.float32
    )
    initial = embedding.astype(np.float64)
    edges = np.array([[0, 1], [2, 3]], dtype=np.intp)
    ranks = np.array([1, 2], dtype=np.int32)
    learning_rate = 0.01
    normalizer = 1.0 + 0.5
    expected = initial.copy()
    alpha = learning_rate * 5 * len(embedding)
    for (left, right), rank in zip(edges, ranks):
        delta = expected[left] - expected[right]
        d2 = float(np.dot(delta, delta))
        factor = (
            alpha * (1.0 / normalizer) * (1.0 / rank) * _attraction_factor(d2, 1e-4)
        )
        expected[left] -= factor * delta
        expected[right] += factor * delta
    expected -= expected.mean(axis=0)

    optimize_embedding(
        embedding,
        edges,
        ranks,
        n_epochs=1,
        learning_rate=learning_rate,
        negative_ratio=0,
        lambda_rep=0.0,
        epsilon=1e-4,
        margin=1.0,
        temperature=0.5,
        random_state=4,
        schedule_by_rank=False,
    )
    np.testing.assert_allclose(embedding, expected, rtol=1e-6, atol=1e-7)


def test_optimizer_accepts_direct_positive_weights_normalized_by_total():
    embedding = np.array(
        [[0.0, 0.0], [1.0, 0.5], [2.0, -1.0], [-1.0, 2.0]], dtype=np.float32
    )
    initial = embedding.astype(np.float64)
    edges = np.array([[0, 1], [2, 3]], dtype=np.intp)
    weights = np.array([2.0, 1.0])
    expected = initial.copy()
    alpha = 0.01 * 5 * len(embedding)
    for (left, right), weight in zip(edges, weights):
        delta = expected[left] - expected[right]
        d2 = float(np.dot(delta, delta))
        factor = alpha * (weight / weights.sum()) * _attraction_factor(d2, 1e-4)
        expected[left] -= factor * delta
        expected[right] += factor * delta
    expected -= expected.mean(axis=0)

    optimize_embedding(
        embedding,
        edges,
        np.ones(len(edges), dtype=np.int32),
        n_epochs=1,
        learning_rate=0.01,
        negative_ratio=0,
        lambda_rep=0.0,
        epsilon=1e-4,
        margin=1.0,
        temperature=0.5,
        random_state=4,
        schedule_by_rank=False,
        edge_weights=weights,
    )
    np.testing.assert_allclose(embedding, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_mps_optimizer_accepts_direct_edge_weights():
    from imste._optimizer_mps import optimize_embedding_mps

    embedding = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, -1.0]], dtype=np.float32)
    optimize_embedding_mps(
        embedding,
        np.array([[0, 1], [1, 2]], dtype=np.int64),
        np.ones(2, dtype=np.int32),
        n_epochs=2,
        batch_size=2,
        learning_rate=0.001,
        negative_ratio=0,
        lambda_rep=0.0,
        epsilon=1e-4,
        margin=1.0,
        temperature=0.5,
        random_state=9,
        schedule_by_rank=False,
        edge_weights=np.array([2.0, 1.0]),
    )
    assert np.isfinite(embedding).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_optimizer_accepts_direct_edge_weights():
    from imste._optimizer_torch import optimize_embedding_torch

    embedding = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, -1.0]], dtype=np.float32)
    optimize_embedding_torch(
        embedding,
        np.array([[0, 1], [1, 2]], dtype=np.int64),
        np.ones(2, dtype=np.int32),
        device_type="cuda",
        n_epochs=2,
        batch_size=2,
        learning_rate=0.001,
        negative_ratio=0,
        lambda_rep=0.0,
        epsilon=1e-4,
        margin=1.0,
        temperature=0.5,
        random_state=9,
        schedule_by_rank=False,
        edge_weights=np.array([2.0, 1.0]),
    )
    assert np.isfinite(embedding).all()


def test_optimizer_is_seeded_and_supports_arbitrary_dimensions():
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.intp)
    ranks = np.array([1, 1, 2], dtype=np.int32)
    outputs = []
    for _ in range(2):
        embedding = np.random.RandomState(8).standard_normal((4, 5)).astype(np.float32)
        optimize_embedding(
            embedding,
            edges,
            ranks,
            n_epochs=5,
            learning_rate=0.03,
            negative_ratio=2,
            lambda_rep=0.5,
            epsilon=1e-4,
            margin=1.0,
            temperature=0.5,
            random_state=12,
        )
        outputs.append(embedding)
    np.testing.assert_array_equal(outputs[0], outputs[1])
    assert np.isfinite(outputs[0]).all()
