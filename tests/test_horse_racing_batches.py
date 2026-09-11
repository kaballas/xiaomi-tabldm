from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune_tabldm import (  # noqa: E402
    CachedEpisode,
    Episode,
    cache_episode,
    episode_batch_key,
    episode_batch_tensors,
    episode_batches,
    materialize_episode,
    optimizer_step,
    prepare_episodes,
    set_gradient_checkpointing,
    task_loss,
)


def make_episode(race_id, context_rows, query_rows, features, offset=0):
    context_values = np.arange(context_rows * features, dtype=np.float32).reshape(
        context_rows, features
    )
    query_values = np.arange(query_rows * features, dtype=np.float32).reshape(
        query_rows, features
    )
    return Episode(
        race_id=str(race_id),
        X_context=context_values + offset,
        y_context=np.arange(context_rows, dtype=np.int64) % 2,
        context_sizes=(context_rows,),
        X_query=query_values + offset + 100,
        y_query=np.arange(query_rows, dtype=np.int64) % 2,
        winner_query=np.eye(1, query_rows, dtype=np.int64).reshape(-1),
    )


def test_episode_batches_keep_shapes_separate_and_all_episodes():
    episodes = [
        make_episode("a", 4, 3, 2),
        make_episode("b", 4, 3, 2),
        make_episode("c", 5, 3, 2),
        make_episode("d", 4, 3, 2),
    ]

    batches = episode_batches(episodes, batch_size=2)

    assert sorted(index for batch in batches for index in batch) == list(range(len(episodes)))
    assert all(
        len({episode_batch_key(episodes[index]) for index in batch}) == 1
        for batch in batches
    )
    assert sorted(map(len, batches)) == [1, 1, 2]


def test_episode_batch_tensors_stack_on_batch_dimension():
    episodes = [make_episode("a", 4, 3, 2), make_episode("b", 4, 3, 2, offset=10)]

    X, y_context, y_query = episode_batch_tensors(episodes, torch.device("cpu"))

    assert X.shape == (2, 7, 2)
    assert y_context.shape == (2, 4)
    assert y_query.shape == (2, 3)
    np.testing.assert_array_equal(X[0, :4].numpy(), episodes[0].X_context)
    np.testing.assert_array_equal(X[1, 4:].numpy(), episodes[1].X_query)


def test_cached_episode_round_trip_and_batch_loading(tmp_path):
    episode = make_episode("cached", 4, 3, 2)

    cached = cache_episode(episode, tmp_path, 7)

    assert isinstance(cached, CachedEpisode)
    assert cached.feature_path == tmp_path / "00000007.npy"
    assert episode_batch_key(cached) == (4, 3, 2)
    restored = materialize_episode(cached)
    np.testing.assert_array_equal(restored.X_context, episode.X_context)
    np.testing.assert_array_equal(restored.X_query, episode.X_query)
    X, y_context, y_query = episode_batch_tensors([cached], torch.device("cpu"))
    np.testing.assert_array_equal(X[0, :4].numpy(), episode.X_context)
    np.testing.assert_array_equal(y_context[0].numpy(), episode.y_context)
    np.testing.assert_array_equal(y_query[0].numpy(), episode.y_query)


def test_prepare_episodes_retains_only_cached_references(tmp_path, monkeypatch):
    import finetune_tabldm

    prepared = [make_episode(index, 4, 3, 2, offset=index) for index in range(3)]
    specs = [((), index) for index in range(3)]

    monkeypatch.setattr(
        finetune_tabldm,
        "prepare_episode",
        lambda _context, query, _features: prepared[query],
    )
    episodes = prepare_episodes(
        specs,
        ["feature"],
        "test",
        workers=1,
        cache_dir=tmp_path,
    )

    assert all(isinstance(episode, CachedEpisode) for episode in episodes)
    assert [episode.race_id for episode in episodes] == ["0", "1", "2"]
    assert len(list(tmp_path.glob("*.npy"))) == 3


def test_gradient_checkpointing_toggle_reaches_nested_encoders():
    class RecomputingModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.recompute = False

    model = torch.nn.Sequential(
        RecomputingModule(),
        torch.nn.Sequential(RecomputingModule()),
    )

    assert set_gradient_checkpointing(model, True) == 2
    assert model[0].recompute is True
    assert model[1][0].recompute is True
    assert set_gradient_checkpointing(model, False) == 2
    assert model[0].recompute is False
    assert model[1][0].recompute is False


def test_optimizer_step_unscales_before_clipping_and_stepping():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([8.0])
    optimizer = torch.optim.SGD([parameter], lr=0.5)

    class RecordingScaler:
        def __init__(self):
            self.calls = []

        def unscale_(self, received_optimizer):
            assert received_optimizer is optimizer
            self.calls.append("unscale")
            parameter.grad.div_(2)

        def step(self, received_optimizer):
            self.calls.append("step")
            received_optimizer.step()

        def update(self):
            self.calls.append("update")

    scaler = RecordingScaler()
    optimizer_step(
        optimizer,
        [parameter],
        accumulated=4,
        grad_clip=0,
        grad_scaler=scaler,
    )

    assert scaler.calls == ["unscale", "step", "update"]
    assert parameter.item() == pytest.approx(0.5)
    assert parameter.grad is None


@pytest.mark.parametrize("listwise_weight", [0.0, 0.25])
def test_batched_task_loss_matches_mean_serial_loss(listwise_weight):
    logits = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 2.0], [1.0, 0.0]],
            [[0.0, 2.0], [2.0, 0.0], [0.0, 1.0]],
        ]
    )
    targets = torch.tensor([[0, 1, 0], [1, 0, 1]])

    batched = task_loss(logits, targets, listwise_weight)
    serial = [
        task_loss(logits[index : index + 1], targets[index : index + 1], listwise_weight)
        for index in range(2)
    ]

    for batch_value, serial_values in zip(batched, zip(*serial)):
        assert torch.allclose(batch_value, torch.stack(serial_values).mean())
