from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune_tabldm import (  # noqa: E402
    Episode,
    episode_batch_key,
    episode_batch_tensors,
    episode_batches,
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
