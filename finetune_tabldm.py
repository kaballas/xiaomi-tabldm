"""Race-aware downstream fine-tuning for the Xiaomi TabLDM classifier.

Each episode contains K complete earlier races as labelled in-context rows and
one later complete race as query rows. Race IDs define chronological groups but
are never passed to the model as features.
"""

from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import multiprocessing
from pathlib import Path
import json
import os
import random
import tempfile
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from horse_racing_dataset import resolve_race_csvs
from horse_racing_preprocessing import preprocess_episode_features
from tabldm import InferenceConfig, TabLDMClassifier
from tutorials.horse_racing_top3 import read_feature_columns


ROOT = Path(__file__).resolve().parent
TARGET = "top3_mask"


@dataclass
class Race:
    race_id: str
    start_time: pd.Timestamp
    frame: pd.DataFrame
    y: np.ndarray
    winner: np.ndarray


@dataclass
class Episode:
    race_id: str
    X_context: np.ndarray
    y_context: np.ndarray
    context_sizes: tuple[int, ...]
    X_query: np.ndarray
    y_query: np.ndarray
    winner_query: np.ndarray


@dataclass
class CachedEpisode:
    """Lightweight reference to feature arrays stored outside process memory."""

    race_id: str
    feature_path: Path
    context_rows: int
    query_rows: int
    feature_count: int
    y_context: np.ndarray
    context_sizes: tuple[int, ...]
    y_query: np.ndarray
    winner_query: np.ndarray


def positive_integer(value):
    value = int(value)
    if value < 1:
        raise ValueError("value must be at least 1")
    return value


def positive_float(value):
    value = float(value)
    if value <= 0:
        raise ValueError("value must be greater than zero")
    return value


def nonnegative_float(value):
    value = float(value)
    if value < 0:
        raise ValueError("value must be non-negative")
    return value


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=os.environ.get("TABLDM_CLF_CKPT"))
    parser.add_argument(
        "--dataset",
        default="Kaballas/races",
        help="Hugging Face dataset repository containing training.csv, validation.csv, and test.csv",
    )
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help="optional Hugging Face branch, tag, or commit (defaults to the repository's main branch)",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="local training CSV override (requires --validation-csv)",
    )
    parser.add_argument(
        "--validation-csv",
        type=Path,
        default=None,
        help="local validation CSV override (requires --train-csv)",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=None,
        help="local test CSV override used with --evaluate-test",
    )
    parser.add_argument("--features-json", type=Path, default=ROOT / "a.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/tabldm_horse_finetuned.ckpt")
    parser.add_argument("--finetune-mode", choices=("decoder", "icl", "row_icl", "full"), default="decoder")
    parser.add_argument("--context-races", type=positive_integer, default=10)
    parser.add_argument("--epochs", type=positive_integer, default=20)
    parser.add_argument("--learning-rate", type=positive_float, default=None)
    parser.add_argument("--weight-decay", type=nonnegative_float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=positive_integer, default=4)
    parser.add_argument("--grad-clip", type=nonnegative_float, default=1.0)
    parser.add_argument("--listwise-weight", type=nonnegative_float, default=0.0)
    parser.add_argument("--moe-aux-weight", type=nonnegative_float, default=1.0)
    parser.add_argument("--patience", type=positive_integer, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp",
        action=BooleanOptionalAction,
        default=None,
        help="use CUDA FP16 mixed precision (default: enabled on CUDA)",
    )
    parser.add_argument("--num-threads", type=positive_integer, default=None)
    parser.add_argument(
        "--preprocessing-workers",
        type=positive_integer,
        default=min(4, os.cpu_count() or 1),
        help="CPU processes used to prepare race episodes",
    )
    parser.add_argument("--max-train-races", type=positive_integer, default=None)
    parser.add_argument("--max-validation-races", type=positive_integer, default=None)
    parser.add_argument("--max-test-races", type=positive_integer, default=None)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="evaluate the sealed test split after selecting the best checkpoint",
    )
    parser.add_argument("--no-runner-shuffle", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def load_races(csv_path, feature_columns):
    required = {"race_id", "start_time_iso", "runner_mask", TARGET, "is_winner", *feature_columns}
    columns = pd.read_csv(csv_path, nrows=0).columns
    missing = sorted(required.difference(columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")
    frame = pd.read_csv(csv_path, low_memory=False, usecols=required)

    races = []
    for race_id, group in frame.groupby("race_id", sort=False):
        active = group.loc[group["runner_mask"].eq(1)].copy().reset_index(drop=True)
        if active.empty or active[TARGET].isna().any():
            continue
        labels = active[TARGET].astype(int).to_numpy()
        winner = active["is_winner"].astype(int).to_numpy()
        if set(np.unique(labels)) != {0, 1} or int(labels.sum()) != 3 or int(winner.sum()) != 1:
            continue
        start_time = pd.to_datetime(active["start_time_iso"].iloc[0], utc=True, errors="raise")
        races.append(
            Race(
                str(race_id),
                start_time,
                active.loc[:, feature_columns].copy(),
                labels,
                winner,
            )
        )

    races.sort(key=lambda race: (race.start_time, race.race_id))
    if not races:
        raise ValueError(f"No eligible labelled races found in {csv_path}")
    return races, columns


def validate_chronology(train_races, validation_races, test_races=None):
    if train_races[-1].start_time >= validation_races[0].start_time:
        raise ValueError("Training races are not strictly earlier than validation races")
    if test_races is not None and validation_races[-1].start_time >= test_races[0].start_time:
        raise ValueError("Validation races are not strictly earlier than test races")


def episode_specs(query_races, initial_history, context_races, limit=None):
    history = list(initial_history)
    specs = []
    position = 0
    while position < len(query_races):
        start_time = query_races[position].start_time
        end = position
        while end < len(query_races) and query_races[end].start_time == start_time:
            end += 1
        simultaneous_races = query_races[position:end]
        for query in simultaneous_races:
            if len(history) >= context_races:
                specs.append((tuple(history[-context_races:]), query))
                if limit is not None and len(specs) >= limit:
                    return specs
        # Results from simultaneous races become history only after all queries
        # at that start time have been constructed.
        history.extend(simultaneous_races)
        position = end
    return specs


def prepare_episode(context_races, query_race, feature_columns):
    context_frame = pd.concat([race.frame for race in context_races], ignore_index=True)
    context_y = np.concatenate([race.y for race in context_races])
    context_sizes = tuple(len(race.frame) for race in context_races)

    # Fit encoding, constant filtering, scaling, and outlier handling on the
    # labelled context only, then apply the same transformations to the query.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The following categorical columns have a cardinality above 40")
        warnings.filterwarnings("ignore", message="Skipping features without any observed values")
        X_all = preprocess_episode_features(
            context_frame,
            query_race.frame,
            feature_columns,
        )
    y_context = np.asarray(context_y, dtype=np.int64)

    if not np.isfinite(X_all).all():
        raise ValueError(f"Preprocessing produced non-finite values for query race {query_race.race_id}")
    split = len(context_y)
    return Episode(
        race_id=query_race.race_id,
        X_context=X_all[:split],
        y_context=y_context,
        context_sizes=context_sizes,
        X_query=X_all[split:],
        y_query=query_race.y.astype(np.int64),
        winner_query=query_race.winner.astype(np.int64),
    )


_PREPROCESSING_SPECS = None
_PREPROCESSING_FEATURE_COLUMNS = None
_PREPROCESSING_CACHE_DIR = None


def cache_episode(episode, cache_dir, index):
    """Persist the large feature matrix and retain only small episode metadata."""
    feature_path = Path(cache_dir) / f"{index:08d}.npy"
    features = np.concatenate([episode.X_context, episode.X_query], axis=0)
    np.save(feature_path, features, allow_pickle=False)
    return CachedEpisode(
        race_id=episode.race_id,
        feature_path=feature_path,
        context_rows=len(episode.X_context),
        query_rows=len(episode.X_query),
        feature_count=episode.X_context.shape[1],
        y_context=episode.y_context,
        context_sizes=episode.context_sizes,
        y_query=episode.y_query,
        winner_query=episode.winner_query,
    )


def materialize_episode(episode):
    """Load one cached episode lazily for the duration of a minibatch."""
    if not isinstance(episode, CachedEpisode):
        return episode
    features = np.load(episode.feature_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (
        episode.context_rows + episode.query_rows,
        episode.feature_count,
    )
    if features.shape != expected_shape:
        raise ValueError(
            f"Cached episode {episode.race_id} has shape {features.shape}, "
            f"expected {expected_shape}"
        )
    return Episode(
        race_id=episode.race_id,
        X_context=features[: episode.context_rows],
        y_context=episode.y_context,
        context_sizes=episode.context_sizes,
        X_query=features[episode.context_rows :],
        y_query=episode.y_query,
        winner_query=episode.winner_query,
    )


def _prepare_episode_at_index(index):
    context, query = _PREPROCESSING_SPECS[index]
    episode = prepare_episode(context, query, _PREPROCESSING_FEATURE_COLUMNS)
    if _PREPROCESSING_CACHE_DIR is not None:
        return cache_episode(episode, _PREPROCESSING_CACHE_DIR, index)
    return episode


def prepare_episodes(specs, feature_columns, label, workers=1, cache_dir=None):
    if not specs:
        return []
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    worker_count = min(workers, len(specs))
    if worker_count > 1:
        try:
            context = multiprocessing.get_context("fork")
        except ValueError:
            worker_count = 1

    if worker_count == 1:
        def serial_iterator():
            for index, (context_races, query_race) in enumerate(specs):
                episode = prepare_episode(context_races, query_race, feature_columns)
                if cache_dir is not None:
                    episode = cache_episode(episode, cache_dir, index)
                yield episode

        iterator = serial_iterator()
        pool = None
    else:
        global _PREPROCESSING_SPECS, _PREPROCESSING_FEATURE_COLUMNS, _PREPROCESSING_CACHE_DIR
        _PREPROCESSING_SPECS = specs
        _PREPROCESSING_FEATURE_COLUMNS = feature_columns
        _PREPROCESSING_CACHE_DIR = cache_dir
        pool = context.Pool(processes=worker_count)
        # Episode payloads are large; small chunks keep worker transients bounded.
        chunk_size = max(1, min(4, len(specs) // (worker_count * 8)))
        iterator = pool.imap(_prepare_episode_at_index, range(len(specs)), chunksize=chunk_size)

    episodes = []
    completed = False
    try:
        for index, episode in enumerate(iterator, start=1):
            episodes.append(episode)
            if index % 50 == 0 or index == len(specs):
                print(f"Prepared {label} episodes: {index}/{len(specs)}")
        completed = True
    finally:
        if pool is not None:
            if completed:
                pool.close()
            else:
                pool.terminate()
            pool.join()
            _PREPROCESSING_SPECS = None
            _PREPROCESSING_FEATURE_COLUMNS = None
            _PREPROCESSING_CACHE_DIR = None
    return episodes


def load_model(checkpoint_path, device):
    loader = TabLDMClassifier(
        model_path=checkpoint_path,
        checkpoint_version="checkpoints/clf_default.ckpt",
        allow_auto_download=True,
        device=device,
    )
    loader._load_model()
    model = loader.model_.to(device)
    source_path = Path(loader.model_path_)
    source_checkpoint = torch.load(source_path, map_location="cpu", weights_only=True)
    return model, source_checkpoint, source_path


def configure_finetuning(model, mode):
    model.requires_grad_(False)
    if mode == "decoder":
        modules = [model.icl_predictor.decoder]
    elif mode == "icl":
        modules = [model.icl_predictor]
    elif mode == "row_icl":
        modules = [model.row_interactor, model.icl_predictor]
    else:
        modules = [model]
    for module in modules:
        module.requires_grad_(True)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(f"Fine-tune mode {mode} selected no parameters")
    return trainable


def shuffled_episode_arrays(episode, rng, shuffle_runners):
    if not shuffle_runners:
        return episode.X_context, episode.y_context, episode.X_query, episode.y_query

    context_indices = []
    offset = 0
    for size in episode.context_sizes:
        context_indices.extend((offset + rng.permutation(size)).tolist())
        offset += size
    query_indices = rng.permutation(len(episode.y_query))
    return (
        episode.X_context[context_indices],
        episode.y_context[context_indices],
        episode.X_query[query_indices],
        episode.y_query[query_indices],
    )


def episode_tensors(episode, device, rng=None, shuffle_runners=False):
    return episode_batch_tensors([episode], device, rng, shuffle_runners)


def episode_batch_key(episode):
    """Return dimensions that must match before episodes can be stacked."""
    if isinstance(episode, CachedEpisode):
        return episode.context_rows, episode.query_rows, episode.feature_count
    if episode.X_context.ndim != 2 or episode.X_query.ndim != 2:
        raise ValueError("Episode feature arrays must be two-dimensional")
    if episode.X_context.shape[1] != episode.X_query.shape[1]:
        raise ValueError(f"Episode {episode.race_id} has mismatched context/query features")
    return (
        len(episode.X_context),
        len(episode.X_query),
        episode.X_context.shape[1],
    )


def episode_batches(episodes, batch_size, order=None):
    """Group episode indices into shape-compatible minibatches."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if order is None:
        order = range(len(episodes))

    buckets = {}
    for episode_index in order:
        key = episode_batch_key(episodes[episode_index])
        buckets.setdefault(key, []).append(int(episode_index))

    batches = []
    for indices in buckets.values():
        batches.extend(
            indices[start : start + batch_size]
            for start in range(0, len(indices), batch_size)
        )
    return batches


def episode_batch_tensors(episodes, device, rng=None, shuffle_runners=False):
    """Stack a shape-compatible group of episodes and move it to a device."""
    episodes = [materialize_episode(episode) for episode in episodes]
    arrays = [
        shuffled_episode_arrays(episode, rng, shuffle_runners)
        for episode in episodes
    ]
    X = torch.from_numpy(
        np.stack(
            [np.concatenate([X_context, X_query], axis=0) for X_context, _, X_query, _ in arrays]
        )
    ).to(device)
    y_context_tensor = torch.from_numpy(
        np.stack([y_context for _, y_context, _, _ in arrays])
    ).to(device=device, dtype=torch.float32)
    y_query_tensor = torch.from_numpy(
        np.stack([y_query for _, _, _, y_query in arrays])
    ).to(device=device, dtype=torch.long)
    return X, y_context_tensor, y_query_tensor


def task_loss(logits, targets, listwise_weight):
    active_logits = logits[..., :2]
    cross_entropy = F.cross_entropy(active_logits.reshape(-1, 2), targets.reshape(-1))
    if listwise_weight == 0:
        return cross_entropy, cross_entropy, cross_entropy.new_zeros(())

    positive_scores = active_logits[..., 1] - active_logits[..., 0]
    target_distribution = targets.float() / targets.sum(dim=1, keepdim=True).clamp_min(1)
    listwise = -(target_distribution * F.log_softmax(positive_scores, dim=1)).sum(dim=1).mean()
    return cross_entropy + listwise_weight * listwise, cross_entropy, listwise


def set_gradient_checkpointing(model, enabled):
    """Toggle recomputation on every encoder that exposes the setting."""
    changed = 0
    for module in model.modules():
        if hasattr(module, "recompute"):
            module.recompute = enabled
            changed += 1
    return changed


def make_grad_scaler(enabled):
    """Construct a CUDA scaler across supported PyTorch AMP API versions."""
    if not enabled:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=True)


def optimizer_step(
    optimizer,
    trainable_parameters,
    accumulated,
    grad_clip,
    grad_scaler=None,
):
    if grad_scaler is not None:
        grad_scaler.unscale_(optimizer)
    for parameter in trainable_parameters:
        if parameter.grad is not None:
            parameter.grad.div_(accumulated)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip)
    if grad_scaler is None:
        optimizer.step()
    else:
        grad_scaler.step(optimizer)
        grad_scaler.update()
    optimizer.zero_grad(set_to_none=True)


def train_epoch(
    model,
    episodes,
    optimizer,
    trainable_parameters,
    args,
    epoch,
    grad_scaler=None,
):
    model.train()
    if args.resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    checkpointed_modules = set_gradient_checkpointing(
        model, args.finetune_mode != "decoder"
    )
    optimizer.zero_grad(set_to_none=True)
    rng = np.random.default_rng(args.seed + epoch)
    order = rng.permutation(len(episodes))
    batch_size = getattr(args, "batch_size", 1)
    if batch_size == 1:
        # Preserve the original seeded episode and runner-shuffle order.
        batches = [[int(index)] for index in order]
    else:
        batches = episode_batches(episodes, batch_size, order)
        batches = [batches[index] for index in rng.permutation(len(batches))]
    if epoch == 1:
        mean_batch_size = len(episodes) / len(batches)
        print(
            f"Shape-compatible training batches: {len(episodes)} episodes -> "
            f"{len(batches)} batches (mean={mean_batch_size:.2f}, "
            f"maximum={batch_size})"
        )
        print(
            f"Gradient checkpointing: "
            f"{'enabled' if args.finetune_mode != 'decoder' else 'disabled'} "
            f"for {checkpointed_modules} modules"
        )
    metric_totals = {
        "objective": None,
        "cross_entropy": None,
        "listwise": None,
        "moe_aux": None,
    }
    accumulated = 0

    for position, batch_indices in enumerate(batches, start=1):
        batch_episodes = [episodes[index] for index in batch_indices]
        X, y_context, y_query = episode_batch_tensors(
            batch_episodes,
            args.resolved_device,
            rng=rng,
            shuffle_runners=not args.no_runner_shuffle,
        )
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=getattr(args, "amp", False),
        ):
            logits = model(X, y_train=y_context, embed_with_test=False)
            loss, cross_entropy, listwise = task_loss(
                logits, y_query, args.listwise_weight
            )
            auxiliary = loss.new_zeros(())
            if args.finetune_mode != "decoder" and args.moe_aux_weight > 0:
                auxiliary = model.moe_aux_loss()
                loss = loss + args.moe_aux_weight * auxiliary
        if not torch.isfinite(loss):
            race_ids = ", ".join(episode.race_id for episode in batch_episodes)
            raise RuntimeError(f"Non-finite training loss for query race batch: {race_ids}")

        episodes_in_batch = len(batch_episodes)
        scaled_loss = loss * episodes_in_batch
        if grad_scaler is None:
            scaled_loss.backward()
        else:
            grad_scaler.scale(scaled_loss).backward()
        for name, value in {
            "objective": loss,
            "cross_entropy": cross_entropy,
            "listwise": listwise,
            "moe_aux": auxiliary,
        }.items():
            value = value.detach().float() * episodes_in_batch
            if metric_totals[name] is None:
                metric_totals[name] = value
            else:
                metric_totals[name].add_(value)
        accumulated += episodes_in_batch

        if accumulated >= args.gradient_accumulation or position == len(batches):
            optimizer_step(
                optimizer,
                trainable_parameters,
                accumulated,
                args.grad_clip,
                grad_scaler,
            )
            accumulated = 0
    episode_count = len(episodes)
    return {name: float(total.cpu() / episode_count) for name, total in metric_totals.items()}


def build_inference_config(device, use_amp=False):
    config = InferenceConfig()
    common = {"device": device, "use_amp": use_amp, "use_fa3": False}
    config.update_from_dict(
        {
            "COL_CONFIG": dict(common),
            "ROW_CONFIG": dict(common),
            "ICL_CONFIG": dict(common),
        }
    )
    return config


@torch.no_grad()
def evaluate(model, episodes, device, inference_config, batch_size=1):
    model.eval()
    log_losses = []
    top1_winner = []
    winner_in_top3 = []
    top3_recalls = []
    exact_top3 = []
    hit_counts = []

    for batch_indices in episode_batches(episodes, batch_size):
        batch_episodes = [materialize_episode(episodes[index]) for index in batch_indices]
        X, y_context, y_query = episode_batch_tensors(batch_episodes, device)
        logits = model(
            X,
            y_train=y_context,
            embed_with_test=False,
            inference_config=inference_config,
        )[..., :2].float()
        per_row_losses = F.cross_entropy(
            logits.reshape(-1, 2), y_query.reshape(-1), reduction="none"
        ).view(len(batch_episodes), -1)
        log_losses.extend(per_row_losses.mean(dim=1).cpu().tolist())
        batch_scores = (logits[..., 1] - logits[..., 0]).cpu().numpy()
        for episode, scores in zip(batch_episodes, batch_scores):
            ranked = np.argsort(-scores, kind="stable")
            selected = set(ranked[:3].tolist())
            actual_top3 = set(np.flatnonzero(episode.y_query == 1).tolist())
            winner_index = int(np.flatnonzero(episode.winner_query == 1)[0])
            top1_winner.append(float(ranked[0] == winner_index))
            winner_in_top3.append(float(winner_index in selected))
            hits = len(selected.intersection(actual_top3))
            hit_counts.append(hits)
            top3_recalls.append(hits / 3)
            exact_top3.append(float(selected == actual_top3))

    metrics = {
        "race_log_loss": float(np.mean(log_losses)),
        "top1_winner_accuracy": float(np.mean(top1_winner)),
        "winner_in_top3": float(np.mean(winner_in_top3)),
        "top3_recall": float(np.mean(top3_recalls)),
        "exact_top3_rate": float(np.mean(exact_top3)),
        "races": len(episodes),
    }
    for hits in range(4):
        metrics[f"hit_rate_{hits}_of_3"] = float(np.mean(np.asarray(hit_counts) == hits))
    return metrics


def format_metrics(metrics):
    return (
        f"loss={metrics['race_log_loss']:.4f} "
        f"winner@1={metrics['top1_winner_accuracy']:.3f} "
        f"winner@3={metrics['winner_in_top3']:.3f} "
        f"top3_recall={metrics['top3_recall']:.3f} "
        f"exact_top3={metrics['exact_top3_rate']:.3f} "
        f"hits[3/2/1/0]={metrics['hit_rate_3_of_3']:.3f}/"
        f"{metrics['hit_rate_2_of_3']:.3f}/"
        f"{metrics['hit_rate_1_of_3']:.3f}/"
        f"{metrics['hit_rate_0_of_3']:.3f} "
        f"races={metrics['races']}"
    )


def checkpoint_score(metrics):
    """Rank checkpoints by Top-3 selection quality, then probability quality."""
    return (
        metrics["top3_recall"],
        metrics["exact_top3_rate"],
        -metrics["race_log_loss"],
    )


def save_trainable_parameters(model, path):
    state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save(state, path)


def restore_trainable_parameters(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].device))


def save_finetuned_checkpoint(model, source_checkpoint, output_path, metadata):
    checkpoint = {
        "config": source_checkpoint["config"],
        "dual_stream_config": source_checkpoint.get("dual_stream_config", {}),
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "fine_tuning": metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def run(args, episode_cache_root):
    if args.learning_rate is None:
        args.learning_rate = 1e-4 if args.finetune_mode == "decoder" else 1e-5

    train_csv, validation_csv, test_csv, data_source = resolve_race_csvs(args)
    if data_source["type"] == "huggingface_dataset":
        revision = args.dataset_revision or "main"
        print(f"Dataset: https://huggingface.co/datasets/{args.dataset} (revision {revision})")
    else:
        print(f"Dataset: local CSV overrides ({train_csv}, {validation_csv})")

    train_header = pd.read_csv(train_csv, nrows=0).columns
    feature_columns = read_feature_columns(args.features_json, train_header)
    train_races, train_columns = load_races(train_csv, feature_columns)
    validation_races, validation_columns = load_races(validation_csv, feature_columns)
    test_races = None
    if args.evaluate_test:
        test_races, test_columns = load_races(test_csv, feature_columns)
        if list(train_columns) != list(test_columns):
            raise ValueError("Train and test CSV schemas do not match")
    if list(train_columns) != list(validation_columns):
        raise ValueError("Train and validation CSV schemas do not match")
    validate_chronology(train_races, validation_races, test_races)

    train_specs = episode_specs(train_races, [], args.context_races, args.max_train_races)
    validation_specs = episode_specs(
        validation_races, train_races, args.context_races, args.max_validation_races
    )
    test_specs = None
    if test_races is not None:
        test_specs = episode_specs(
            test_races, train_races + validation_races, args.context_races, args.max_test_races
        )
    if not train_specs or not validation_specs or (test_specs is not None and not test_specs):
        raise ValueError("Every requested split must produce at least one race episode")

    split_summary = f"train={len(train_races)}, validation={len(validation_races)}"
    if test_races is not None:
        split_summary += f", test={len(test_races)}"
    print(f"Chronological races: {split_summary}; context={args.context_races} race(s)")
    print(f"Configured features: {len(feature_columns)} from {args.features_json}")
    print(f"Episode preprocessing workers: {args.preprocessing_workers}")

    # A 100-race context produces roughly 1.8 MB of float32 features per
    # episode. Keeping every episode resident can exceed 12 GB for this
    # dataset, so cache features and load only a minibatch at a time.
    print(f"Episode feature cache: {episode_cache_root}")
    train_episodes = prepare_episodes(
        train_specs,
        feature_columns,
        "training",
        args.preprocessing_workers,
        episode_cache_root / "training",
    )
    validation_episodes = prepare_episodes(
        validation_specs,
        feature_columns,
        "validation",
        args.preprocessing_workers,
        episode_cache_root / "validation",
    )
    test_episodes = (
        prepare_episodes(
            test_specs,
            feature_columns,
            "test",
            args.preprocessing_workers,
            episode_cache_root / "test",
        )
        if test_specs is not None
        else None
    )
    del train_specs, validation_specs, test_specs
    del train_races, validation_races, test_races
    gc.collect()

    seed_everything(args.seed)
    args.resolved_device = resolve_device(args.device)
    if args.amp is None:
        args.amp = args.resolved_device.type == "cuda"
    elif args.amp and args.resolved_device.type != "cuda":
        raise ValueError("--amp requires a CUDA device; use --no-amp on CPU")
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    model, source_checkpoint, source_path = load_model(args.checkpoint, args.resolved_device)
    trainable_parameters = configure_finetuning(model, args.finetune_mode)
    inference_config = build_inference_config(args.resolved_device, use_amp=args.amp)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    print(f"Checkpoint: {source_path}")
    print(
        f"Device: {args.resolved_device}; mode={args.finetune_mode}; "
        f"amp={'fp16' if args.amp else 'off'}; "
        f"trainable={trainable_count:,}/{total_parameters:,} parameters ({trainable_count / total_parameters:.2%})"
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    grad_scaler = make_grad_scaler(args.amp)
    history = []
    baseline_validation = evaluate(model, validation_episodes, args.resolved_device, inference_config)
    print(f"Pretrained validation: {format_metrics(baseline_validation)}")
    if args.resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    best_score = checkpoint_score(baseline_validation)
    best_epoch = 0
    stale_epochs = 0

    with tempfile.TemporaryDirectory(prefix="tabldm-finetune-") as temporary_directory:
        best_parameters_path = Path(temporary_directory) / "best-trainable.pt"
        save_trainable_parameters(model, best_parameters_path)
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_epoch(
                model,
                train_episodes,
                optimizer,
                trainable_parameters,
                args,
                epoch,
                grad_scaler,
            )
            validation_metrics = evaluate(
                model, validation_episodes, args.resolved_device, inference_config
            )
            history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
            print(
                f"Epoch {epoch:03d}: train_objective={train_metrics['objective']:.4f} "
                f"ce={train_metrics['cross_entropy']:.4f} "
                f"listwise={train_metrics['listwise']:.4f} aux={train_metrics['moe_aux']:.4f}; "
                f"val {format_metrics(validation_metrics)}"
            )

            score = checkpoint_score(validation_metrics)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                stale_epochs = 0
                save_trainable_parameters(model, best_parameters_path)
            else:
                stale_epochs += 1
                if stale_epochs >= args.patience:
                    print(f"Early stopping after {epoch} epochs")
                    break

        restore_trainable_parameters(model, best_parameters_path)

    final_validation = evaluate(model, validation_episodes, args.resolved_device, inference_config)
    print(f"Selected checkpoint epoch: {best_epoch} (0 means untouched pretrained weights)")
    print(f"Best validation: {format_metrics(final_validation)}")
    final_test = None
    if test_episodes is not None:
        final_test = evaluate(model, test_episodes, args.resolved_device, inference_config)
        print(f"Sealed test:     {format_metrics(final_test)}")
    else:
        print("Sealed test not accessed; pass --evaluate-test only for final evaluation")

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(source_path),
        "data_source": data_source,
        "target": TARGET,
        "feature_config": str(args.features_json),
        "features": feature_columns,
        "preprocessing": "per-episode context-fitted, normalization=none, no feature/class shuffle",
        "finetune_mode": args.finetune_mode,
        "context_races": args.context_races,
        "preprocessing_workers": args.preprocessing_workers,
        "amp": args.amp,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "listwise_weight": args.listwise_weight,
        "moe_aux_weight": args.moe_aux_weight,
        "seed": args.seed,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "checkpoint_selection": ["top3_recall", "exact_top3_rate", "negative_race_log_loss"],
        "best_score": list(best_score),
        "history": history,
        "pretrained_validation": baseline_validation,
        "validation": final_validation,
        "test": final_test,
    }
    if args.no_save:
        print("Checkpoint saving disabled by --no-save")
    else:
        save_finetuned_checkpoint(model, source_checkpoint, args.output, metadata)
        print(f"Saved fine-tuned checkpoint: {args.output}")
        print(f"Saved training metadata: {args.output.with_suffix(args.output.suffix + '.json')}")


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the disk cache alive through preprocessing, evaluation, and every
    # epoch, and remove it reliably after success, failure, or Ctrl-C.
    with tempfile.TemporaryDirectory(
        prefix=".tabldm-episodes-",
        dir=args.output.parent,
    ) as episode_cache_directory:
        run(args, Path(episode_cache_directory))


if __name__ == "__main__":
    main()
