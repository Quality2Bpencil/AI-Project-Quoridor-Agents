"""Run resumable staged AlphaZero-style training from a JSON config."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quoridor.training.alphazero import save_alphazero_checkpoint, train_alphazero_self_play


TRAIN_KEYS = {
    "simulations",
    "max_turns",
    "hidden_size",
    "action_limit",
    "wall_limit",
    "batch_size",
    "epochs_per_game",
    "lr",
    "temperature_turns",
    "replay_capacity",
    "draw_value_mode",
    "draw_value_scale",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default=None, help="Override config device, e.g. cuda or cpu.")
    parser.add_argument("--output", default=None, help="Override config base_output.")
    parser.add_argument("--initial-checkpoint", default=None, help="Override config initial_checkpoint.")
    parser.add_argument("--start-stage", type=int, default=0)
    parser.add_argument("--max-stages", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    stages = list(config.get("stages", []))
    if not stages:
        raise ValueError("config must define at least one stage")
    if args.start_stage < 0 or args.start_stage >= len(stages):
        raise ValueError("--start-stage must point to an existing stage")

    base_output = Path(args.output or config.get("base_output", "experiments/results/alphazero_policy_value.pt"))
    stage_dir = Path(config.get("stage_dir", base_output.parent / "alphazero_stages"))
    initial_checkpoint = _resolve_initial_checkpoint(args.initial_checkpoint, config.get("initial_checkpoint"))
    device = args.device if args.device is not None else config.get("device")
    seed = int(config.get("seed", 0))

    selected_indexes = list(range(args.start_stage, len(stages)))
    if args.max_stages is not None:
        selected_indexes = selected_indexes[: args.max_stages]

    if args.dry_run:
        print(_dry_run_summary(config_path, base_output, stage_dir, initial_checkpoint, device, stages, selected_indexes))
        return

    stage_dir.mkdir(parents=True, exist_ok=True)
    previous_checkpoint = initial_checkpoint

    for index, stage in enumerate(stages):
        stage_output = _stage_output_path(stage_dir, index, stage)
        if index < args.start_stage:
            if stage_output.exists():
                previous_checkpoint = stage_output
            continue
        if index not in selected_indexes:
            break

        previous_checkpoint = _run_stage(
            config_path=config_path,
            stage_dir=stage_dir,
            index=index,
            stage=stage,
            previous_checkpoint=previous_checkpoint,
            device=device,
            base_seed=seed,
            resume=args.resume,
        )

    if previous_checkpoint is None:
        raise RuntimeError("training produced no checkpoint")

    base_output.parent.mkdir(parents=True, exist_ok=True)
    if previous_checkpoint.resolve() != base_output.resolve():
        shutil.copy2(previous_checkpoint, base_output)
    print(
        "alphazero staged training complete",
        {
            "config": str(config_path),
            "output": str(base_output),
            "last_checkpoint": str(previous_checkpoint),
        },
    )


def _resolve_initial_checkpoint(arg_value: str | None, config_value: str | None) -> Path | None:
    value = arg_value if arg_value is not None else config_value
    if not value:
        return None
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"initial checkpoint not found: {path}")
    return path


def _run_stage(
    *,
    config_path: Path,
    stage_dir: Path,
    index: int,
    stage: dict[str, Any],
    previous_checkpoint: Path | None,
    device: str | None,
    base_seed: int,
    resume: bool,
) -> Path:
    total_games = int(stage["games"])
    if total_games < 1:
        raise ValueError("stage games must be at least 1")
    chunk_games = int(stage.get("chunk_games", total_games))
    if chunk_games < 1:
        raise ValueError("chunk_games must be at least 1")

    stage_slug = _stage_slug(index, stage)
    chunks = math.ceil(total_games / chunk_games)
    last_checkpoint = previous_checkpoint
    started = time.perf_counter()
    print(
        "starting alphazero stage",
        {
            "stage": stage_slug,
            "games": total_games,
            "chunk_games": chunk_games,
            "chunks": chunks,
            "previous_checkpoint": str(previous_checkpoint) if previous_checkpoint else None,
        },
    )

    for chunk_index in range(chunks):
        current_games = min(chunk_games, total_games - chunk_index * chunk_games)
        chunk_output = stage_dir / f"{stage_slug}_chunk_{chunk_index:04d}.pt"
        if resume and chunk_output.exists():
            last_checkpoint = chunk_output
            print("skipping existing alphazero chunk", {"stage": stage_slug, "chunk": chunk_index})
            continue

        train_kwargs = {key: stage[key] for key in TRAIN_KEYS if key in stage}
        train_kwargs.update(
            {
                "games": current_games,
                "seed": base_seed + index * 1_000_003 + chunk_index,
                "device": device,
                "initial_checkpoint": last_checkpoint,
            }
        )
        model, stats = train_alphazero_self_play(**train_kwargs)
        metadata = {
            "config": str(config_path),
            "stage_index": index,
            "stage_name": stage.get("name", stage_slug),
            "chunk_index": chunk_index,
            "chunk_games": current_games,
            "stage_games": total_games,
            "source_checkpoint": str(last_checkpoint) if last_checkpoint else None,
            "stage": stage,
            "stats": asdict(stats),
        }
        save_alphazero_checkpoint(model, chunk_output, metadata=metadata)
        last_checkpoint = chunk_output
        print(
            "finished alphazero chunk",
            {
                "stage": stage_slug,
                "chunk": chunk_index + 1,
                "chunks": chunks,
                "output": str(chunk_output),
                "examples": stats.examples,
                "updates": stats.updates,
                "wins": stats.wins,
                "draws": stats.draws,
                "value_mean_abs": round(stats.value_mean_abs, 6),
                "value_nonzero_examples": stats.value_nonzero_examples,
                "elapsed_seconds": round(stats.elapsed_seconds, 3),
            },
        )

    if last_checkpoint is None:
        raise RuntimeError(f"stage {stage_slug} produced no checkpoint")
    stage_output = _stage_output_path(stage_dir, index, stage)
    if last_checkpoint.resolve() != stage_output.resolve():
        shutil.copy2(last_checkpoint, stage_output)
    print(
        "finished alphazero stage",
        {
            "stage": stage_slug,
            "output": str(stage_output),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
    )
    return stage_output


def _stage_output_path(stage_dir: Path, index: int, stage: dict[str, Any]) -> Path:
    explicit = stage.get("output")
    if explicit:
        return Path(explicit)
    return stage_dir / f"{_stage_slug(index, stage)}.pt"


def _stage_slug(index: int, stage: dict[str, Any]) -> str:
    raw = str(stage.get("name", f"stage_{index}"))
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in raw).strip("_")
    return f"{index:02d}_{safe or 'stage'}"


def _dry_run_summary(
    config_path: Path,
    base_output: Path,
    stage_dir: Path,
    initial_checkpoint: Path | None,
    device: str | None,
    stages: list[dict[str, Any]],
    selected_indexes: list[int],
) -> dict[str, Any]:
    return {
        "config": str(config_path),
        "output": str(base_output),
        "stage_dir": str(stage_dir),
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
        "device": device,
        "selected_stages": [
            {
                "index": index,
                "name": stages[index].get("name", f"stage_{index}"),
                "games": stages[index]["games"],
                "chunk_games": stages[index].get("chunk_games", stages[index]["games"]),
                "simulations": stages[index].get("simulations"),
                "action_limit": stages[index].get("action_limit"),
                "wall_limit": stages[index].get("wall_limit"),
                "batch_size": stages[index].get("batch_size"),
                "replay_capacity": stages[index].get("replay_capacity"),
            }
            for index in selected_indexes
        ],
    }


if __name__ == "__main__":
    main()
