#!/usr/bin/env python3
"""Export a fairseq checkpoint for inference-only use."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove EMA and training-only state from a fairseq checkpoint."
    )
    parser.add_argument("--input", required=True, help="Source training checkpoint.")
    parser.add_argument("--output", required=True, help="Inference checkpoint path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output checkpoint if it already exists.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    try:
        torch.save(state, str(path), _use_new_zipfile_serialization=False)
    except TypeError:
        torch.save(state, str(path))


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def extract_num_updates(state: Dict[str, Any]) -> Optional[int]:
    optimizer_history = state.get("optimizer_history") or []
    if optimizer_history and isinstance(optimizer_history[-1], dict):
        num_updates = optimizer_history[-1].get("num_updates")
        if num_updates is not None:
            return int(num_updates)

    extra_state = state.get("extra_state") or {}
    if isinstance(extra_state, dict) and extra_state.get("num_updates") is not None:
        return int(extra_state["num_updates"])
    return None


def sanitize_optimizer_history(state: Dict[str, Any]) -> list:
    optimizer_history = state.get("optimizer_history") or []
    if not optimizer_history:
        num_updates = extract_num_updates(state)
        return [{"num_updates": num_updates or 0, "lr_scheduler_state": {}, "optimizer_name": ""}]

    last = dict(optimizer_history[-1])
    last.pop("optimizer", None)
    if "num_updates" not in last:
        last["num_updates"] = extract_num_updates(state) or 0
    if "lr_scheduler_state" not in last:
        last["lr_scheduler_state"] = {}
    if "optimizer_name" not in last:
        last["optimizer_name"] = ""
    return [last]


def build_inference_state(state: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
    if "model" not in state:
        raise KeyError("checkpoint is missing required key: model")
    if state.get("args") is None and state.get("cfg") is None:
        raise KeyError("checkpoint is missing required key: cfg or args")

    extra_state = state.get("extra_state") or {}
    if not isinstance(extra_state, dict):
        extra_state = {}

    inference_extra_state = {
        "train_iterator": extra_state.get("train_iterator"),
        "val_loss": extra_state.get("val_loss"),
        "RS_valid_loss": extra_state.get("RS_valid_loss"),
        "best": extra_state.get("best"),
        "num_updates": extract_num_updates(state),
        "source_checkpoint": str(source_path),
        "inference_export": True,
    }
    inference_extra_state = {
        key: value for key, value in inference_extra_state.items() if value is not None
    }

    inference_state = {
        "args": state.get("args"),
        "cfg": state.get("cfg"),
        "model": state["model"],
        "optimizer_history": sanitize_optimizer_history(state),
        "task_state": state.get("task_state", {}),
        "extra_state": inference_extra_state,
    }
    return inference_state


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists; use --overwrite")

    state = load_checkpoint(input_path)
    inference_state = build_inference_state(state, input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(inference_state, output_path)

    source_extra = state.get("extra_state") or {}
    output_extra = inference_state.get("extra_state") or {}
    removed_top_level = sorted(set(state.keys()) - set(inference_state.keys()))

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"input_size: {format_size(input_path.stat().st_size)}")
    print(f"output_size: {format_size(output_path.stat().st_size)}")
    print("serialization: legacy")
    print(f"top_level_keys: {sorted(inference_state.keys())}")
    print(f"removed_top_level_keys: {removed_top_level}")
    print(f"input_has_ema: {'ema_state_dict' in source_extra}")
    print(f"output_has_ema: {'ema_state_dict' in output_extra}")
    print(f"num_updates: {output_extra.get('num_updates')}")


if __name__ == "__main__":
    main()
