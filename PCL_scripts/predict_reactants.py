#!/usr/bin/env python3
"""Run Retro-SD retrosynthesis inference for product SMILES.

The trained Retro-SD model is a multilingual fairseq translation model:
product SMILES are the source side (src1), and each reaction class is a
target language (tgt1..tgt10). This script tokenizes raw SMILES into the
space-separated token format used during preprocessing, runs fairseq
interactive decoding for the requested target classes, and merges the
reactant predictions into a ranked list.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"rdkit\.Chem\.MolStandardize.*")
        from rdkit import Chem
        from rdkit.Chem.MolStandardize.rdMolStandardize import Cleanup, Uncharger
except ImportError:  # pragma: no cover - depends on the runtime image
    Chem = None
    Cleanup = None
    Uncharger = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_BIN = PROJECT_ROOT / "data_bin" / "uspto50k_aug20_o2m"
DEFAULT_REACTION_DICT = DEFAULT_DATA_BIN / "reaction_list.txt"
DEFAULT_TARGETS = tuple(range(1, 11))
DEFAULT_SOURCE_TEMPLATE = "src1"
DEFAULT_LANG_ARG_STYLE = "lang"

SMILES_TOKEN_RE = re.compile(
    r"(\[[^\[\]]+\]|Br|Cl|Si|Sn|Mg|Zn|Cu|Se|se|B|C|N|O|P|S|F|I|H|"
    r"b|c|n|o|p|s|\%\d{2}|[0-9]|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?)"
)
D_LINE_RE = re.compile(r"^D-(\d+)\t([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\t(.*)$")


class Prediction(NamedTuple):
    reactants: str
    score: float
    reaction_class: str
    rank: int


class AugmentedInput(NamedTuple):
    product_index: int
    augmentation_index: int
    raw_smiles: str
    standardized_smiles: str
    augmented_smiles: str


def _require_rdkit() -> None:
    if Chem is None:
        raise RuntimeError(
            "RDKit is required for SMILES standardization and augmentation. "
            "Install RDKit or pass --no-rdkit-standardize for raw tokenization."
        )


def standardize_smiles(smiles: str, keep_atom_map: bool = False) -> str:
    """Standardize a SMILES string with RDKit and return canonical isomeric SMILES."""
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")

    if Cleanup is not None and Uncharger is not None:
        mol = Cleanup(mol)
        mol = Uncharger().uncharge(mol)

    if not keep_atom_map:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")

    standardized = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    if not standardized:
        raise ValueError(f"RDKit produced an empty standardized SMILES for: {smiles}")
    return standardized


def augment_smiles(smiles: str, augmentation: int, seed: int) -> List[str]:
    """Create exactly augmentation product-side rooted/randomized SMILES variants."""
    if augmentation < 1:
        raise ValueError("--augmentation must be at least 1")
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse standardized SMILES: {smiles}")

    variants: List[str] = []

    def add_variant(candidate: str) -> None:
        if candidate and candidate not in variants:
            variants.append(candidate)

    add_variant(Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True))

    atom_indices = list(range(mol.GetNumAtoms()))
    rng = random.Random(seed)
    rng.shuffle(atom_indices)
    for atom_index in atom_indices:
        add_variant(
            Chem.MolToSmiles(
                mol,
                isomericSmiles=True,
                canonical=True,
                rootedAtAtom=atom_index,
            )
        )
        if len(variants) >= augmentation:
            break

    attempts = 0
    while len(variants) < augmentation and attempts < max(augmentation * 20, 20):
        add_variant(Chem.MolToSmiles(mol, isomericSmiles=True, canonical=False, doRandom=True))
        attempts += 1

    while len(variants) < augmentation:
        variants.append(variants[len(variants) % len(variants)])

    return variants[:augmentation]


def build_augmented_inputs(
    raw_inputs: Sequence[str],
    augmentation: int,
    seed: int,
    standardize: bool = True,
    keep_atom_map: bool = False,
) -> List[AugmentedInput]:
    augmented: List[AugmentedInput] = []
    for product_index, raw_smiles in enumerate(raw_inputs):
        standardized = (
            standardize_smiles(raw_smiles, keep_atom_map=keep_atom_map)
            if standardize
            else raw_smiles.strip()
        )
        variants = augment_smiles(standardized, augmentation, seed + product_index) if standardize else [standardized]
        if not standardize and augmentation > 1:
            variants.extend([standardized] * (augmentation - 1))
        for augmentation_index, variant in enumerate(variants[:augmentation]):
            augmented.append(
                AugmentedInput(
                    product_index=product_index,
                    augmentation_index=augmentation_index,
                    raw_smiles=raw_smiles,
                    standardized_smiles=standardized,
                    augmented_smiles=variant,
                )
            )
    return augmented


def tokenize_smiles(smiles: str) -> List[str]:
    """Tokenize a SMILES string with the same atom-level convention as the data."""
    tokens: List[str] = []
    pos = 0
    text = smiles.strip()
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        match = SMILES_TOKEN_RE.match(text, pos)
        if not match:
            context = text[max(0, pos - 8) : pos + 8]
            raise ValueError(
                f"Unsupported SMILES token at character {pos}: {text[pos]!r} in ...{context}..."
            )
        tokens.append(match.group(1))
        pos = match.end()
    if not tokens:
        raise ValueError("SMILES input is empty after trimming whitespace")
    return tokens


def detokenize_smiles(tokenized: str) -> str:
    """Convert fairseq's space-separated token output back to compact SMILES."""
    return tokenized.replace(" ", "").strip()


def parse_targets(value: str) -> List[int]:
    """Parse target class selections like 'all', '1-3,5', or '2,4,10'."""
    if value.strip().lower() == "all":
        return list(DEFAULT_TARGETS)

    targets: List[int] = []
    seen = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"Invalid target range {part!r}: start is greater than end")
            values = range(start, end + 1)
        else:
            values = [int(part)]
        for target in values:
            if target not in DEFAULT_TARGETS:
                raise ValueError("Target classes must be in the range 1..10")
            if target not in seen:
                seen.add(target)
                targets.append(target)
    if not targets:
        raise ValueError("No target classes selected")
    return targets


def parse_interactive_predictions(output: str, reaction_class: str) -> Dict[int, List[Prediction]]:
    """Extract detokenized predictions from fairseq interactive output."""
    predictions: Dict[int, List[Prediction]] = defaultdict(list)
    for line in output.splitlines():
        match = D_LINE_RE.match(line.strip())
        if not match:
            continue
        sample_id = int(match.group(1))
        score = float(match.group(2))
        reactants = detokenize_smiles(match.group(3))
        if not reactants:
            continue
        rank = len(predictions[sample_id]) + 1
        predictions[sample_id].append(
            Prediction(
                reactants=reactants,
                score=score,
                reaction_class=reaction_class,
                rank=rank,
            )
        )
    return predictions


def normalize_prediction_smiles(reactants: str, standardize: bool = True) -> str:
    if not standardize:
        return reactants.strip()
    try:
        return standardize_smiles(reactants)
    except Exception:
        return ""


def merge_predictions(
    predictions: Iterable[Prediction],
    limit: Optional[int],
    alpha: float = 1.0,
    standardize: bool = True,
) -> List[Prediction]:
    """Deduplicate reactants and rank candidates by augmentation/class rank votes."""
    best_by_reactants: Dict[str, Prediction] = {}
    votes_by_reactants: Dict[str, float] = defaultdict(float)
    for prediction in predictions:
        reactants = normalize_prediction_smiles(prediction.reactants, standardize=standardize)
        if not reactants:
            continue
        votes_by_reactants[reactants] += 1.0 / (alpha * max(prediction.rank - 1, 0) + 1.0)
        normalized_prediction = Prediction(
            reactants=reactants,
            score=prediction.score,
            reaction_class=prediction.reaction_class,
            rank=prediction.rank,
        )
        current = best_by_reactants.get(reactants)
        if current is None or _prediction_sort_key(normalized_prediction) < _prediction_sort_key(current):
            best_by_reactants[reactants] = normalized_prediction

    merged = sorted(
        best_by_reactants.values(),
        key=lambda item: (
            -votes_by_reactants[item.reactants],
            -item.score,
            item.rank,
            _target_number(item.reaction_class),
            item.reactants,
        ),
    )
    if limit is not None:
        return merged[:limit]
    return merged


def _prediction_sort_key(prediction: Prediction):
    return (-prediction.score, prediction.rank, _target_number(prediction.reaction_class), prediction.reactants)


def _target_number(reaction_class: str) -> int:
    match = re.search(r"(\d+)$", reaction_class)
    return int(match.group(1)) if match else 10**9


def load_vocab_tokens(dict_path: Path) -> set[str]:
    tokens = set()
    with dict_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            tokens.add(line.split()[0])
    return tokens


def validate_known_tokens(
    tokenized_inputs: Sequence[List[str]],
    data_bin: Path,
    allow_unk: bool,
    source_names: Sequence[str],
) -> None:
    if allow_unk:
        return
    vocab = set()
    for source_name in sorted(set(source_names)):
        dict_path = data_bin / f"dict.{source_name}.txt"
        if dict_path.exists():
            vocab.update(load_vocab_tokens(dict_path))
    if not vocab:
        return
    unknown = sorted({token for tokens in tokenized_inputs for token in tokens if token not in vocab})
    if unknown:
        raise ValueError(
            "Input contains token(s) not present in source dictionary files: "
            + ", ".join(unknown)
            + ". Use --allow-unk to let fairseq map them to <unk>."
        )


def resolve_checkpoint(path: Optional[str]) -> Path:
    if path:
        checkpoint = Path(path)
        if checkpoint.is_dir():
            for name in ("checkpoint_best.pt", "checkpoint_last.pt", "checkpoint240.pt"):
                candidate = checkpoint / name
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f"No known checkpoint file found in directory: {checkpoint}")
        return checkpoint

    candidates = [
        PROJECT_ROOT / "save_models" / "Retro_SD" / "checkpoint_best.pt",
        PROJECT_ROOT / "save_models" / "Retro_SD" / "checkpoint_last.pt",
        PROJECT_ROOT / "save_models" / "Retro_SD3" / "checkpoint240.pt",
        PROJECT_ROOT / "save_models" / "Retro_SD_temp10" / "checkpoint240.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No default checkpoint found. Pass --checkpoint explicitly.")


def read_inputs(smiles_args: Sequence[str], input_file: Optional[str]) -> List[str]:
    inputs = [item.strip() for item in smiles_args if item.strip()]
    if input_file:
        with Path(input_file).open("r", encoding="utf-8") as handle:
            inputs.extend(line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#"))
    if not inputs:
        raise ValueError("Provide at least one product SMILES via --smiles or --input-file")
    return inputs


def source_name_for_target(source_template: str, target: int) -> str:
    if "{target}" in source_template:
        return source_template.format(target=target)
    if "{}" in source_template:
        return source_template.format(target)
    return source_template


def build_reaction_pairs(source_template: str) -> str:
    return ",".join(f"{source_name_for_target(source_template, i)}-tgt{i}" for i in DEFAULT_TARGETS)


def run_interactive_for_target(
    tokenized_inputs: Sequence[str],
    target: int,
    checkpoint: Path,
    data_bin: Path,
    reaction_dict: Path,
    python_executable: str,
    beam: int,
    nbest: int,
    max_len_b: int,
    batch_size: int,
    cpu: bool,
    source_template: str,
    lang_arg_style: str,
    reaction_pairs: str,
) -> Dict[int, List[Prediction]]:
    interactive_py = PROJECT_ROOT / "fairseq_cli" / "interactive.py"
    if not interactive_py.exists():
        raise FileNotFoundError(f"Cannot find fairseq interactive entrypoint: {interactive_py}")

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".smi", delete=False) as handle:
        input_path = Path(handle.name)
        for tokenized in tokenized_inputs:
            handle.write(tokenized + "\n")

    command = [
        python_executable,
        str(interactive_py),
        str(data_bin),
        "--path",
        str(checkpoint),
        "--task",
        "translation_multi_simple_epoch",
        "--reaction-dict",
        str(reaction_dict),
        "--reaction-pairs",
        reaction_pairs,
    ]
    if lang_arg_style == "reaction":
        command.extend(["--source-reaction", source_name_for_target(source_template, target)])
        command.extend(["--target-reaction", f"tgt{target}"])
    else:
        command.extend(["--source-lang", source_name_for_target(source_template, target)])
        command.extend(["--target-lang", f"tgt{target}"])
    command.extend(
        [
        "--encoder-langtok",
        "tgt",
        "--decoder-langtok",
        "--remove-bpe",
        "sentencepiece",
        "--beam",
        str(beam),
        "--nbest",
        str(nbest),
        "--max-len-b",
        str(max_len_b),
        "--batch-size",
        str(batch_size),
        "--buffer-size",
        str(max(batch_size, len(tokenized_inputs))),
        "--input",
        str(input_path),
        ]
    )
    if cpu:
        command.append("--cpu")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        result = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    finally:
        input_path.unlink(missing_ok=True)

    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-40:])
        raise RuntimeError(
            f"fairseq interactive failed for tgt{target} with exit code {result.returncode}.\n"
            f"Command: {' '.join(command)}\n"
            f"Output tail:\n{tail}"
        )

    parsed = parse_interactive_predictions(result.stdout, f"tgt{target}")
    if not parsed:
        tail = "\n".join(result.stdout.splitlines()[-40:])
        raise RuntimeError(f"No predictions parsed for tgt{target}. Output tail:\n{tail}")
    return parsed


def predict(args: argparse.Namespace) -> List[dict]:
    data_bin = Path(args.data_bin)
    reaction_dict = Path(args.reaction_dict)
    checkpoint = resolve_checkpoint(args.checkpoint)
    if not data_bin.exists():
        raise FileNotFoundError(f"Data-bin directory does not exist: {data_bin}")
    if not reaction_dict.exists():
        raise FileNotFoundError(f"Reaction dictionary does not exist: {reaction_dict}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    raw_inputs = read_inputs(args.smiles or [], args.input_file)
    augmented_inputs = build_augmented_inputs(
        raw_inputs,
        augmentation=args.augmentation,
        seed=args.seed,
        standardize=not args.no_rdkit_standardize,
        keep_atom_map=args.keep_atom_map,
    )
    tokenized_lists = [tokenize_smiles(item.augmented_smiles) for item in augmented_inputs]
    tokenized_inputs = [" ".join(tokens) for tokens in tokenized_lists]

    targets = parse_targets(args.targets)
    source_names = [source_name_for_target(args.source_template, target) for target in targets]
    validate_known_tokens(tokenized_lists, data_bin, args.allow_unk, source_names)
    reaction_pairs = args.reaction_pairs or build_reaction_pairs(args.source_template)
    per_input_predictions: List[List[Prediction]] = [[] for _ in raw_inputs]
    batch_size = max(1, min(args.batch_size, len(augmented_inputs)))

    for target in targets:
        parsed = run_interactive_for_target(
            tokenized_inputs=tokenized_inputs,
            target=target,
            checkpoint=checkpoint,
            data_bin=data_bin,
            reaction_dict=reaction_dict,
            python_executable=args.python,
            beam=args.beam,
            nbest=args.nbest,
            max_len_b=args.max_len_b,
            batch_size=batch_size,
            cpu=args.cpu,
            source_template=args.source_template,
            lang_arg_style=args.lang_arg_style,
            reaction_pairs=reaction_pairs,
        )
        for sample_id, predictions in parsed.items():
            if 0 <= sample_id < len(augmented_inputs):
                product_index = augmented_inputs[sample_id].product_index
                per_input_predictions[product_index].extend(predictions)

    output = []
    standardized_by_product = {
        item.product_index: item.standardized_smiles
        for item in augmented_inputs
        if item.augmentation_index == 0
    }
    for product_index, (product, predictions) in enumerate(zip(raw_inputs, per_input_predictions)):
        merged = merge_predictions(
            predictions,
            args.topk,
            alpha=args.score_alpha,
            standardize=not args.no_rdkit_standardize,
        )
        output.append(
            {
                "product": product,
                "standardized_product": standardized_by_product.get(product_index, product),
                "augmentation": args.augmentation,
                "predictions": [
                    {
                        "reactants": prediction.reactants,
                        "score": prediction.score,
                        "reaction_class": prediction.reaction_class,
                        "rank_in_class": prediction.rank,
                    }
                    for prediction in merged
                ],
            }
        )
    return output


def write_prediction_blocks(results: Sequence[dict], output_file: Path, topk: int) -> None:
    if topk < 1:
        raise ValueError("--topk must be at least 1 when writing prediction blocks")
    with output_file.open("w", encoding="utf-8", newline="\n") as handle:
        for item in results:
            predictions = item.get("predictions", [])[:topk]
            for prediction in predictions:
                handle.write(prediction["reactants"] + "\n")
            for _ in range(topk - len(predictions)):
                handle.write("\n")


def print_text(results: Sequence[dict]) -> None:
    for item in results:
        print(f"Product: {item['product']}")
        if not item["predictions"]:
            print("  No predictions.")
            continue
        for index, prediction in enumerate(item["predictions"], start=1):
            print(
                f"  {index:>2}. {prediction['reactants']}"
                f"  score={prediction['score']:.4f}"
                f"  class={prediction['reaction_class']}"
                f"  class_rank={prediction['rank_in_class']}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict retrosynthesis reactants for product SMILES with a trained Retro-SD model."
    )
    parser.add_argument("--smiles", action="append", help="Product SMILES. Can be supplied multiple times.")
    parser.add_argument("--input-file", help="Text file with one product SMILES per line.")
    parser.add_argument("--output-file", help="Write one Top-k prediction block per input product.")
    parser.add_argument("--checkpoint", help="Checkpoint .pt path or checkpoint directory.")
    parser.add_argument("--data-bin", default=str(DEFAULT_DATA_BIN), help="Fairseq binarized data directory.")
    parser.add_argument("--reaction-dict", default=str(DEFAULT_REACTION_DICT), help="reaction_list.txt path.")
    parser.add_argument("--reaction-pairs", help="Override fairseq reaction-pairs string.")
    parser.add_argument("--source-template", default=DEFAULT_SOURCE_TEMPLATE, help="Source name template, e.g. src1 or src{target}.")
    parser.add_argument(
        "--lang-arg-style",
        choices=("lang", "reaction"),
        default=DEFAULT_LANG_ARG_STYLE,
        help="Use fairseq --source-lang/--target-lang or --source-reaction/--target-reaction flags.",
    )
    parser.add_argument("--targets", default="all", help="Reaction classes to try, e.g. all, 1-10, or 1,2,6.")
    parser.add_argument("--beam", type=int, default=10, help="Beam size for fairseq decoding.")
    parser.add_argument("--nbest", type=int, default=10, help="Hypotheses retained per reaction class.")
    parser.add_argument("--topk", type=int, default=10, help="Final deduplicated predictions per product.")
    parser.add_argument("--augmentation", type=int, default=20, help="Rooted/randomized product SMILES per input.")
    parser.add_argument("--score-alpha", type=float, default=1.0, help="Rank aggregation alpha across augmentations.")
    parser.add_argument("--seed", type=int, default=1, help="Seed for deterministic augmentation order.")
    parser.add_argument("--max-len-b", type=int, default=200, help="Fairseq generation max_len_b.")
    parser.add_argument("--batch-size", type=int, default=32, help="Interactive decoding batch size.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("--allow-unk", action="store_true", help="Allow tokens not present in source dictionaries.")
    parser.add_argument("--no-rdkit-standardize", action="store_true", help="Skip RDKit standardization and augmentation.")
    parser.add_argument("--keep-atom-map", action="store_true", help="Keep atom-map numbers during RDKit standardization.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch fairseq_cli/interactive.py.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        results = predict(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print_text(results)
    if args.output_file:
        write_prediction_blocks(results, Path(args.output_file), args.topk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
