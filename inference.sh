#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Editable defaults. Environment variables or positional args override these.
PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_FILE="${INPUT_FILE:-${1:-products.txt}}"
OUTPUT_FILE="${OUTPUT_FILE:-${2:-predictions.txt}}"
TOP_K="${TOP_K:-${3:-10}}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/save_models/Retro_SD/checkpoint_best.pt}"
DATA_BIN="${DATA_BIN:-${PROJECT_DIR}/data_bin/uspto50k_aug20_o2m}"
REACTION_DICT="${REACTION_DICT:-${DATA_BIN}/reaction_list.txt}"
TARGETS="${TARGETS:-all}"
SOURCE_TEMPLATE="${SOURCE_TEMPLATE:-src1}"
LANG_ARG_STYLE="${LANG_ARG_STYLE:-lang}"
AUGMENTATION="${AUGMENTATION:-20}"
BEAM="${BEAM:-${TOP_K}}"
NBEST="${NBEST:-${TOP_K}}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LEN_B="${MAX_LEN_B:-200}"
SCORE_ALPHA="${SCORE_ALPHA:-1.0}"
SEED="${SEED:-1}"
CPU="${CPU:-0}"
ALLOW_UNK="${ALLOW_UNK:-0}"
KEEP_ATOM_MAP="${KEEP_ATOM_MAP:-0}"
NO_RDKIT_STANDARDIZE="${NO_RDKIT_STANDARDIZE:-0}"

if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "Input file not found: ${INPUT_FILE}" >&2
  echo "Usage: bash inference.sh <input.txt> [predictions.txt] [top_k]" >&2
  exit 2
fi

args=(
  "${PROJECT_DIR}/PCL_scripts/predict_reactants.py"
  --input-file "${INPUT_FILE}"
  --output-file "${OUTPUT_FILE}"
  --checkpoint "${CHECKPOINT}"
  --data-bin "${DATA_BIN}"
  --reaction-dict "${REACTION_DICT}"
  --targets "${TARGETS}"
  --source-template "${SOURCE_TEMPLATE}"
  --lang-arg-style "${LANG_ARG_STYLE}"
  --topk "${TOP_K}"
  --beam "${BEAM}"
  --nbest "${NBEST}"
  --augmentation "${AUGMENTATION}"
  --batch-size "${BATCH_SIZE}"
  --max-len-b "${MAX_LEN_B}"
  --score-alpha "${SCORE_ALPHA}"
  --seed "${SEED}"
)

if [[ "${CPU}" == "1" ]]; then
  args+=(--cpu)
fi
if [[ "${ALLOW_UNK}" == "1" ]]; then
  args+=(--allow-unk)
fi
if [[ "${KEEP_ATOM_MAP}" == "1" ]]; then
  args+=(--keep-atom-map)
fi
if [[ "${NO_RDKIT_STANDARDIZE}" == "1" ]]; then
  args+=(--no-rdkit-standardize)
fi

PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" "${args[@]}"
