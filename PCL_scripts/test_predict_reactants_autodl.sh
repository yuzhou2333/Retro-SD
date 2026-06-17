#!/usr/bin/env bash

set -euo pipefail

# AutoDL smoke test for PCL_scripts/predict_reactants.py.
#
# Usage examples:
#   bash PCL_scripts/test_predict_reactants_autodl.sh
#   SMILES='OCCCSc1ccncc1' TARGETS='1-10' BEAM=10 NBEST=10 TOPK=20 bash PCL_scripts/test_predict_reactants_autodl.sh
#   CHECKPOINT=/root/autodl-tmp/Retro-SD/save_models/Retro_SD3/checkpoint240.pt bash PCL_scripts/test_predict_reactants_autodl.sh
#   INPUT_FILE=/root/autodl-tmp/Retro-SD/examples/products.smi bash PCL_scripts/test_predict_reactants_autodl.sh

PYTHON_PATH="${PYTHON_PATH:-/root/miniconda3/bin/python}"
PROJECT_PATH="${PROJECT_PATH:-/root/autodl-tmp/Retro-SD}"
DATA_BIN="${DATA_BIN:-${PROJECT_PATH}/data_bin/uspto50k_aug20_o2m}"
REACTION_DICT="${REACTION_DICT:-${DATA_BIN}/reaction_list.txt}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_PATH}/save_models/Retro_SD/checkpoint_best.pt}"
SMILES="${SMILES:-OCCCSc1ccncc1}"
INPUT_FILE="${INPUT_FILE:-}"
TARGETS="${TARGETS:-1}"
BEAM="${BEAM:-2}"
NBEST="${NBEST:-2}"
TOPK="${TOPK:-5}"
MAX_LEN_B="${MAX_LEN_B:-200}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CPU="${CPU:-0}"
ALLOW_UNK="${ALLOW_UNK:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_PATH}/results/manual_predict_$(date +%Y%m%d_%H%M%S)}"

PREDICT_SCRIPT="${PROJECT_PATH}/PCL_scripts/predict_reactants.py"
JSON_FILE="${OUTPUT_DIR}/predictions.json"
LOG_FILE="${OUTPUT_DIR}/predict.log"
ENV_FILE="${OUTPUT_DIR}/env_check.txt"

mkdir -p "${OUTPUT_DIR}"

echo "[Retro-SD predict smoke test]"
echo "PROJECT_PATH=${PROJECT_PATH}"
echo "PYTHON_PATH=${PYTHON_PATH}"
echo "DATA_BIN=${DATA_BIN}"
echo "REACTION_DICT=${REACTION_DICT}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "TARGETS=${TARGETS}"
echo "BEAM=${BEAM}"
echo "NBEST=${NBEST}"
echo "TOPK=${TOPK}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

if [[ ! -f "${PREDICT_SCRIPT}" ]]; then
  echo "error: predict script not found: ${PREDICT_SCRIPT}" >&2
  exit 1
fi

if [[ ! -d "${DATA_BIN}" ]]; then
  echo "error: data-bin directory not found: ${DATA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${REACTION_DICT}" ]]; then
  echo "error: reaction dictionary not found: ${REACTION_DICT}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "error: checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

echo
echo "[environment check]"
"${PYTHON_PATH}" - <<'PY' | tee "${ENV_FILE}"
import importlib
import sys

print("python:", sys.executable)
print("version:", sys.version.replace("\n", " "))
for name in ["torch", "numpy", "omegaconf", "hydra", "sacrebleu", "fairseq"]:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{name}: ok ({version})")
    except Exception as exc:
        print(f"{name}: missing/error ({type(exc).__name__}: {exc})")
PY

cmd=(
  "${PYTHON_PATH}"
  "${PREDICT_SCRIPT}"
  "--checkpoint" "${CHECKPOINT}"
  "--data-bin" "${DATA_BIN}"
  "--reaction-dict" "${REACTION_DICT}"
  "--targets" "${TARGETS}"
  "--beam" "${BEAM}"
  "--nbest" "${NBEST}"
  "--topk" "${TOPK}"
  "--max-len-b" "${MAX_LEN_B}"
  "--batch-size" "${BATCH_SIZE}"
  "--json"
)

if [[ -n "${INPUT_FILE}" ]]; then
  cmd+=("--input-file" "${INPUT_FILE}")
else
  cmd+=("--smiles" "${SMILES}")
fi

if [[ "${CPU}" == "1" ]]; then
  cmd+=("--cpu")
fi

if [[ "${ALLOW_UNK}" == "1" ]]; then
  cmd+=("--allow-unk")
fi

echo
echo "[command]"
printf '%q ' "${cmd[@]}"
echo

set +e
"${cmd[@]}" >"${JSON_FILE}" 2>"${LOG_FILE}"
status=$?
set -e

echo
echo "[predict.log]"
cat "${LOG_FILE}"

if [[ "${status}" -ne 0 ]]; then
  echo
  echo "error: prediction command failed with exit code ${status}" >&2
  echo "json file: ${JSON_FILE}" >&2
  echo "log file: ${LOG_FILE}" >&2
  exit "${status}"
fi

echo
echo "[predictions.json]"
cat "${JSON_FILE}"
echo
echo "saved json: ${JSON_FILE}"
echo "saved log: ${LOG_FILE}"
