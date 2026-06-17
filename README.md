# Retrosynthesis Self-Distillation

A deep learning framework for retrosynthesis prediction using self-distillation techniques.

## 📋 Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Testing](#testing)
- [Prediction](#prediction)
- [Model Access](#model-access)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Retro-SD
```

Alternatively, you can download the zip file directly.

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Install the package in editable mode:

```bash
pip install --editable ./
```

Build the extension modules:

```bash
python setup.py build_ext --inplace
```

## 📁 Data Preparation

### 3. Extract Data Files

Extract the data archive to the `data_bin` folder:

```bash
unzip uspto50k_aug20_o2m.zip -d data_bin/
```

PS：The original dataset used in this project comes from the [GLN repository](https://github.com/Hanjun-Dai/GLN),  
which provides the **USPTO-50k** dataset and its processed versions.

## 🎯 Training

### 4. Configure Training Scripts

Before running the training scripts, you need to update the following paths in the shell scripts:

- `project_path`: Set to your actual project directory path
- `python_path`: Set to your Python executable path  
- `path_2_data`: Set to your data directory path

### 5. Run Training

Execute the training script:

```bash
./PCL_scripts/uspto50k_aug20_o2m/Retro-SD.sh
```

For additional training configurations, refer to the `run.txt` file for more examples.

## 🧪 Testing

### 6. Run Test Scripts

Execute the test script:

```bash
./PCL_scripts/uspto50k_aug20_o2m/test2.sh
```

## 🔮 Prediction

### 7. Prepare Prediction Environment

Move the following files from the main directory to your prediction results folder (e.g., `./results/Retro-SD/157`):

- `combine.py`
- `score.sh`
- `score.py`
- `take.py`

### 8. Generate Predictions

Navigate to your prediction folder and execute the following commands in sequence:

```bash
python take.py
```

```bash
python combine.py
```

```bash
bash score.sh
```

### 9. Evaluate Different Reaction Types

You can modify the `score.sh` script to evaluate different target and prediction files:

**For all reaction types evaluation:**
```bash
-targets ./targets.txt \
-predictions ./predictions.txt \
```

**For high-resource reaction classes evaluation:**
```bash
-targets ./targets_high.txt \
-predictions ./predictions_high.txt \
```

**For low-resource reaction classes evaluation:**
```bash
-targets ./targets_low.txt \
-predictions ./predictions_low.txt \
```

## 📚 Model Access
Pre-trained models will be available at:
🔗 [Model Repository](https://drive.google.com/drive/folders/1A8pVJkRvkpLyKRnOo0MMcE_ZqMmQRfwz)

## 📝 Notes

- Ensure all file paths are correctly configured before running any scripts
- The training process may take considerable time depending on your hardware
- Make sure you have sufficient disk space for the data and model checkpoints
- Self-distillation technique helps improve model performance through knowledge transfer

---

*For additional support or questions, please refer to the documentation or open an issue.*

## Inference with Custom Product Files

This repository includes `inference.sh` for running retrosynthesis prediction on a custom text file. The input file should contain one product SMILES per line. The script standardizes each product with RDKit, generates rooted/randomized SMILES augmentations, runs model decoding, aggregates predictions across augmentations, and writes ranked Top-k reactant predictions to `predictions.txt`.

Basic usage:

```bash
bash inference.sh products.txt predictions.txt 10
```

Equivalent configurable usage:

```bash
INPUT_FILE=products.txt \
OUTPUT_FILE=predictions.txt \
TOP_K=10 \
CHECKPOINT=/path/to/checkpoint_best.pt \
DATA_BIN=/path/to/data_bin/uspto50k_aug20_o2m \
bash inference.sh
```

Output format is fixed-size blocks. If `TOP_K=10`, lines 1-10 are the ranked predictions for the first product, lines 11-20 are for the second product, and so on. Missing predictions are padded with blank lines so product boundaries remain stable.

Important runtime parameters:

- `CHECKPOINT`: model checkpoint path. This release does not include checkpoints.
- `DATA_BIN`: fairseq binarized data directory containing dictionaries and `reaction_list.txt`. This release does not include datasets.
- `TOP_K`: number of final predictions per input product.
- `AUGMENTATION`: number of rooted/randomized product SMILES variants per input, default `20`.
- `TARGETS`: reaction classes to decode, default `all`.
- `CPU=1`: force CPU inference.

Retro-SD uses one source language and multiple target classes by default: `src1` to `tgt1..tgt10`. Override `SOURCE_TEMPLATE` or `LANG_ARG_STYLE` only if your checkpoint was trained with a different language-pair convention.
