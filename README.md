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
🔗 [Model Repository](# "Coming Soon")

## 📝 Notes

- Ensure all file paths are correctly configured before running any scripts
- The training process may take considerable time depending on your hardware
- Make sure you have sufficient disk space for the data and model checkpoints
- Self-distillation technique helps improve model performance through knowledge transfer

---

*For additional support or questions, please refer to the documentation or open an issue.*
