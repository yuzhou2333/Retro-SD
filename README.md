# Retrosysthesis Self-Distillation

## Install
### 1. Downloading this repository.
```git clone``` or downloading the zip directly.
### 2. Installing the dependency.

```pip install -r requirements.txt```

```pip install --editable ./ ```

```python setup.py build_ext --inplace ```

### 3. zip the data file to folder(data-bin)

```unzip uspto50k_aug20_o2m.zip -d data_bin/```

### 4. Training.

replace the "project_path" to the real path.
replace the "python_path" to the real path.
replace the "path_2_data" to the real path.

run the command according to the run.txt.Here is a example:

```bash ./PCL_scripts/uspto50k_aug20_o2m/Retro-SD.sh```

```bash ./PCL_scripts/uspto50k_aug20_o2m/test2.sh```

### 5. Predicting.

move the "combine.py, score.sh, score.py and take.py" in the results folder to the real prediction folder like "./results/Retro-SD/157"

After completing the above steps, execute the following command within this folder:

```python take.py```

```python combine.py```

```bash score.sh```

You can change the targets and predictions in "score.sh" to get the corresponding accuracy.

like:

-targets ./targets.txt \
-predictions ./predictions.txt \

-targets ./targets_high.txt \
-predictions ./predictions_high.txt \

-targets ./targets_low.txt \
-predictions ./predictions_low.txt \
