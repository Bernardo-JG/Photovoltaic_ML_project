# Photovoltaic module efficiency — machine learning project

A university team project for the **DACO curricular unit**, comparing machine-learning and neural-network models on two related tasks:

1. **Regression:** predict a module's `expected_efficiency`.
2. **Classification:** assign one of four efficiency levels: `extremely_bad`, `bad`, `moderate` or `good`.

The inputs are tabular module characteristics and operating/condition variables, not panel photographs. The work covers exploratory analysis, preprocessing, model comparison, feature selection and evaluation.

## Team

- Ana Carolina da Costa Alves (up202106111)
- Bernardo Justiça Godinho (up202107351)
- Ema Fernandes (up201909527)

Bernardo primarily handled model training, evaluation and the deep-learning comparison. The project was developed collaboratively with Ana Carolina Alves and Ema Fernandes.

## Data and preparation

The repository includes `data/raw/pv_module_efficiency_dataset.csv` and an Excel copy. Inputs include affected area, temperature, irradiance, open-circuit voltage (`Voc`), short-circuit current (`Isc`), module type, hotspot, bird-dropping, soiling and junction-box categories.

`notebooks/data_preprocessing.ipynb` separates both targets from the inputs, one-hot encodes categorical variables, uses an **80/20 random train/test split** (`random_state=42`) and fits a `StandardScaler` to the training set's numerical features. Binary features remain unscaled. Saved experiments show 80,000 training rows, 20,000 test rows and 17 encoded inputs.

The notebook also attempts to remove records with an affected area but no recorded condition. Its comparisons use the literal string `None`; pandas' missing-value parsing should be checked before treating that filter as effective. The repository does not establish the dataset's original collection/generation method or an external validation cohort.

## Models and experiment workflow

| Stage | Implementation |
| --- | --- |
| Exploration | `notebooks/eda.ipynb` |
| Preprocessing | `notebooks/data_preprocessing.ipynb` |
| Five-fold model comparison | `notebooks/cross_validation.ipynb`: kNN regression/classification, linear regression, a custom Gaussian/Bernoulli Naive Bayes combination, random forest, logistic regression and XGBoost |
| Selected-model training | `notebooks/model_training.ipynb`: final traditional-model comparison, training time and feature-importance experiments |
| Classification evaluation | `notebooks/model_evaluation.ipynb`: accuracy, weighted F1, multiclass ROC AUC and confusion matrices |
| Neural-network comparison | `notebooks/deeplearning.ipynb`: separate dense MLP regressors and classifiers |

The MLPs use two 64-unit ReLU layers with 30% dropout, Adam and early stopping. Regression has one linear output; classification has four softmax outputs. Twenty percent of the training partition is reserved for neural-network validation. Despite its filename, `results/cnn_predictions.csv` contains predictions from the **dense MLP regressor**, not a convolutional network.

## Recorded results

These values are **saved experiment results**, not a fresh rerun or an estimate of field performance. The baseline table comes from [`results/combined_results.csv`](results/combined_results.csv).

| Model | Task | Saved test result |
| --- | --- | --- |
| Linear regression | Regression | MSE 1.416 × 10⁻⁶; R² 0.998680 |
| XGBoost regressor | Regression | MSE 4.684 × 10⁻⁷; R² 0.999563 |
| Random forest | Classification | Accuracy 98.125% |
| Logistic regression | Classification | Accuracy 98.945% |
| XGBoost classifier, all features | Classification | Accuracy 99.110% |
| Dense MLP | Regression | MSE 3.374 × 10⁻⁵; R² 0.968533 |
| Dense MLP | Classification | Accuracy 99.090% |

The MLP values come from saved outputs in `deeplearning.ipynb`. A later feature-importance experiment records **99.135%** accuracy for a reduced-feature XGBoost classifier. This differs from the baseline CSV because the later cells overwrite classifier prediction files while retaining the original summary CSV. Reduced-feature regression traded lower training time for lower R² (0.998802).

The experiments favour simpler traditional models for this dataset: the dense networks did not improve on the strongest traditional baselines, and linear regression offered a useful simplicity/training-time trade-off.

## Explore or rerun

There is no checked-in dependency lockfile or `requirements.txt`, so the historical environment is not fully reproducible. A starting environment for the notebook imports is:

```bash
git clone https://github.com/Bernardo-JG/Photovoltaic_ML_project.git
cd Photovoltaic_ML_project
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` on macOS/Linux), then install the notebook dependencies:

```bash
python -m pip install jupyterlab pandas numpy scipy scikit-learn xgboost matplotlib seaborn joblib
# Optional: required for notebooks/deeplearning.ipynb
python -m pip install tensorflow
python -m jupyter lab
```

Open notebooks with their working directory set to `notebooks/`, since they use paths such as `../data/processed`. Start with EDA/preprocessing, then cross-validation, model training and model evaluation; run the deep-learning comparison separately. Inspect saved outputs first if you only want to review the work.

Rerunning training overwrites files under `results/`; preserve a copy if you need the original outputs. Dependency versions can change behaviour and results. TensorFlow/Keras and scikit-learn/XGBoost compatibility need to be checked in the chosen environment.

## Repository notes and limitations

- `models.ipynb` is an early exploratory notebook with different path assumptions, not the main reproducible entry point.
- `scripts/` contains auxiliary/example helpers; some assumptions differ from the notebook workflow (for example, a `Type A` module category). Treat the notebooks as the experiment record.
- `docs/methodology.md`, `docs/results.md`, `data/processed/processed_data.csv`, `results/figures` and `results/metrics` are empty placeholders in this snapshot; the last two are files, not populated directories.
- Five-fold cross-validation uses the already-scaled training matrix. For strict fold-independent validation, fit preprocessing inside each fold using a pipeline.
- Feature-selection variants are compared on the same test partition. A fresh held-out evaluation would be needed after selecting a final configuration.
- Class frequencies are uneven; aggregate accuracy alone does not describe every efficiency class.
- The traditional-model and MLP notebooks use different integer label mappings. Keep each mapping with its predictions when comparing outputs.
- A high score on a random split of this dataset does not establish performance on other module populations, operating environments or independently measured installations.

The project is an academic model-comparison study, not a deployed PV monitoring or fault-diagnosis system.
