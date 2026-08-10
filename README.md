# Phase 1 — AI Governance Platform: Income Classification Baseline

A reproducible, leakage-free machine-learning baseline on the **UCI Adult / Census Income**
dataset (UCI repository `id=2`). Trains and compares **Logistic Regression**, **Random Forest**
and **XGBoost**, and exports per-model predictions that retain `sex` and `race` so Phase 2 can
run fairness / bias audits without retraining.

Verified end-to-end on **Windows 11 Pro + Python 3.14.6 (64-bit)**, Intel i7, CPU only.

---

## 1. Project structure

```text
project_KMIT/
├── data/
│   └── raw/
│       ├── adult_raw.csv                # untouched snapshot fetched from UCI (48,842 × 15)
│       └── adult_metadata.txt           # UCI metadata + variable dictionary
├── models/
│   ├── logistic_regression_pipeline.joblib   # full fitted Pipeline (preprocessing + model)
│   ├── random_forest_pipeline.joblib
│   └── xgboost_pipeline.joblib
├── predictions/
│   ├── logistic_regression_test_predictions.csv
│   ├── random_forest_test_predictions.csv
│   └── xgboost_test_predictions.csv
├── results/
│   ├── model_metrics.csv                # the comparison table
│   ├── model_comparison.png             # grouped bar chart of all metrics
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrix_xgboost.png
│   └── classification_report_<model>.txt
├── src/
│   ├── data_loader.py                   # fetch + raw snapshot + cleaning
│   ├── preprocessing.py                 # ColumnTransformer + the canonical split
│   ├── train.py                         # entry point: train all 3 models
│   └── evaluate.py                      # metrics, prediction export, plots
├── requirements.txt
└── README.md
```

---

## 2. Setup — Windows (exact commands)

Open **PowerShell** and `cd` into the project folder.

```powershell
cd C:\Users\nreddy\Downloads\project_KMIT

# 1. Create a virtual environment
py -3.14 -m venv .venv

# 2. Activate it
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation with *"running scripts is disabled on this system"*, allow
signed local scripts for your user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**Using `cmd.exe` instead of PowerShell?** Only the activation line changes:

```bat
.venv\Scripts\activate.bat
```

**Prefer not to activate at all?** Call the venv interpreter directly:

```powershell
.\.venv\Scripts\python.exe src\train.py
```

---

## 3. Run

```powershell
# (venv active)
python src\train.py
```

That single command does everything: downloads the dataset on first run, cleans it, splits it,
trains all three models, evaluates them on the held-out test set, and writes every artefact
listed in section 1. Runtime on an i7 CPU: **~30 seconds** after the download.

### Other entry points

```powershell
# Inspect the cleaned data only (shape, class balance, nulls, dtypes)
python src\data_loader.py

# Re-score the saved .joblib models without retraining.
# The split is deterministic, so the metrics reproduce exactly.
python src\evaluate.py

# Force a fresh download, ignoring data\raw\adult_raw.csv
python src\train.py --force-download
```

> Note: `evaluate.py` rewrites `results\model_metrics.csv` without the `fit_seconds`
> column (it never trains, so it has no timings). Run `train.py` if you want that column.

---

## 4. Results (held-out test set, 9,769 rows, positive class = `>50K`)

| model | accuracy | precision | recall | f1 | roc_auc | fit (s) |
|---|---|---|---|---|---|---|
| **xgboost** | **0.8782** | **0.7911** | **0.6672** | **0.7239** | **0.9299** | 15.3 |
| random_forest | 0.8684 | 0.7834 | 0.6219 | 0.6934 | 0.9186 | 4.5 |
| logistic_regression | 0.8507 | 0.7314 | 0.5941 | 0.6557 | 0.9042 | 1.0 |

XGBoost wins on every metric. Ranking and absolute values match the published literature for
Adult, which is a good sanity check that no leakage is inflating the scores.

**Read accuracy with care.** The target is imbalanced — 76.1% of records are `<=50K`, so a model
that predicts "everyone earns `<=50K`" scores **0.761 accuracy** while being useless. F1 and
ROC-AUC are the metrics to judge on. Note that recall on the high-earner class is only 0.59–0.67:
all three models miss roughly a third of actual high earners. That error is not evenly
distributed across demographic groups, which is exactly what Phase 2 must quantify.

---

## 5. Methodology

### Dataset
- Source: `ucimlrepo.fetch_ucirepo(id=2)` — 48,842 rows, 14 features, target `income`.
- `data/raw/adult_raw.csv` is written **verbatim** (no cleaning) as an immutable audit trail.
  Later runs read the snapshot, so results are reproducible and work offline.
- The UCI endpoint intermittently resets connections mid-download, so the fetch retries
  up to 5 times with linear backoff. Transport-level retry only — the data is unchanged.

### Cleaning
- `?`, blanks and similar tokens → real nulls. Found: `workclass` 2,799, `occupation` 2,809,
  `native-country` 857 (6,465 missing cells total).
- Target encoded `>50K` → **1**, `<=50K` → **0**.
- The raw file mixes `<=50K` with `<=50K.` (the trailing period comes from the original
  `adult.test` split). Both spellings are collapsed, otherwise the target would have 4 levels
  instead of 2 — a silent, results-wrecking bug.
- `sex` and `race` are **kept as model features** and echoed into every prediction CSV.

### Split
- 80/20 via `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`
  → 39,073 train / 9,769 test.
- `stratify=y` holds the positive rate at 0.2393 in **both** folds. Adult is imbalanced, so an
  unstratified split can shift the class ratio between folds and make recall and ROC-AUC noisy
  and non-comparable across models.
- `random_state=42` makes the split deterministic, so all three models see identical rows and
  `evaluate.py` can rebuild the exact same test set later.

### Data-leakage prevention
This is the part that matters most for governance, and it is enforced structurally:

1. **Split happens first.** `make_split` is called on cleaned-but-untransformed data. At that
   moment no fitted transformer exists anywhere in the process.
2. **All learned transformations live inside the Pipeline.** Imputation medians, the one-hot
   vocabulary and scaler mean/std are parameters *learned from data*. They sit in a
   `ColumnTransformer` that is the first step of each model's `Pipeline`, so
   `pipeline.fit(X_train, y_train)` — the only `fit` call in the codebase — fits them on the
   **training fold only**. Computing a median or scaler over the full dataset before splitting
   is the classic leak: test-set information reaches training and inflates the score.
3. **The test set is touched only via `predict` / `predict_proba`**, which *apply* the
   already-fitted transformers without refitting.
4. **Each model gets its own preprocessor instance**, so nothing is shared or reused across fits.
5. **The whole Pipeline is serialised**, not just the estimator. Inference cannot accidentally
   apply different preprocessing — no train/serve skew.

### Preprocessing detail
| branch | steps | applies to |
|---|---|---|
| numeric (6 cols) | `SimpleImputer(strategy="median")` → `StandardScaler` | Logistic Regression |
| numeric (6 cols) | `SimpleImputer(strategy="median")` | Random Forest, XGBoost |
| categorical (8 cols) | `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")` | all models |

- **Median, not mean**, for numeric imputation: `capital-gain` and `fnlwgt` are heavily
  right-skewed, so the mean gets dragged by outliers.
- **Scaling only for Logistic Regression.** It is a gradient/penalty-based linear model, so
  unscaled `fnlwgt` (~1e5) would swamp `age` (~1e1) and the L2 penalty would be applied
  unevenly. Trees split on ordered thresholds, so monotonic rescaling is a mathematical no-op
  for Random Forest and XGBoost.
- **`handle_unknown="ignore"`** encodes an unseen category as an all-zero block instead of
  raising — required for robustness against categories that appear only in test or in future
  production data.

### Models
All use `random_state=42`. Fixed, sensible defaults — **no hyperparameter tuning** — so the
baseline is fast and reproducible on a laptop.

- `LogisticRegression(max_iter=2000, solver="lbfgs")` — interpretable governance baseline.
- `RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1)`
- `XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=6, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, tree_method="hist", n_jobs=-1)`

### Prediction output schema
One row per test record, per model:

| column | meaning |
|---|---|
| `actual_income` | ground truth, 0 / 1 |
| `predicted_income` | predicted class at the default 0.5 threshold |
| `predicted_probability` | `P(income > 50K)` from `predict_proba` |
| `sex` | original value, for fairness analysis |
| `race` | original value, for fairness analysis |
| `actual_income_label` | convenience: `<=50K` / `>50K` |
| `predicted_income_label` | convenience: `<=50K` / `>50K` |

`sex` and `race` are copied from the **raw, pre-transform** test features, so they are
human-readable values rather than one-hot columns — group-by ready.

---

## 6. Assumptions and deliberate non-choices

Recorded explicitly, since each is a decision Phase 2 may want to revisit:

- **Rows with a missing target are dropped** (none in practice). You cannot supervise on a null
  label, and imputing a *label* would fabricate ground truth.
- **`fnlwgt` and `education-num` are kept.** `fnlwgt` is a census sampling weight, not a personal
  attribute, and is often dropped; `education-num` is the ordinal twin of `education`, so the two
  are redundant. Both were left in rather than removed silently.
- **No class re-weighting, no resampling, default 0.5 threshold.** All three models are compared
  under identical, untouched class priors so the table is apples-to-apples. Threshold tuning and
  `class_weight="balanced"` are Phase-2 fairness levers.
- **No hyperparameter tuning and no cross-validation.** This is a baseline. Any future tuning
  must run CV *inside the training fold* to stay leakage-free.
- **Duplicate rows are not removed.** Adult contains some exact duplicates; dropping them is a
  modelling decision, so it was left to Phase 2.
- **`sex` and `race` are used as predictors.** Dropping them would not remove bias — it is
  recoverable from correlated proxies like `occupation` and `relationship` — while it *would*
  remove the ability to measure bias. Fairness-aware training is Phase 2.

---

## 7. Environment notes / troubleshooting

- Installed on this machine: `pandas 3.0.5`, `numpy 2.5.2`, `scikit-learn 1.9.0`,
  `xgboost 3.4.0`, `matplotlib 3.11.1`, `joblib 1.5.3`, `ucimlrepo 0.0.7`.
- `requirements.txt` uses **lower bounds only**, so pip picks wheels that exist for your Python
  version. All of the above ship `cp314` wheels — no compiler needed.
- The code is written to work on both pandas 2.x and 3.x. Text-column detection is done by
  *exclusion* of numeric/bool/datetime dtypes rather than `dtype == object`, because pandas 3
  stores strings as a dedicated `str` dtype and an `== object` test would silently skip every
  text column.
- `random_forest_pipeline.joblib` is ~100 MB (300 unpruned trees). Expected, not an error.
  Raise `min_samples_leaf` or lower `n_estimators` to shrink it.
- matplotlib uses the headless `Agg` backend, so plots are written straight to PNG and no window
  ever opens.
- **`ConnectionResetError` / `WinError 10054` on first run**: transient UCI server issue. The
  loader retries automatically; if all 5 attempts fail, check your proxy/VPN and re-run.
- **`ModuleNotFoundError: No module named 'xgboost'`**: the venv is not active. Re-run
  `.\.venv\Scripts\Activate.ps1`, or call `.\.venv\Scripts\python.exe` directly.
