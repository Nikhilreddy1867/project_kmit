"""
data_loader.py
==============
Data acquisition + cleaning for the UCI Adult / Census Income dataset (UCI id=2).

Responsibilities
----------------
1. Fetch the dataset once via `ucimlrepo` and persist an *untouched* raw CSV
   snapshot in ``data/raw/``. Every later run reads that snapshot, which makes
   the pipeline reproducible and runnable offline.
2. Clean the data:
     * ``?`` / blank-ish tokens  -> real nulls (NaN)
     * ``income`` is the target, encoded as ``>50K`` = 1, ``<=50K`` = 0
     * ``sex`` and ``race`` are deliberately KEPT as ordinary model features and
       are also carried through to the prediction CSVs for later fairness work.

IMPORTANT: cleaning here is limited to *row-wise, split-independent* operations
(token -> NaN, target encoding). Anything that must *learn* from the data
(imputation values, one-hot categories, scaler mean/std) is intentionally NOT
done here -- it lives inside the scikit-learn Pipeline so it is fitted on the
training fold only. That is the core data-leakage guarantee of this project.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Project paths (all relative to the repo root, so the code is location-safe)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
RESULTS_DIR = PROJECT_ROOT / "results"

RAW_CSV_PATH = DATA_RAW_DIR / "adult_raw.csv"
RAW_META_PATH = DATA_RAW_DIR / "adult_metadata.txt"

TARGET_COL = "income"
# Sensitive attributes retained for Phase-2 fairness / bias auditing.
SENSITIVE_COLS = ["sex", "race"]

# Tokens that the Adult dataset (and CSV round-tripping) uses for "missing".
# Compared case-insensitively after whitespace stripping.
MISSING_TOKENS = {"?", "", "na", "n/a", "nan", "none", "null", "-", "--"}


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for d in (DATA_RAW_DIR, MODELS_DIR, PREDICTIONS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. Raw acquisition
# --------------------------------------------------------------------------- #
def download_raw(save: bool = True, max_attempts: int = 5) -> pd.DataFrame:
    """
    Fetch UCI Adult (id=2) with `ucimlrepo` and return features + target joined.

    The frame is saved verbatim -- no cleaning, no type coercion, no renaming --
    so ``data/raw/adult_raw.csv`` is an immutable audit trail of what we pulled.

    The UCI endpoint intermittently drops the connection mid-download
    (``ConnectionResetError``/``WinError 10054``), so the call is retried with
    linear backoff. This is a transport-level retry only -- the data source and
    contents are unchanged.
    """
    import time

    from ucimlrepo import fetch_ucirepo  # imported lazily: only needed on first run

    print("[data_loader] Fetching UCI Adult (id=2) via ucimlrepo ...")
    adult = None
    for attempt in range(1, max_attempts + 1):
        try:
            adult = fetch_ucirepo(id=2)
            break
        except Exception as exc:
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Could not download UCI dataset 2 after {max_attempts} attempts. "
                    "Check your internet connection / proxy and retry."
                ) from exc
            wait = 3 * attempt
            print(f"[data_loader] Attempt {attempt} failed ({type(exc).__name__}); "
                  f"retrying in {wait}s ...")
            time.sleep(wait)

    X = adult.data.features
    y = adult.data.targets  # single column DataFrame named 'income'

    # Join features and target side by side -> one raw table.
    raw = pd.concat([X, y], axis=1)

    if save:
        ensure_dirs()
        raw.to_csv(RAW_CSV_PATH, index=False)
        print(f"[data_loader] Raw snapshot saved -> {RAW_CSV_PATH}  shape={raw.shape}")

        # Persist metadata + variable dictionary alongside the data (governance).
        try:
            with open(RAW_META_PATH, "w", encoding="utf-8") as fh:
                fh.write("=== UCI repo metadata (id=2) ===\n")
                fh.write(str(adult.metadata))
                fh.write("\n\n=== Variable information ===\n")
                fh.write(adult.variables.to_string())
            print(f"[data_loader] Metadata saved  -> {RAW_META_PATH}")
        except Exception as exc:  # metadata is nice-to-have, never fatal
            print(f"[data_loader] WARNING: could not write metadata ({exc})")

    return raw


def load_raw(force_download: bool = False) -> pd.DataFrame:
    """
    Return the raw frame, preferring the local snapshot.

    `keep_default_na=False` on read means the literal strings in the CSV are
    preserved (e.g. ``?``) and handled by our single explicit missing-value
    rule in :func:`clean_data`, instead of pandas silently applying its own.
    """
    if RAW_CSV_PATH.exists() and not force_download:
        raw = pd.read_csv(RAW_CSV_PATH, keep_default_na=False, dtype=str, na_filter=False)
        print(f"[data_loader] Loaded cached raw CSV <- {RAW_CSV_PATH}  shape={raw.shape}")
        return raw
    return download_raw(save=True)


# --------------------------------------------------------------------------- #
# 2. Cleaning
# --------------------------------------------------------------------------- #
def _is_texty(series: pd.Series) -> bool:
    """
    True if the column holds text rather than numbers/bools/dates.

    Checked by *exclusion* instead of ``dtype == object`` because pandas 3.x
    stores strings as a dedicated ``str`` dtype, not ``object`` -- an
    ``== object`` test silently skips every text column there.
    """
    return not (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
    )


def _normalise_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace on text columns and convert missing-ish tokens to NaN."""
    out = df.copy()
    for col in out.columns:
        if _is_texty(out[col]):
            s = out[col].astype("string").str.strip()
            # Case-insensitive comparison against the missing-token set.
            mask = s.str.lower().isin(MISSING_TOKENS)
            out[col] = s.mask(mask, other=pd.NA)
    return out


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restore true numeric dtypes.

    We read the raw CSV as all-string on purpose (so token handling is explicit),
    so numeric columns must be converted back. A column is treated as numeric
    only if every non-null value parses as a number -- this keeps the
    numeric/categorical split data-driven rather than hard-coded.
    """
    out = df.copy()
    for col in out.columns:
        if col == TARGET_COL or not _is_texty(out[col]):
            continue
        converted = pd.to_numeric(out[col], errors="coerce")
        non_null = out[col].notna()
        # Only accept the conversion if it lost no information.
        if non_null.sum() > 0 and converted[non_null].notna().all():
            out[col] = converted.astype("float64")
    return out


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Clean the raw frame and return ``(X, y)``.

    Steps
    -----
    * missing tokens (``?``, blanks, ...) -> NaN
    * target label normalisation: the UCI file mixes ``<=50K`` with ``<=50K.``
      (the trailing period comes from the original *adult.test* split). Both
      spellings must collapse to one class or the target would have 4 levels.
    * rows with an unusable target are dropped -- you cannot supervise on NaN,
      and imputing a *label* would fabricate ground truth.
    * ``>50K`` -> 1 (positive / minority class), ``<=50K`` -> 0.
    """
    df = _normalise_missing(raw)

    if TARGET_COL not in df.columns:
        raise KeyError(f"Expected target column '{TARGET_COL}'. Got: {list(df.columns)}")

    # --- target normalisation -------------------------------------------------
    y_txt = (
        df[TARGET_COL]
        .astype("string")
        .str.strip()
        .str.rstrip(".")  # '<=50K.' -> '<=50K'
        .str.upper()      # guard against casing drift
    )

    n_before = len(df)
    keep = y_txt.notna()
    df, y_txt = df.loc[keep].copy(), y_txt.loc[keep]
    if n_before != len(df):
        print(f"[data_loader] Dropped {n_before - len(df)} row(s) with missing target.")

    mapping = {"<=50K": 0, ">50K": 1}
    y = y_txt.map(mapping)

    unmapped = y_txt[y.isna()].unique().tolist()
    if unmapped:
        raise ValueError(f"Unrecognised income labels: {unmapped}")
    y = y.astype("int8").rename(TARGET_COL).reset_index(drop=True)

    # --- features ------------------------------------------------------------
    # NOTE: 'sex' and 'race' stay in X. They are legitimate predictors here and
    # dropping them would not remove bias (it is recoverable from proxies) while
    # it would remove our ability to measure it. They are also echoed into the
    # prediction CSVs for group-wise fairness metrics in Phase 2.
    # NOTE: 'fnlwgt' (census sampling weight) and 'education-num' (ordinal twin
    # of 'education') are kept as-is. Removing them is a modelling decision, so
    # it is left explicit rather than done silently.
    X = df.drop(columns=[TARGET_COL]).reset_index(drop=True)
    X = _coerce_numeric(X)

    # Normalise categoricals to plain `object` with `np.nan` holes. sklearn's
    # SimpleImputer / OneHotEncoder are happiest with that combination, and it
    # behaves identically on pandas 2.x and 3.x.
    for col in X.columns:
        if _is_texty(X[col]):
            X[col] = X[col].astype(object).where(X[col].notna(), np.nan)

    missing = [c for c in SENSITIVE_COLS if c not in X.columns]
    if missing:
        raise KeyError(f"Sensitive columns missing from features: {missing}")

    return X, y


def load_dataset(force_download: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Convenience entry point: raw snapshot -> cleaned ``(X, y)``."""
    return clean_data(load_raw(force_download=force_download))


if __name__ == "__main__":
    ensure_dirs()
    X, y = load_dataset()
    print("\n--- Cleaned dataset summary ---")
    print(f"X shape: {X.shape}   y shape: {y.shape}")
    print(f"Target balance (1 = >50K):\n{y.value_counts(normalize=True).rename('share')}")
    print(f"\nNulls per column (non-zero only):\n{X.isna().sum().loc[lambda s: s > 0]}")
    print(f"\nDtypes:\n{X.dtypes}")
