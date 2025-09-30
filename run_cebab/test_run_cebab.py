#!/usr/bin/env python3
"""Lightweight test runner for the CEBaB models inside run_cebab.

This mirrors the structure / reporting format of `main/test_main.py` but
imports the local (original) implementations in this folder instead of the
refactored versions under `main/`.

- Forces num_epochs=1 for a very quick sanity run.
- Produces a CSV `result_test.csv` in this folder with the same columns:
    dataset,function,model,score
  where score is the raw list returned by each get_cbm_* function (list of
  (accuracy, macro_f1) tuples). A pivot-style pretty print is also shown.

NOTE: Running this script will (re)train each model for 1 epoch and may
overwrite any existing `*_model_*.pth` / `*_classifier_*.pth` weights in
this directory (because the original training code saves to fixed names).
If you need to preserve the original long-trained weights, back them up
before running.
"""
import os
import sys
import pandas as pd
from typing import Optional, Callable, List, Tuple, Dict, Any

# To make relative imports inside cbm_* modules (e.g. `from .cbm_template_models import ...`) work
# we must import them as part of the package `run_cebab`. Therefore we add the *parent* directory
# (repo root) to sys.path, NOT this directory itself. Then we import via `run_cebab.xxx`.
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Import local (original) functions through the package namespace
from run_cebab.cbm_standard import get_cbm_standard  # type: ignore
from run_cebab.cbm_joint import get_cbm_joint        # type: ignore
from run_cebab.cbm_LLM_mix_joint import get_cbm_LLM_mix_joint  # type: ignore

# ----------------- Helpers (mirroring main/test_main.py) -----------------

def get_average_scores(score_list: List[Tuple[float, float]]):
    if not score_list:
        return (0.0, 0.0)
    s1 = s2 = 0.0
    for a, b in score_list:
        s1 += a
        s2 += b
    n = len(score_list)
    return (s1 / n * 100, s2 / n * 100)  # convert to percentage like original

def get_tuple_2f_fmt(tp: Tuple[float, float]) -> str:
    f1, f2 = tp
    return f"{f1:.2f}/{f2:.2f}"

# Unified quick-run settings (align with main/test_main.py choices)
NUM_EPOCHS = 1
MAX_LEN = 512
BATCH_SIZE = 8

# Models to evaluate (consistent ordering for comparison)
MODELS = ["bert-base-uncased", "roberta-base", "gpt2", "lstm"]

# We produce a single logical dataset label 'cebab' so it lines up with
# `main/test_main.py` (which also has 'cebab'). The underlying code here
# internally may use different data_type variants (pure / augmented), but
# for high-level comparison we report them all under the same dataset name.
DATASET_LABEL = "cebab"
OUTPUT_CSV = os.path.join(THIS_DIR, "result_test.csv")

# Learning rates copied from main/test_main.py logic
LR_MAPPING = {
    'lstm': 1e-2,
    'gpt2': 1e-4,
    'roberta-base': 1e-5,
    'bert-base-uncased': 1e-5,
}

def get_learning_rate(model_name: str) -> Optional[float]:
    return LR_MAPPING.get(model_name)

# Map the function display names to callables. We'll iterate data_type variants
# to produce D (pure_cebab) and D^ (aug_cebab) for each function where relevant.
PLM_FUNCS: Dict[str, Dict[str, Any]] = {
    'PLMs': {
        'callable': get_cbm_standard,
        # For standard, compare pure_cebab (D) vs aug_cebab (D^)
        'variants': [('pure_cebab', 'D'), ('aug_cebab', 'D^')],
    },
    'CBE-PLMs': {
        'callable': get_cbm_joint,
        # For joint, compare pure_cebab (D) vs aug_cebab (D^)
        'variants': [('pure_cebab', 'D'), ('aug_cebab', 'D^')],
    },
    'CBE-PLMs-CM': {
        'callable': get_cbm_LLM_mix_joint,
        # For mix-joint, its default uses Yelp too; still produce D^ using aug_cebab
        # and keep a fallback to aug_cebab if aug_cebab_yelp isn't available.
        'variants': [('pure_cebab', 'D'), ('aug_cebab', 'D^')],
        'prefer': 'aug_cebab_yelp',
    },
}


def run_experiments_for_function(func_name: str, func_info: Dict[str, Any]):
    rows = []
    func: Callable = func_info['callable']
    variants = func_info.get('variants', [('pure_cebab', 'D')])
    prefer = func_info.get('prefer')

    print(f"Running {func_name}...")
    dataset_label = DATASET_LABEL
    for model_name in MODELS:
        lr = get_learning_rate(model_name)
        print(f"\tModel: {model_name}  lr={lr}")
        for data_type, data_label in [(prefer, 'D^')] if prefer else []:
            # An optional preferred type to try first (e.g., aug_cebab_yelp) for mix-joint
            if data_type is None:
                continue
            try:
                score = func(
                    model_name=model_name,
                    num_epochs=NUM_EPOCHS,
                    max_len=MAX_LEN,
                    batch_size=BATCH_SIZE,
                    optimizer_lr=lr,
                    data_type=data_type,
                )
                rows.append({'dataset': dataset_label, 'data_type': data_label, 'function': func_name, 'model': model_name, 'score': score})
                # If prefer succeeds, still run the explicit variants below for completeness
            except Exception as e:
                print(f"\tWarning: {func_name}/{model_name} preferred data_type={data_type} failed: {e}")

        for data_type, data_label in variants:
            try:
                score = func(
                    model_name=model_name,
                    num_epochs=NUM_EPOCHS,
                    max_len=MAX_LEN,
                    batch_size=BATCH_SIZE,
                    optimizer_lr=lr,
                    data_type=data_type,
                )
            except Exception as e:
                print(f"\tWarning: {func_name}/{model_name} data_type={data_type} failed: {e}")
                # Special-case fallback for mix-joint if Yelp files unavailable
                if func_name == 'CBE-PLMs-CM' and data_type == 'aug_cebab_yelp':
                    try:
                        alt_data_type = 'aug_cebab'
                        print(f"\tRetrying {func_name}/{model_name} with data_type={alt_data_type}...")
                        score = func(
                            model_name=model_name,
                            num_epochs=NUM_EPOCHS,
                            max_len=MAX_LEN,
                            batch_size=BATCH_SIZE,
                            optimizer_lr=lr,
                            data_type=alt_data_type,
                        )
                        data_type = alt_data_type
                    except Exception as e2:
                        print(f"\tRetry failed: {func_name}/{model_name}: {e2}")
                        score = []
                else:
                    score = []
            rows.append({
                'dataset': dataset_label,
                'data_type': data_label,
                'function': func_name,
                'model': model_name,
                'score': score,
            })
    return rows


def run_all_experiments() -> pd.DataFrame:
    all_rows = []
    for fname, f_info in PLM_FUNCS.items():
        all_rows.extend(run_experiments_for_function(fname, f_info))
    return pd.DataFrame(all_rows)


def build_pivot_table(df: pd.DataFrame):
    df = df.copy()
    df['score_avg'] = df.score.apply(get_average_scores)
    df['score_fmted'] = df.score_avg.apply(get_tuple_2f_fmt)

    # Pivot by data_type (D vs D^) like the notebook for CEBaB
    dfp = df.pivot(index=['function', 'model'], columns=['data_type'], values='score_fmted')

    # Consistent ordering for readability
    func_order = ["PLMs", "CBE-PLMs", "CBE-PLMs-CM"]
    model_order = ["LSTM", "GPT2", "BERT", "RoBERTa"]
    mapping = {
        'lstm': 'LSTM',
        'gpt2': 'GPT2',
        'bert-base-uncased': 'BERT',
        'roberta-base': 'RoBERTa',
    }
    dfp = dfp.reset_index()
    dfp['model'] = dfp['model'].map(mapping)
    dfp = dfp.set_index(['function', 'model'])
    dfp = dfp.reindex(pd.MultiIndex.from_product([func_order, model_order], names=["function", "model"]))
    return df, dfp


def main():
    # Ensure we run inside THIS_DIR so that relative paths like "../dataset/cebab/*.csv"
    # used in the original cbm_* implementations resolve correctly.
    orig_cwd = os.getcwd()
    if orig_cwd != THIS_DIR:
        os.chdir(THIS_DIR)
    try:
        df = run_all_experiments()
        df, dfp = build_pivot_table(df)
        df.to_csv(OUTPUT_CSV, index=False)

        print("\nPivot summary (score_avg as Acc/F1 in %):")
        print(dfp)
        print(f"\nSaved results to: {OUTPUT_CSV}")
    finally:
        # Restore original working directory to avoid side-effects for callers.
        if os.getcwd() != orig_cwd:
            os.chdir(orig_cwd)


if __name__ == "__main__":
    main()
