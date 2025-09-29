#!/usr/bin/env python3
"""Lightweight test runner: executes the same pipelines with num_epochs=1.

This file mirrors the structure of `main.py` but reduces training epochs to 1
for quick sanity checks.
"""
import os
import sys
import pandas as pd

# Ensure project root on sys.path so we can import the main package
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from main import (
    get_cbm_standard,
    get_cbm_joint,
    get_cbm_LLM_mix_joint,
)
from main.config.defaults import RunConfig
from typing import Optional


def get_average_scores(score_list):
    if not score_list:
        return (0.0, 0.0)
    s1 = s2 = 0.0
    n = 0
    for a, b in score_list:
        s1 += a
        s2 += b
        n += 1
    return ((s1 / n * 100), (s2 / n * 100))


def get_tuple_2f_fmt(tp):
    f1, f2 = tp
    return f"{f1:.2f}/{f2:.2f}"


# Base run settings: only 1 epoch for quick tests
BASE_RUN = RunConfig(
    num_epochs=1,
    max_len=512,
    batch_size=8,
)

# Keep the same datasets/models as main.py
DATASETS = ["cebab", "imdb"]
MODELS = ["bert-base-uncased", "roberta-base", "gpt2", "lstm"]

OUTPUT_CSV = os.path.join(MAIN_DIR, "result_test.csv")


def get_learning_rate(model_name: str) -> Optional[float]:
    lr_rate_dt = {
        'lstm': 1e-2,
        'gpt2': 1e-4,
        'roberta-base': 1e-5,
        'bert-base-uncased': 1e-5,
    }
    return lr_rate_dt.get(model_name)


def run_experiments_for_function(func_name: str, func):
    rows = []
    print(f"Running {func_name}...")
    for dataset in DATASETS:
        print(f"\tRunning dataset: {dataset}...")
        for model_name in MODELS:
            lr = get_learning_rate(model_name)
            print(f"\t\tRunning {model_name}... with learning rate: {lr}")
            try:
                score = func(
                    model_name=model_name,
                    num_epochs=BASE_RUN.num_epochs,
                    dataset=dataset,
                    max_len=BASE_RUN.max_len,
                    batch_size=BASE_RUN.batch_size,
                    optimizer_lr=lr,
                )
            except Exception as e:
                print(f"\t\tWarning: {func_name}/{dataset}/{model_name} failed: {e}")
                score = []
            rows.append({
                'dataset': dataset,
                'function': func_name,
                'model': model_name,
                'score': score,
            })
    return rows


def run_all_experiments() -> pd.DataFrame:
    plms_funcs = {
        'PLMs': get_cbm_standard,
        'CBE-PLMs': get_cbm_joint,
        'CBE-PLMs-CM': get_cbm_LLM_mix_joint,
    }
    all_rows = []
    for fname, f in plms_funcs.items():
        all_rows.extend(run_experiments_for_function(fname, f))
    df = pd.DataFrame(all_rows)
    return df


def build_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['score_avg'] = df.score.apply(get_average_scores)
    df['score_fmted'] = df.score_avg.apply(get_tuple_2f_fmt)

    dfp = df.pivot(index=['function', 'model'], columns=['dataset'], values='score_fmted')
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
    df = run_all_experiments()
    df, dfp = build_pivot_table(df)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nPivot summary (score_avg as XX.XX/YY.YY):")
    print(dfp)
    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


