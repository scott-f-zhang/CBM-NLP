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
        if dataset == 'cebab':
            variant_plan = [
                ('pure', 'D'),
                ('aug', 'D^'),
            ]
        else:
            # For IMDB, map manual->D and generated->D^
            variant_plan = [('manual', 'D'), ('generated', 'D^')]
        for model_name in MODELS:
            lr = get_learning_rate(model_name)
            print(f"\t\tRunning {model_name}... with learning rate: {lr}")
            for variant, data_type in variant_plan:
                # Skip LSTM for CBE-PLMs-CM (not supported in mix-joint)
                if func_name == 'CBE-PLMs-CM' and model_name == 'lstm':
                    print(f"\t\tSkipping LSTM for {func_name} (not supported)")
                    continue
                # CM skips D on both datasets (cebab pure, imdb manual)
                if func_name == 'CBE-PLMs-CM':
                    if (dataset == 'cebab' and variant == 'pure') or (dataset == 'imdb' and variant == 'manual'):
                        print("\t\tSkipping D for CBE-PLMs-CM per paper")
                        continue
                try:
                    kwargs = dict(
                        model_name=model_name,
                        num_epochs=BASE_RUN.num_epochs,
                        dataset=dataset,
                        max_len=BASE_RUN.max_len,
                        batch_size=BASE_RUN.batch_size,
                        optimizer_lr=lr,
                    )
                    if variant is not None:
                        kwargs['variant'] = variant
                    result = func(**kwargs)
                except Exception as e:
                    print(f"\t\tWarning: {func_name}/{dataset}/{model_name}/variant={variant} failed: {e}")
                    result = []
                if func_name == 'PLMs':
                    task_scores = result if isinstance(result, list) else []
                    concept_scores = []
                elif func_name == 'CBE-PLMs':
                    task_scores = result.get('task', []) if isinstance(result, dict) else []
                    concept_scores = result.get('concept', []) if isinstance(result, dict) else []
                else:
                    if isinstance(result, dict):
                        task_scores = result.get('task', [])
                        concept_scores = result.get('concept', [])
                    else:
                        task_scores = []
                        concept_scores = []
                rows.append({
                    'dataset': dataset,
                    'data_type': data_type,
                    'function': func_name,
                    'model': model_name,
                    'score': task_scores,
                    'concept_score': concept_scores,
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

    func_order = ["PLMs", "CBE-PLMs", "CBE-PLMs-CM"]
    model_order = ["LSTM", "GPT2", "BERT", "RoBERTa"]
    mapping = {
        'lstm': 'LSTM',
        'gpt2': 'GPT2',
        'bert-base-uncased': 'BERT',
        'roberta-base': 'RoBERTa',
    }

    # Build unified multi-index columns: (dataset, D/D^, metric)
    task_df = df.copy()
    task_df['score_avg'] = task_df['score'].apply(get_average_scores)
    task_df['fmt'] = task_df['score_avg'].apply(get_tuple_2f_fmt)
    task_df['metric'] = 'task'

    concept_df = df.copy()
    if 'concept_score' in concept_df.columns:
        concept_df['score_avg'] = concept_df['concept_score'].apply(get_average_scores)
        concept_df['fmt'] = concept_df['score_avg'].apply(get_tuple_2f_fmt)
        concept_df['metric'] = 'concept'
    else:
        concept_df = concept_df.iloc[0:0]

    merged = pd.concat([
        task_df[['function','model','dataset','data_type','metric','fmt']],
        concept_df[['function','model','dataset','data_type','metric','fmt']]
    ], ignore_index=True)

    merged['model'] = merged['model'].map(mapping)

    wide = merged.pivot_table(index=['function','model'],
                              columns=['dataset','data_type','metric'],
                              values='fmt', aggfunc='first')
    wide = wide.reindex(pd.MultiIndex.from_product([func_order, model_order], names=["function","model"]))

    desired_cols = []
    for ds in ['cebab','imdb']:
        for dt in ['D','D^']:
            for m in ['task','concept']:
                desired_cols.append((ds, dt, m))
    for col in desired_cols:
        if col not in wide.columns:
            wide[col] = pd.NA
    wide = wide[desired_cols]

    return df, wide


def main():
    df = run_all_experiments()
    df, dfp = build_pivot_table(df)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nUnified Pivot (dataset, D/D^, task/concept):")
    print(dfp)
    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


