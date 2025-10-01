#!/usr/bin/env python3
"""Quick LSTM-only sanity test for PLMs and CBE-PLMs.

Runs 1 epoch per setting and prints a compact pivot. This mirrors main/test_main.py
but filters to the LSTM backbone and excludes CBE-PLMs-CM (not supported for LSTM).
"""
import os
import sys
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from typing import Optional
from main import (
    get_cbm_standard,
    get_cbm_joint,
)
from main.config.defaults import RunConfig


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


BASE_RUN = RunConfig(
    num_epochs=1,
    max_len=512,
    batch_size=8,
)

DATASETS = ["cebab", "imdb"]
MODELS = ["lstm"]
OUTPUT_CSV = os.path.join(MAIN_DIR, "result_lstm_test.csv")


def get_learning_rate(model_name: str) -> Optional[float]:
    return {"lstm": 1e-2}.get(model_name)


def run_experiments_for_function(func_name: str, func):
    rows = []
    print(f"Running {func_name} (LSTM only)...")
    for dataset in DATASETS:
        print(f"\tRunning dataset: {dataset}...")
        # cebab: D (pure) and D^ (aug); imdb: manual
        if dataset == 'cebab':
            variant_plan = [('pure', 'D'), ('aug', 'D^')]
        else:
            variant_plan = [('manual', dataset)]

        for model_name in MODELS:
            lr = get_learning_rate(model_name)
            print(f"\t\tRunning {model_name}... lr={lr}")
            for variant, data_type in variant_plan:
                try:
                    score = func(
                        model_name=model_name,
                        num_epochs=BASE_RUN.num_epochs,
                        dataset=dataset,
                        max_len=BASE_RUN.max_len,
                        batch_size=BASE_RUN.batch_size,
                        optimizer_lr=lr,
                        variant=variant,
                    )
                except Exception as e:
                    print(f"\t\tWarning: {func_name}/{dataset}/{model_name}/variant={variant} failed: {e}")
                    score = []
                rows.append({
                    'dataset': dataset,
                    'data_type': data_type,
                    'function': func_name,
                    'model': model_name,
                    'score': score,
                })
    return rows


def run_all_experiments() -> pd.DataFrame:
    funcs = {
        'PLMs': get_cbm_standard,
        'CBE-PLMs': get_cbm_joint,
        # CBE-PLMs-CM intentionally excluded for LSTM
    }
    all_rows = []
    for name, fn in funcs.items():
        all_rows.extend(run_experiments_for_function(name, fn))
    return pd.DataFrame(all_rows)


def build_pivot_table(df: pd.DataFrame):
    df = df.copy()
    df['score_avg'] = df.score.apply(get_average_scores)
    df['score_fmted'] = df.score_avg.apply(get_tuple_2f_fmt)

    # CEBaB: D/D^; IMDB: single column imdb
    func_order = ["PLMs", "CBE-PLMs"]
    model_order = ["LSTM"]

    df_cebab = df[df['dataset'] == 'cebab'].copy()
    if not df_cebab.empty:
        dfp_c = df_cebab.pivot(index=['function', 'model'], columns=['data_type'], values='score_fmted')
        dfp_c = dfp_c.reset_index().set_index(['function', 'model'])
        dfp_c = dfp_c.reindex(pd.MultiIndex.from_product([func_order, model_order], names=["function", "model"]))
    else:
        dfp_c = pd.DataFrame(index=pd.MultiIndex.from_product([func_order, model_order], names=["function", "model"]))

    df_imdb = df[df['dataset'] == 'imdb'].copy()
    if not df_imdb.empty:
        dfp_i = df_imdb.pivot(index=['function', 'model'], columns=['dataset'], values='score_fmted')
        dfp_i = dfp_i.reset_index().set_index(['function', 'model'])
        dfp_i = dfp_i.reindex(pd.MultiIndex.from_product([func_order, model_order], names=["function", "model"]))
    else:
        dfp_i = pd.DataFrame(index=pd.MultiIndex.from_product([func_order, model_order], names=["function", "model"]))

    return df, dfp_c, dfp_i


def main():
    df = run_all_experiments()
    df, dfp_cebab, dfp_imdb = build_pivot_table(df)
    df.to_csv(OUTPUT_CSV, index=False)
    print("\nCEBaB (LSTM) D vs D^:")
    print(dfp_cebab)
    print("\nIMDB (LSTM):")
    print(dfp_imdb)
    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


