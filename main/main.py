#!/usr/bin/env python3
"""Run CBM experiments across datasets and model backbones.

This script centralizes experiment configuration at the top of the file and
exposes a small set of helper functions to:
- execute runs for each pipeline family
- aggregate raw scores into a DataFrame
- build a pivoted, human-readable summary
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


def get_average_scores(score_list):
    """Average a list of (acc, macro_f1) tuples and scale to percentage.

    Returns a pair (acc_pct, macro_f1_pct) in [0, 100].
    """
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
    """Format a (x, y) tuple as a string with two decimals: 'xx.xx/yy.yy'."""
    f1, f2 = tp
    return f"{f1:.2f}/{f2:.2f}"

# Base run settings aligned with main.config.defaults.RunConfig
BASE_RUN = RunConfig(
    num_epochs=20,
    max_len=512,
    batch_size=8,
)

# Which datasets and models to run
DATASETS = ["cebab", "imdb"]  # choose from: "cebab", "imdb"
MODELS = ["bert-base-uncased", "roberta-base", "gpt2", "lstm"]

# Output CSV path
OUTPUT_CSV = os.path.join(MAIN_DIR, "result.csv")


from typing import Optional


def get_learning_rate(model_name: str) -> Optional[float]:
    """Return a reasonable default learning rate for a given backbone name."""
    lr_rate_dt = {
        'lstm': 1e-2,
        'gpt2': 1e-4,
        'roberta-base': 1e-5,
        'bert-base-uncased': 1e-5,
    }
    return lr_rate_dt.get(model_name)


def run_experiments_for_function(func_name: str, func):
    """Run one pipeline family (e.g., PLMs) across datasets and models.

    For CEBaB, run both variants corresponding to D (pure) and D^ (aug_cebab).
    Returns a list of dict rows compatible with pandas.DataFrame.
    """
    rows = []
    print(f"Running {func_name}...")

    for dataset in DATASETS:
        print(f"\tRunning dataset: {dataset}...")

        # Determine which variants to run and the data_type label
        if dataset == 'cebab':
            variant_plan = [
                ('pure', 'D'),      # original CEBaB
                ('aug', 'D^'),      # aug_cebab
            ]
        else:
            # IMDB unchanged; single variant tied to its defaults per pipeline
            variant_plan = [(None, dataset)]  # label with dataset name to avoid clash

        for model_name in MODELS:
            lr = get_learning_rate(model_name)
            print(f"\t\tRunning {model_name}... with learning rate: {lr}")

            for variant, data_type in variant_plan:
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
                    score = func(**kwargs)
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
    """Run all pipeline families and return a long-form results DataFrame."""
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
    """Create pivot tables and formatted output.

    - For CEBaB, pivot by data_type -> columns D and D^.
    - For IMDB, keep a simple dataset column.
    Returns the pair (df_with_scores, pivot_df_cebab, pivot_df_imdb).
    """
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

    # CEBaB: expects data_type in {D, D^}
    df_cebab = df[df['dataset'] == 'cebab'].copy()
    if 'data_type' in df_cebab.columns:
        df_cebab_p = df_cebab.pivot(index=['function', 'model'], columns=['data_type'], values='score_fmted')
    else:
        # Backward-compatible: if not present, treat everything as D
        df_cebab['data_type'] = 'D'
        df_cebab_p = df_cebab.pivot(index=['function', 'model'], columns=['data_type'], values='score_fmted')

    df_cebab_p = df_cebab_p.reset_index()
    df_cebab_p['model'] = df_cebab_p['model'].map(mapping)
    df_cebab_p = df_cebab_p.set_index(['function', 'model'])
    df_cebab_p = df_cebab_p.reindex(pd.MultiIndex.from_product([func_order, model_order], names=["function", "model"]))

    # IMDB: pivot remains by dataset (single column 'imdb')
    df_imdb = df[df['dataset'] == 'imdb'].copy()
    if not df_imdb.empty:
        df_imdb_p = df_imdb.pivot(index=['function', 'model'], columns=['dataset'], values='score_fmted')
        df_imdb_p = df_imdb_p.reset_index()
        df_imdb_p['model'] = df_imdb_p['model'].map(mapping)
        df_imdb_p = df_imdb_p.set_index(['function', 'model'])
        df_imdb_p = df_imdb_p.reindex(pd.MultiIndex.from_product([func_order, model_order], names=["function", "model"]))
    else:
        df_imdb_p = pd.DataFrame(index=pd.MultiIndex.from_product([func_order, model_order], names=["function", "model"]))

    return df, df_cebab_p, df_imdb_p


def main():
    """Entrypoint: run experiments, save CSV, and print CEBaB D/D^ and IMDB pivots."""
    df = run_all_experiments()
    df, dfp_cebab, dfp_imdb = build_pivot_table(df)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nCEBaB Pivot (D vs D^, score_avg as XX.XX/YY.YY):")
    print(dfp_cebab)
    print("\nIMDB Pivot (score_avg as XX.XX/YY.YY):")
    print(dfp_imdb)
    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
