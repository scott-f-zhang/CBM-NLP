#!/usr/bin/env python3
"""RoBERTa-only test runner for PLMs, CBE-PLMs, and CBE-PLMs-CM with switches."""
import os
import sys
import argparse
import pandas as pd
import torch
import traceback

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from typing import Optional
from main import (
    get_cbm_standard,
    get_cbm_joint,
    get_cbm_LLM_mix_joint,
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


BASE_RUN = RunConfig(num_epochs=1, max_len=512, batch_size=8)
DATASETS = ["cebab", "imdb"]
MODEL_NAME = "roberta-base"
# Output CSV path - save to test_results directory
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(TESTS_DIR, "test_results", "result_roberta_test.csv")


def get_learning_rate(model_name: str) -> Optional[float]:
    return {"roberta-base": 1e-5}.get(model_name)


def _variant_plan_for_dataset(dataset: str, variants_filter: str):
    mapping_cebab = {"D": ("pure", "D"), "D^": ("aug", "D^")}
    mapping_imdb = {"D": ("manual", "D"), "D^": ("gen", "D^")}
    plans = []
    if variants_filter in ("all", "D"):
        plans.append(mapping_cebab["D"] if dataset == "cebab" else mapping_imdb["D"])
    if variants_filter in ("all", "D^"):
        plans.append(mapping_cebab["D^"] if dataset == "cebab" else mapping_imdb["D^"])
    return plans


def run_experiments_for_function(func_name: str, func, datasets_sel, variants_filter: str, metrics_filter: str):
    rows = []
    print(f"Running {func_name} (RoBERTa)...")
    for dataset in datasets_sel:
        print(f"\tRunning dataset: {dataset}...")
        variant_plan = _variant_plan_for_dataset(dataset, variants_filter)
        lr = get_learning_rate(MODEL_NAME)
        for variant, data_type in variant_plan:
            try:
                print(f"\t\tVariant={variant}  data_type={data_type}  lr={lr}")
                score = func(
                    model_name=MODEL_NAME,
                    num_epochs=BASE_RUN.num_epochs,
                    dataset=dataset,
                    max_len=BASE_RUN.max_len,
                    batch_size=BASE_RUN.batch_size,
                    optimizer_lr=lr,
                    variant=variant,
                )
                if isinstance(score, dict):
                    task_scores = score.get('task', []) if metrics_filter in ("task", "all") else []
                    concept_scores = score.get('concept', []) if metrics_filter in ("concept", "all") else []
                    score_out = {k: v for k, v in (("task", task_scores), ("concept", concept_scores)) if v}
                else:
                    score_out = score if metrics_filter in ("task", "all") else []
            except Exception as e:
                print(f"\t\tWarning: {func_name}/{dataset}/{MODEL_NAME}/variant={variant} failed: {e}")
                print(traceback.format_exc())
                score_out = []
            rows.append({
                'dataset': dataset,
                'data_type': data_type,
                'function': func_name,
                'model': MODEL_NAME,
                'score': score_out if not isinstance(score_out, dict) else score_out.get('task', []),
                'concept_score': [] if not isinstance(score_out, dict) else score_out.get('concept', []),
            })
    return rows


def run_all_experiments(datasets_sel, variants_filter: str, metrics_filter: str) -> pd.DataFrame:
    funcs = {
        'PLMs': get_cbm_standard,
        'CBE-PLMs': get_cbm_joint,
        'CBE-PLMs-CM': get_cbm_LLM_mix_joint,
    }
    all_rows = []
    for name, fn in funcs.items():
        all_rows.extend(run_experiments_for_function(name, fn, datasets_sel, variants_filter, metrics_filter))
    return pd.DataFrame(all_rows)


def build_pivot_table(df: pd.DataFrame):
    df = df.copy()
    df['score_avg'] = df.score.apply(get_average_scores)
    df['score_fmted'] = df.score_avg.apply(get_tuple_2f_fmt)

    func_order = ["PLMs", "CBE-PLMs", "CBE-PLMs-CM"]
    model_order = ["RoBERTa"]

    df_cebab = df[df['dataset'] == 'cebab'].copy()
    if not df_cebab.empty:
        dfp_c = df_cebab.pivot(index=['function', 'model'], columns=['data_type'], values='score_fmted')
        dfp_c = dfp_c.reset_index()
        dfp_c['model'] = dfp_c['model'].map({'roberta-base': 'RoBERTa'}).fillna(dfp_c['model'])
        dfp_c = dfp_c.set_index(['function', 'model'])
        dfp_c = dfp_c.reindex(pd.MultiIndex.from_product([func_order, model_order], names=["function", "model"]))
    else:
        dfp_c = pd.DataFrame(index=pd.MultiIndex.from_product([func_order, model_order], names=["function", "model"]))

    df_imdb = df[df['dataset'] == 'imdb'].copy()
    if not df_imdb.empty:
        dfp_i = df_imdb.pivot(index=['function', 'model'], columns=['dataset'], values='score_fmted')
        dfp_i = dfp_i.reset_index()
        dfp_i['model'] = dfp_i['model'].map({'roberta-base': 'RoBERTa'}).fillna(dfp_i['model'])
        dfp_i = dfp_i.set_index(['function', 'model'])
        dfp_i = dfp_i.reindex(pd.MultiIndex.from_product([func_order, model_order], names=["function", "model"]))
    else:
        dfp_i = pd.DataFrame(index=pd.MultiIndex.from_product([func_order, model_order], names=["function", "model"]))

    return df, dfp_c, dfp_i


def main():
    parser = argparse.ArgumentParser(description="RoBERTa test with dataset/variant/metric filters")
    parser.add_argument("--datasets", default="all", choices=["cebab", "imdb", "all"], help="Datasets to run")
    parser.add_argument("--variants", default="all", choices=["D", "D^", "all"], help="Data variants to run")
    parser.add_argument("--metrics", default="all", choices=["task", "concept", "all"], help="Which metrics to evaluate/report")
    args = parser.parse_args()

    datasets_sel = DATASETS if args.datasets == "all" else [args.datasets]
    variants_filter = args.variants
    metrics_filter = args.metrics

    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    df = run_all_experiments(datasets_sel, variants_filter, metrics_filter)
    df, dfp_cebab, dfp_imdb = build_pivot_table(df)
    # Ensure test_results directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print("\nCEBaB (RoBERTa) D vs D^:")
    print(dfp_cebab)
    print("\nIMDB (RoBERTa):")
    print(dfp_imdb)
    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


