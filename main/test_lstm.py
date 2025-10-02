#!/usr/bin/env python3
"""LSTM-only test runner for PLMs and CBE-PLMs with flexible switches.

Features:
- 1 epoch quick runs
- filters: dataset(s) [cebab/imdb/all], variant(s) [D/D^/all], metrics [task/concept/all]
- excludes CBE-PLMs-CM for LSTM per paper
"""
import os
import sys
import pandas as pd
import traceback
import torch
import argparse

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
    num_epochs=20,
    max_len=512,
    batch_size=8,
)

DATASETS = ["cebab", "imdb"]
MODELS = ["lstm"]
OUTPUT_CSV = os.path.join(MAIN_DIR, "result_lstm_test.csv")


def get_learning_rate(model_name: str) -> Optional[float]:
    return {"lstm": 1e-2}.get(model_name)


def _variant_plan_for_dataset(dataset: str, variants_filter: str):
    # Map D/D^ to internal variant strings per dataset
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
    print(f"Running {func_name} (LSTM only)...")
    print(f"\tCWD={os.getcwd()}  MAIN_DIR={MAIN_DIR}  ROOT_DIR={ROOT_DIR}")
    for dataset in datasets_sel:
        print(f"\tRunning dataset: {dataset}...")
        variant_plan = _variant_plan_for_dataset(dataset, variants_filter)

        for model_name in MODELS:
            lr = get_learning_rate(model_name)
            print(f"\t\tRunning {model_name}... lr={lr}")
            for variant, data_type in variant_plan:
                try:
                    print(f"\t\tVariant={variant}  data_type={data_type}  lr={lr}")
                    # Quick filesystem diagnostics for checkpoints
                    std_model = os.path.join(MAIN_DIR, f"{model_name}_model_standard.pth")
                    std_head = os.path.join(MAIN_DIR, f"{model_name}_classifier_standard.pth")
                    jt_model = os.path.join(MAIN_DIR, f"{model_name}_joint.pth")
                    jt_head = os.path.join(MAIN_DIR, f"{model_name}_ModelXtoCtoY_layer_joint.pth")
                    print(f"\t\tExisting checkpoints before run: std_model={os.path.exists(std_model)} std_head={os.path.exists(std_head)} jt_model={os.path.exists(jt_model)} jt_head={os.path.exists(jt_head)}")
                    score = func(
                        model_name=model_name,
                        num_epochs=BASE_RUN.num_epochs,
                        dataset=dataset,
                        max_len=BASE_RUN.max_len,
                        batch_size=BASE_RUN.batch_size,
                        optimizer_lr=lr,
                        variant=variant,
                    )
                    # score is list[(acc,f1)] for PLMs, dict for joint; filter by metrics
                    if isinstance(score, dict):
                        task_scores = score.get('task', []) if metrics_filter in ("task", "all") else []
                        concept_scores = score.get('concept', []) if metrics_filter in ("concept", "all") else []
                        score_out = {k: v for k, v in (("task", task_scores), ("concept", concept_scores)) if v}
                    else:
                        # PLMs returns task-only list
                        score_out = score if metrics_filter in ("task", "all") else []
                    print(f"\t\tReturned score type={'dict' if isinstance(score_out, dict) else 'list'}")
                    print(f"\t\tExisting checkpoints after run: std_model={os.path.exists(std_model)} std_head={os.path.exists(std_head)} jt_model={os.path.exists(jt_model)} jt_head={os.path.exists(jt_head)}")
                except Exception as e:
                    print(f"\t\tWarning: {func_name}/{dataset}/{model_name}/variant={variant} failed: {e}")
                    print(traceback.format_exc())
                    score_out = []
                rows.append({
                    'dataset': dataset,
                    'data_type': data_type,
                    'function': func_name,
                    'model': model_name,
                    'score': score_out if not isinstance(score_out, dict) else score_out.get('task', []),
                    'concept_score': [] if not isinstance(score_out, dict) else score_out.get('concept', []),
                })
    return rows


def run_all_experiments(datasets_sel, variants_filter: str, metrics_filter: str) -> pd.DataFrame:
    funcs = {
        'PLMs': get_cbm_standard,
        'CBE-PLMs': get_cbm_joint,
        # CBE-PLMs-CM intentionally excluded for LSTM
    }
    all_rows = []
    for name, fn in funcs.items():
        all_rows.extend(run_experiments_for_function(name, fn, datasets_sel, variants_filter, metrics_filter))
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
        # Normalize model display names to match reindex order
        dfp_c = dfp_c.reset_index()
        dfp_c['model'] = dfp_c['model'].map({'lstm': 'LSTM'}).fillna(dfp_c['model'])
        dfp_c = dfp_c.set_index(['function', 'model'])
        dfp_c = dfp_c.reindex(pd.MultiIndex.from_product([func_order, model_order], names=["function", "model"]))
    else:
        dfp_c = pd.DataFrame(index=pd.MultiIndex.from_product([func_order, model_order], names=["function", "model"]))

    df_imdb = df[df['dataset'] == 'imdb'].copy()
    if not df_imdb.empty:
        dfp_i = df_imdb.pivot(index=['function', 'model'], columns=['dataset'], values='score_fmted')
        dfp_i = dfp_i.reset_index()
        dfp_i['model'] = dfp_i['model'].map({'lstm': 'LSTM'}).fillna(dfp_i['model'])
        dfp_i = dfp_i.set_index(['function', 'model'])
        dfp_i = dfp_i.reindex(pd.MultiIndex.from_product([func_order, model_order], names=["function", "model"]))
    else:
        dfp_i = pd.DataFrame(index=pd.MultiIndex.from_product([func_order, model_order], names=["function", "model"]))

    return df, dfp_c, dfp_i


def main():
    parser = argparse.ArgumentParser(description="LSTM test with dataset/variant/metric filters")
    parser.add_argument("--datasets", default="all", choices=["cebab", "imdb", "all"], help="Datasets to run")
    parser.add_argument("--variants", default="all", choices=["D", "D^", "all"], help="Data variants to run")
    parser.add_argument("--metrics", default="all", choices=["task", "concept", "all"], help="Which metrics to evaluate/report")
    args = parser.parse_args()

    datasets_sel = DATASETS if args.datasets == "all" else [args.datasets]
    variants_filter = args.variants
    metrics_filter = args.metrics
    # Print quick env diagnostics
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    # Dataset paths diagnostics
    cebab_dir = os.path.join(ROOT_DIR, 'dataset', 'cebab')
    imdb_dir = os.path.join(ROOT_DIR, 'dataset', 'imdb')
    print(f"CEBaB dir={cebab_dir} exists={os.path.isdir(cebab_dir)}")
    print(f"IMDB dir={imdb_dir} exists={os.path.isdir(imdb_dir)}")
    for p in [
        os.path.join(cebab_dir, 'train_cebab_new_concept_single.csv'),
        os.path.join(cebab_dir, 'dev_cebab_new_concept_single.csv'),
        os.path.join(cebab_dir, 'test_cebab_new_concept_single.csv'),
        os.path.join(imdb_dir, 'IMDB-train-manual.csv'),
        os.path.join(imdb_dir, 'IMDB-dev-manual.csv'),
        os.path.join(imdb_dir, 'IMDB-test-manual.csv'),
    ]:
        print(f"path={p} exists={os.path.exists(p)}")

    df = run_all_experiments(datasets_sel, variants_filter, metrics_filter)
    df, dfp_cebab, dfp_imdb = build_pivot_table(df)
    df.to_csv(OUTPUT_CSV, index=False)
    print("\nCEBaB (LSTM) D vs D^:")
    print(dfp_cebab)
    print("\nIMDB (LSTM):")
    print(dfp_imdb)
    print(f"\nSaved results to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


