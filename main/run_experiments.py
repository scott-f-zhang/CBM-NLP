#!/usr/bin/env python3
import os
import sys
import argparse
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


def build_arg_parser():
    p = argparse.ArgumentParser(description="Run CBM experiments (modular)")
    p.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs per run")
    p.add_argument("--max_len", type=int, default=64, help="Max sequence length")
    p.add_argument("--batch_size", type=int, default=2, help="Batch size")
    # Save CSV inside main folder by default
    p.add_argument("--output", type=str, default=os.path.join(MAIN_DIR, "result.csv"), help="Output CSV path")
    p.add_argument("--data_types", type=str, nargs="*", default=["pure_cebab", "aug_cebab"], help="Data types to run")
    p.add_argument("--models", type=str, nargs="*", default=["bert-base-uncased", "roberta-base", "gpt2", "lstm"], help="Models to run")
    return p


def main():
    args = build_arg_parser().parse_args()

    plms_funcs = {
        'PLMs': get_cbm_standard,
        'CBE-PLMs': get_cbm_joint,
        'CBE-PLMs-CM': get_cbm_LLM_mix_joint,
    }

    lr_rate_dt = {
        'lstm': 1e-2,
        'gpt2': 1e-4,
        'roberta-base': 1e-5,
        'bert-base-uncased': 1e-5,
    }

    results = {
        'data_type': [],
        'function': [],
        'model': [],
        'score': [],
    }

    for f_name, f in plms_funcs.items():
        print(f"Running {f_name}...")
        for data_type in args.data_types:
            print(f"\tRunning {data_type}...")
            for model_name in args.models:
                lr = lr_rate_dt.get(model_name)
                print(f"\t\tRunning {model_name}... with learning rate: {lr}")
                try:
                    score = f(
                        model_name=model_name,
                        num_epochs=args.num_epochs,
                        data_type=data_type,
                        max_len=args.max_len,
                        batch_size=args.batch_size,
                        optimizer_lr=lr,
                    )
                except Exception as e:
                    print(f"\t\tWarning: {f_name}/{data_type}/{model_name} failed: {e}")
                    score = []
                results['data_type'].append(data_type)
                results['function'].append(f_name)
                results['model'].append(model_name)
                results['score'].append(score)

    df = pd.DataFrame.from_dict(results)
    df['score_avg'] = df.score.apply(get_average_scores)
    df['score_fmted'] = df.score_avg.apply(get_tuple_2f_fmt)
    df.to_csv(args.output, index=False)

    # Pretty pivot like the notebook
    dfp = df.pivot(index=['function', 'model'], columns=['data_type'], values='score_fmted')
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

    print("\nPivot summary (score_avg as XX.XX/YY.YY):")
    print(dfp)
    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
