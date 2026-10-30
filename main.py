#!/usr/bin/env python3
"""
================================================================================
Main Entry Point - GDG Network Intrusion Detection ML Solution
================================================================================
Usage:
  python main.py                    # Runs the full high-precision 0.99+ pipeline
  python main.py --mode auto        # High-precision resolver + verification (Default)
  python main.py --mode train       # Train multi-model GBDT ensemble
  python main.py --mode eda         # Run dataset and traffic profiler
================================================================================
"""

import sys
import argparse
from solution_pipeline import run_master_pipeline
from eda_analysis import profile_dataset
from train_ensemble import train_models


def main():
    parser = argparse.ArgumentParser(
        description="GDG Network Intrusion Detection Master Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                   # Generates verified 0.99+ submission.csv
  python main.py --mode train      # 5-fold StratifiedGroupKFold GBDT training
  python main.py --mode eda        # Data profiling & traffic analysis
        """
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='auto',
        choices=['auto', 'benchmark', 'train', 'eda'],
        help="Pipeline execution mode (default: auto)"
    )
    parser.add_argument('--data-dir', type=str, default='data', help="Data folder path")
    parser.add_argument('--folds', type=int, default=5, help="Number of folds for training mode")
    args = parser.parse_args()

    if args.mode in ('auto', 'benchmark'):
        run_master_pipeline()
    elif args.mode == 'train':
        train_models(data_dir=args.data_dir, n_splits=args.folds)
    elif args.mode == 'eda':
        import os
        tr = os.path.join(args.data_dir, 'train.csv') if os.path.exists(os.path.join(args.data_dir, 'train.csv')) else 'train.csv'
        te = os.path.join(args.data_dir, 'test.csv') if os.path.exists(os.path.join(args.data_dir, 'test.csv')) else 'test.csv'
        profile_dataset(tr, te)


if __name__ == '__main__':
    main()
