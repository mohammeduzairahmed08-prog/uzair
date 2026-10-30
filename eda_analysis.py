"""
================================================================================
Exploratory Data Analysis (EDA) and Traffic Profiler
================================================================================
Inspects class distributions, port profiles, attack behaviors, and data alignment
across train and test partitions for the GDG Network Intrusion Detection Challenge.
================================================================================
"""

import os
import argparse
import pandas as pd
import numpy as np


def profile_dataset(train_path: str, test_path: str):
    print("=" * 70)
    print("GDG NETWORK INTRUSION DETECTION - TRAFFIC FLOW PROFILER")
    print("=" * 70)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print(f"\n1. DATASET VOLUMETRICS:")
    print(f"   Train samples : {len(train):,} rows, {train.shape[1]} columns")
    print(f"   Test samples  : {len(test):,} rows, {test.shape[1]} columns")

    print(f"\n2. TRAIN TARGET CLASS DISTRIBUTION:")
    for target, count in train['Target'].value_counts().items():
        print(f"   - {target:<16}: {count:>6,} ({count / len(train) * 100:.2f}%)")

    print(f"\n3. CORE RESPONDER PORTS (id.resp_p):")
    top_train_ports = train['id.resp_p'].value_counts().head(8)
    top_test_ports = test['id.resp_p'].value_counts().head(8)

    print("   Top Train Ports:")
    for p, c in top_train_ports.items():
        print(f"     Port {p:<5}: {c:>6,} flows")

    print("   Top Test Ports:")
    for p, c in top_test_ports.items():
        print(f"     Port {p:<5}: {c:>6,} flows")

    print(f"\n4. PROTOCOL BREAKDOWN:")
    print("   Train Protocols:")
    for proto, count in train['proto'].value_counts().items():
        print(f"     {proto:<6}: {count:>6,}")
    print("   Test Protocols:")
    for proto, count in test['proto'].value_counts().items():
        print(f"     {proto:<6}: {count:>6,}")

    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EDA Traffic Profiler")
    parser.add_argument('--train', type=str, default='data/train.csv', help="Path to train.csv")
    parser.add_argument('--test', type=str, default='data/test.csv', help="Path to test.csv")
    args = parser.parse_args()

    if not os.path.exists(args.train):
        args.train = 'train.csv'
    if not os.path.exists(args.test):
        args.test = 'test.csv'

    if os.path.exists(args.train) and os.path.exists(args.test):
        profile_dataset(args.train, args.test)
    else:
        print("Please ensure train.csv and test.csv are in the current or data/ folder.")
