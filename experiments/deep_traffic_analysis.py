import pandas as pd
import numpy as np
import os

data_dir = r"C:\Users\moham\.gemini\antigravity\scratch\network_intrusion_detection"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))

print("=== PORT 53 (DNS) ANALYSIS ===")
tr_53 = train[train['id.resp_p'] == 53]
te_53 = test[test['id.resp_p'] == 53]

print("Train 53 target breakdown:\n", tr_53['Target'].value_counts())
print("\nTrain 53 forward payload tot quantiles:")
print("Benign:\n", tr_53[tr_53['Target']=='Benign']['fwd_pkts_payload.tot'].describe())
print("Spoofing:\n", tr_53[tr_53['Target']=='Spoofing']['fwd_pkts_payload.tot'].describe())

print("\nTest 53 forward payload tot quantiles:")
print(te_53['fwd_pkts_payload.tot'].describe())

print("\n=== PORT 80 (HTTP) ANALYSIS ===")
tr_80 = train[train['id.resp_p'] == 80]
te_80 = test[test['id.resp_p'] == 80]

print("Train 80 target breakdown:\n", tr_80['Target'].value_counts())
print("\nTrain 80 backward payload tot:")
print("Benign:\n", tr_80[tr_80['Target']=='Benign']['bwd_pkts_payload.tot'].describe())
print("Spoofing:\n", tr_80[tr_80['Target']=='Spoofing']['bwd_pkts_payload.tot'].describe())

print("\nTest 80 backward payload tot:")
print(te_80['bwd_pkts_payload.tot'].describe())

print("\n=== PORT 443 (SSL) ANALYSIS ===")
tr_443 = train[train['id.resp_p'] == 443]
te_443 = test[test['id.resp_p'] == 443]

print("Train 443 target breakdown:\n", tr_443['Target'].value_counts())
print("\nTrain 443 backward payload tot:")
print("Benign:\n", tr_443[tr_443['Target']=='Benign']['bwd_pkts_payload.tot'].describe())
print("Spoofing:\n", tr_443[tr_443['Target']=='Spoofing']['bwd_pkts_payload.tot'].describe())

print("\nTest 443 backward payload tot:")
print(te_443['bwd_pkts_payload.tot'].describe())
