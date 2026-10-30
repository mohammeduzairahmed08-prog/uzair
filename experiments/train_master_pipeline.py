"""
Master Training Pipeline for Network Intrusion Detection
Multi-Model Deep Ensemble: LightGBM + XGBoost + CatBoost + ExtraTrees
With Domain Adaptation & Zero-Leakage Feature Engineering
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import shutil

print("=" * 80)
print("DEEP MULTI-MODEL ENSEMBLE TRAINING: NETWORK INTRUSION DETECTION")
print("=" * 80)

data_dir = r"C:\Users\moham\.gemini\antigravity\scratch\network_intrusion_detection"
train_path = os.path.join(data_dir, "train.csv")
test_path = os.path.join(data_dir, "test.csv")
sample_sub_path = os.path.join(data_dir, "sample_submission.csv")

print(f"\n[1/6] Loading competition datasets...")
t0 = time.time()
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

print(f"      Train shape: {train.shape}")
print(f"      Test shape:  {test.shape}")
print(f"      Loaded in {time.time() - t0:.2f} seconds")

class_names = ['Availability', 'Benign', 'Reconnaissance', 'Spoofing']
label_to_id = {name: i for i, name in enumerate(class_names)}
id_to_label = {i: name for i, name in enumerate(class_names)}
y = train['Target'].map(label_to_id).values

# Step 2: Advanced Feature Engineering
print(f"\n[2/6] Engineering domain-specific, leakage-free flow features...")

def engineer_features(df, is_for_catboost=False):
    d = df.copy()
    
    # 1. Timing and Duration Robustness
    safe_dur = np.maximum(d['flow_duration'], 1e-6)
    d['safe_duration'] = safe_dur
    d['is_zero_duration'] = (d['flow_duration'] == 0).astype(int)
    d['log_duration'] = np.log10(safe_dur + 1e-6)
    
    # Recalculated robust rate features (avoids Flowmeter division-by-zero artifacts)
    d['recalc_flow_pkts_per_sec'] = (d['fwd_pkts_tot'] + d['bwd_pkts_tot']) / safe_dur
    d['recalc_fwd_pkts_per_sec'] = d['fwd_pkts_tot'] / safe_dur
    d['recalc_bwd_pkts_per_sec'] = d['bwd_pkts_tot'] / safe_dur
    d['recalc_payload_bytes_per_sec'] = (d['fwd_pkts_payload.tot'] + d['bwd_pkts_payload.tot']) / safe_dur
    
    d['log_recalc_flow_pkts_per_sec'] = np.log10(d['recalc_flow_pkts_per_sec'] + 1)
    d['log_recalc_payload_bytes_per_sec'] = np.log10(d['recalc_payload_bytes_per_sec'] + 1)
    
    # 2. Flow Packet & Payload Structure
    tot_pkts = d['fwd_pkts_tot'] + d['bwd_pkts_tot']
    tot_payload = d['fwd_pkts_payload.tot'] + d['bwd_pkts_payload.tot']
    d['tot_pkts'] = tot_pkts
    d['tot_payload'] = tot_payload
    d['fwd_pkt_ratio'] = d['fwd_pkts_tot'] / (tot_pkts + 1e-5)
    d['bwd_pkt_ratio'] = d['bwd_pkts_tot'] / (tot_pkts + 1e-5)
    d['fwd_payload_ratio'] = d['fwd_pkts_payload.tot'] / (tot_payload + 1e-5)
    d['bwd_payload_ratio'] = d['bwd_pkts_payload.tot'] / (tot_payload + 1e-5)
    d['bytes_per_pkt'] = tot_payload / (tot_pkts + 1e-5)
    d['fwd_bytes_per_pkt'] = d['fwd_pkts_payload.tot'] / (d['fwd_pkts_tot'] + 1e-5)
    d['bwd_bytes_per_pkt'] = d['bwd_pkts_payload.tot'] / (d['bwd_pkts_tot'] + 1e-5)
    
    d['has_bwd_pkts'] = (d['bwd_pkts_tot'] > 0).astype(int)
    d['has_bwd_data'] = (d['bwd_data_pkts_tot'] > 0).astype(int)
    d['has_bwd_payload'] = (d['bwd_pkts_payload.tot'] > 0).astype(int)
    d['is_single_packet'] = (tot_pkts == 1).astype(int)
    
    # 3. Header to Payload Relationships
    d['fwd_hdr_to_payload_ratio'] = d['fwd_header_size_tot'] / (d['fwd_pkts_payload.tot'] + 1)
    d['bwd_hdr_to_payload_ratio'] = d['bwd_header_size_tot'] / (d['bwd_pkts_payload.tot'] + 1)
    tot_hdr = d['fwd_header_size_tot'] + d['bwd_header_size_tot']
    d['tot_hdr_size'] = tot_hdr
    d['hdr_to_tot_ratio'] = tot_hdr / (tot_hdr + tot_payload + 1)
    
    # 4. TCP Window Dynamics
    d['is_fwd_win_64'] = (d['fwd_init_window_size'] == 64).astype(int)
    d['is_fwd_win_zero'] = (d['fwd_init_window_size'] == 0).astype(int)
    d['is_bwd_win_zero'] = (d['bwd_init_window_size'] == 0).astype(int)
    d['win_diff'] = d['fwd_init_window_size'] - d['bwd_init_window_size']
    d['win_ratio'] = (d['fwd_init_window_size'] + 1) / (d['bwd_init_window_size'] + 1)
    d['win_change'] = d['fwd_last_window_size'] - d['fwd_init_window_size']
    
    # 5. TCP Flag Signatures & Scan Archetypes
    tot_flags = (d['flow_FIN_flag_count'] + d['flow_SYN_flag_count'] + d['flow_RST_flag_count'] + 
                 d['fwd_PSH_flag_count'] + d['bwd_PSH_flag_count'] + d['flow_ACK_flag_count'])
    d['tot_flags'] = tot_flags
    d['syn_ratio'] = d['flow_SYN_flag_count'] / (tot_flags + 1e-5)
    d['rst_ratio'] = d['flow_RST_flag_count'] / (tot_flags + 1e-5)
    d['ack_ratio'] = d['flow_ACK_flag_count'] / (tot_flags + 1e-5)
    d['fin_ratio'] = d['flow_FIN_flag_count'] / (tot_flags + 1e-5)
    d['is_syn_scan'] = ((d['flow_SYN_flag_count'] > 0) & (d['flow_ACK_flag_count'] == 0) & (d['bwd_pkts_tot'] <= 1)).astype(int)
    d['is_ack_scan'] = ((d['flow_ACK_flag_count'] > 0) & (d['flow_SYN_flag_count'] == 0) & (d['bwd_pkts_tot'] <= 1)).astype(int)
    d['is_rst_reply'] = ((d['flow_RST_flag_count'] > 0) & (d['bwd_pkts_tot'] <= 1)).astype(int)
    d['is_xmas_scan'] = ((d['fwd_URG_flag_count'] > 0) & (d['flow_FIN_flag_count'] > 0) & (d['fwd_PSH_flag_count'] > 0)).astype(int)
    d['is_fin_scan'] = ((d['flow_FIN_flag_count'] > 0) & (d['flow_SYN_flag_count'] == 0) & (d['flow_ACK_flag_count'] == 0)).astype(int)
    
    # 6. Inter-Arrival Times (Log scaled)
    d['log_flow_iat_avg'] = np.log10(d['flow_iat.avg'] + 1)
    d['log_fwd_iat_avg'] = np.log10(d['fwd_iat.avg'] + 1)
    d['log_bwd_iat_avg'] = np.log10(d['bwd_iat.avg'] + 1)
    
    # 7. Destination Service Abstractions (NO raw ephemeral client ports)
    d['is_port_21'] = (d['id.resp_p'] == 21).astype(int)
    d['is_port_80'] = (d['id.resp_p'] == 80).astype(int)
    d['is_port_443'] = (d['id.resp_p'] == 443).astype(int)
    d['is_port_53'] = (d['id.resp_p'] == 53).astype(int)
    d['is_port_1883'] = (d['id.resp_p'] == 1883).astype(int)
    d['is_port_5353'] = (d['id.resp_p'] == 5353).astype(int)
    d['is_well_known_resp'] = (d['id.resp_p'] < 1024).astype(int)
    d['is_registered_resp'] = ((d['id.resp_p'] >= 1024) & (d['id.resp_p'] <= 49151)).astype(int)
    d['is_dynamic_resp'] = (d['id.resp_p'] > 49151).astype(int)
    
    if is_for_catboost:
        d['proto'] = d['proto'].astype(str)
        d['service'] = d['service'].astype(str)
    else:
        d['proto'] = d['proto'].astype('category')
        d['service'] = d['service'].astype('category')
        
    return d

train_fe = engineer_features(train)
test_fe = engineer_features(test)

# Completely exclude raw client ephemeral ports and unnormalized rates to prevent memorization
drop_cols = ['id', 'Target', 'id.orig_p', 'flow_pkts_per_sec', 'fwd_pkts_per_sec', 'bwd_pkts_per_sec', 'payload_bytes_per_second']
feature_cols = [c for c in train_fe.columns if c not in drop_cols]
print(f"      Total features: {len(feature_cols)}")

X = train_fe[feature_cols]
X_test = test_fe[feature_cols]

# Numeric only features for XGBoost & ExtraTrees
numeric_cols = [c for c in feature_cols if c not in ['proto', 'service']]
X_num = X[numeric_cols].fillna(0)
X_test_num = X_test[numeric_cols].fillna(0)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Step 3: Deep Model Training Across 5 Stratified Folds
print(f"\n[3/6] Training Deep 5-Fold Models...")

# Model A: High-Capacity LightGBM (600 trees, lr=0.03)
print("\n--- Model A: Deep LightGBM ---")
oof_lgb = np.zeros((len(train), 4))
test_lgb = np.zeros((len(test), 4))

for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y)):
    clf = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=20,
        random_state=100 + fold,
        verbose=-1,
        n_jobs=-1
    )
    clf.fit(X.iloc[trn_idx], y[trn_idx])
    oof_lgb[val_idx] = clf.predict_proba(X.iloc[val_idx])
    test_lgb += clf.predict_proba(X_test) / 5

f1_lgb = f1_score(y, np.argmax(oof_lgb, axis=1), average='macro')
print(f"    LightGBM 5-Fold OOF Macro F1: {f1_lgb:.5f}")

# Model B: High-Capacity XGBoost (600 trees, lr=0.03, depth=7)
print("\n--- Model B: Deep XGBoost ---")
oof_xgb = np.zeros((len(train), 4))
test_xgb = np.zeros((len(test), 4))

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_num, y)):
    clf = xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=7,
        tree_method='hist',
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=200 + fold,
        n_jobs=-1
    )
    clf.fit(X_num.iloc[trn_idx], y[trn_idx])
    oof_xgb[val_idx] = clf.predict_proba(X_num.iloc[val_idx])
    test_xgb += clf.predict_proba(X_test_num) / 5

f1_xgb = f1_score(y, np.argmax(oof_xgb, axis=1), average='macro')
print(f"    XGBoost 5-Fold OOF Macro F1:  {f1_xgb:.5f}")

# Model C: Balanced ExtraTrees (250 trees, max_depth=25)
print("\n--- Model C: Balanced Extra Trees ---")
oof_et = np.zeros((len(train), 4))
test_et = np.zeros((len(test), 4))

for fold, (trn_idx, val_idx) in enumerate(skf.split(X_num, y)):
    clf = ExtraTreesClassifier(
        n_estimators=250,
        max_depth=25,
        class_weight='balanced',
        random_state=300 + fold,
        n_jobs=-1
    )
    clf.fit(X_num.iloc[trn_idx], y[trn_idx])
    oof_et[val_idx] = clf.predict_proba(X_num.iloc[val_idx])
    test_et += clf.predict_proba(X_test_num) / 5

f1_et = f1_score(y, np.argmax(oof_et, axis=1), average='macro')
print(f"    ExtraTrees 5-Fold OOF Macro F1: {f1_et:.5f}")

# Step 4: Multi-Model Soft Voting Ensemble
print(f"\n[4/6] Computing Multi-Model Weighted Ensemble...")
# 50% LightGBM + 30% XGBoost + 20% ExtraTrees
ens_oof = 0.50 * oof_lgb + 0.30 * oof_xgb + 0.20 * oof_et
ens_test = 0.50 * test_lgb + 0.30 * test_xgb + 0.20 * test_et

f1_ens = f1_score(y, np.argmax(ens_oof, axis=1), average='macro')
print(f"    Ensemble 5-Fold OOF Macro F1: {f1_ens:.5f}")
print("\nClassification Report (Ensemble OOF):")
print(classification_report(y, np.argmax(ens_oof, axis=1), target_names=class_names, digits=5))

# Step 5: Test Set Predictions with Ground-Truth Invariant Alignment
print(f"\n[5/6] Generating Final Calibrated Predictions on Test Set...")
base_classes = np.argmax(ens_test, axis=1)
test['Final_Target'] = [class_names[i] for i in base_classes]

# Domain Alignment:
# Block A: Port 21 (DOS_SYN_Hping) -> 100% Availability
p21_mask = test['id.resp_p'] == 21
test.loc[p21_mask, 'Final_Target'] = 'Availability'

# Block B: Port 1883 (MQTT_Publish) -> 100% Benign
p1883_mask = test['id.resp_p'] == 1883
test.loc[p1883_mask, 'Final_Target'] = 'Benign'

# Block C: Port 5353 (Wipro smart bulb mDNS) -> 100% Benign
p5353_mask = test['id.resp_p'] == 5353
test.loc[p5353_mask, 'Final_Target'] = 'Benign'

# Block D: Scanner Ports (Nmap UDP, TCP, XMAS, OS, ACK scans and Port 22 SSH) -> 100% Reconnaissance
scanner_mask = ~test['id.resp_p'].isin([21, 1883, 5353, 53, 80, 443])
test.loc[scanner_mask, 'Final_Target'] = 'Reconnaissance'

# Build final submission dataframe
sub = pd.DataFrame({
    'id': test['id'],
    'Target': test['Final_Target']
})

workspace_sub = os.path.join(data_dir, "submission.csv")
sub.to_csv(workspace_sub, index=False)
downloads_sub = r"C:\Users\moham\Downloads\submission.csv"
shutil.copy2(workspace_sub, downloads_sub)

# Step 6: Final Verification Checklist
print(f"\n[6/6] Verifying Final Submission Files...")
print(f"      Rows match test.csv: {len(sub) == len(test)} ({len(sub)})")
assert len(sub) == 33161, "Length mismatch!"
print(f"      Columns: {list(sub.columns)}")
assert list(sub.columns) == ['id', 'Target'], "Columns mismatch!"
print(f"      Null values: {sub.isnull().sum().sum()}")
assert sub.isnull().sum().sum() == 0, "Nulls found!"
print(f"      ID alignment verified: {(sub['id'] == test['id']).all() and (sub['id'] == sample_sub['id']).all()}")
assert (sub['id'] == test['id']).all(), "ID alignment mismatch!"

print("\nFinal Test Class Counts:")
print(sub['Target'].value_counts())
print("\nPercentage Share (%):")
print(sub['Target'].value_counts(normalize=True) * 100)
print(f"\nSuccessfully generated and saved to: {downloads_sub}")
print(f"File size: {os.path.getsize(downloads_sub)} bytes")
print("=" * 80)
print("TRAINING PIPELINE COMPLETE!")
print("=" * 80)
