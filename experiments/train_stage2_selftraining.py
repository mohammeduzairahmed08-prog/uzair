"""
Stage 2: Self-Training / Pseudo-Label Augmented Retraining
Augmenting training distribution with high-confidence test domain flows
to adapt feature representations to test traffic profiles.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import shutil

print("=" * 80)
print("STAGE 2: PSEUDO-LABEL AUGMENTED RETRAINING (SELF-TRAINING)")
print("=" * 80)

data_dir = r"C:\Users\moham\.gemini\antigravity\scratch\network_intrusion_detection"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

class_names = ['Availability', 'Benign', 'Reconnaissance', 'Spoofing']
label_to_id = {name: i for i, name in enumerate(class_names)}
id_to_label = {i: name for i, name in enumerate(class_names)}

from train_clean_ensemble import engineer_clean_features, drop_cols

print("\n1. Engineering features on train and test...")
train_fe = engineer_clean_features(train)
test_fe = engineer_clean_features(test)

feature_cols = [c for c in train_fe.columns if c not in drop_cols]
numeric_cols = [c for c in feature_cols if c not in ['proto', 'service']]

# 2. Select high-confidence invariant test samples for pseudo-label augmentation:
p21_mask = test['id.resp_p'] == 21                # 10,396 Availability
p1883_mask = test['id.resp_p'] == 1883            # 3,181 Benign
p5353_mask = test['id.resp_p'] == 5353            # 2,009 Benign
scanner_mask = ~test['id.resp_p'].isin([21, 1883, 5353, 53, 80, 443]) # 6,881 Recon

pseudo_mask = p21_mask | p1883_mask | p5353_mask | scanner_mask
test_pseudo = test_fe.loc[pseudo_mask].copy()

test_pseudo['Target'] = 'Reconnaissance'
test_pseudo.loc[test_pseudo['id.resp_p'] == 21, 'Target'] = 'Availability'
test_pseudo.loc[test_pseudo['id.resp_p'] == 1883, 'Target'] = 'Benign'
test_pseudo.loc[test_pseudo['id.resp_p'] == 5353, 'Target'] = 'Benign'

# Sample a balanced subset of pseudo labels
p21_idx = test_pseudo[test_pseudo['id.resp_p'] == 21].sample(n=3000, random_state=42).index
other_pseudo_idx = test_pseudo[test_pseudo['id.resp_p'] != 21].index
balanced_pseudo_idx = p21_idx.union(other_pseudo_idx)

pseudo_subset = test_pseudo.loc[balanced_pseudo_idx]
print(f"\n   Balanced pseudo-label subset size: {len(pseudo_subset)}")
print(pseudo_subset['Target'].value_counts())

# Augment training set
augmented_train = pd.concat([train_fe, pseudo_subset], axis=0, ignore_index=True)
for c in ['proto', 'service']:
    augmented_train[c] = augmented_train[c].astype('category')
    test_fe[c] = test_fe[c].astype('category')

print(f"\n3. Augmented Training Set Shape: {augmented_train.shape}")
print(augmented_train['Target'].value_counts())

y_aug = augmented_train['Target'].map(label_to_id).values
X_aug = augmented_train[feature_cols]
X_test = test_fe[feature_cols]

X_aug_num = augmented_train[numeric_cols].fillna(0)
X_test_num = test_fe[numeric_cols].fillna(0)

# 4. Train Deep Models on Augmented Domain Data
print("\n4. Training Self-Trained LightGBM (800 trees) on Augmented Dataset...")
clf_lgb_st = lgb.LGBMClassifier(
    n_estimators=800,
    learning_rate=0.025,
    num_leaves=63,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)
clf_lgb_st.fit(X_aug, y_aug)
preds_lgb_st = clf_lgb_st.predict_proba(X_test)
print("   LightGBM Self-Training completed!")

print("\n5. Training Self-Trained XGBoost (800 trees) on Augmented Dataset...")
clf_xgb_st = xgb.XGBClassifier(
    n_estimators=800,
    learning_rate=0.025,
    max_depth=7,
    tree_method='hist',
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    n_jobs=-1
)
clf_xgb_st.fit(X_aug_num, y_aug)
preds_xgb_st = clf_xgb_st.predict_proba(X_test_num)
print("   XGBoost Self-Training completed!")

# 6. Ensemble Predictions
print("\n6. Blending Self-Trained Predictions (50% LGB + 50% XGB)...")
final_probs = 0.50 * preds_lgb_st + 0.50 * preds_xgb_st
final_labels_idx = np.argmax(final_probs, axis=1)
test['Stage2_Target'] = [class_names[i] for i in final_labels_idx]

# Apply Domain Invariant Overrides
test.loc[p21_mask, 'Stage2_Target'] = 'Availability'
test.loc[p1883_mask, 'Stage2_Target'] = 'Benign'
test.loc[p5353_mask, 'Stage2_Target'] = 'Benign'
test.loc[scanner_mask, 'Stage2_Target'] = 'Reconnaissance'

sub = pd.DataFrame({
    'id': test['id'],
    'Target': test['Stage2_Target']
})

workspace_sub = os.path.join(data_dir, "submission.csv")
sub.to_csv(workspace_sub, index=False)
downloads_sub = r"C:\Users\moham\Downloads\submission.csv"
shutil.copy2(workspace_sub, downloads_sub)

print("\n=== STAGE 2 VERIFICATION ===")
print("Rows match:", len(sub) == len(test), f"({len(sub)})")
assert len(sub) == 33161, "Length mismatch!"
print("Columns match:", list(sub.columns) == ['id', 'Target'])
assert list(sub.columns) == ['id', 'Target'], "Columns mismatch!"
print("Null count:", sub.isnull().sum().sum())
assert sub.isnull().sum().sum() == 0, "Nulls found!"
print("IDs match test:", (sub['id'] == test['id']).all())
assert (sub['id'] == test['id']).all(), "ID mismatch!"

print("\nFinal Stage 2 Class Distribution:")
print(sub['Target'].value_counts())
print("\nPercentage Distribution (%):")
print(sub['Target'].value_counts(normalize=True) * 100)
print(f"\nSuccessfully updated file: {downloads_sub}")
print(f"File size: {os.path.getsize(downloads_sub)} bytes")
print("=" * 80)
print("STAGE 2 RETRAINING COMPLETE!")
print("=" * 80)
