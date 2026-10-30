import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from scipy.optimize import minimize

data_dir = r"C:\Users\moham\.gemini\antigravity\scratch\network_intrusion_detection"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

class_names = ['Availability', 'Benign', 'Reconnaissance', 'Spoofing']
label_to_id = {name: i for i, name in enumerate(class_names)}
id_to_label = {i: name for i, name in enumerate(class_names)}

y = train['Target'].map(label_to_id).values

# Load predictions
lgb_oof = np.load(os.path.join(data_dir, "lgb_oof.npy"))
lgb_test = np.load(os.path.join(data_dir, "lgb_test.npy"))

xgb_oof = np.load(os.path.join(data_dir, "xgb_oof.npy"))
xgb_test = np.load(os.path.join(data_dir, "xgb_test.npy"))

cb_oof = np.load(os.path.join(data_dir, "cb_oof.npy"))
cb_test = np.load(os.path.join(data_dir, "cb_test.npy"))

print("=== INDIVIDUAL MODEL OOF MACRO F1 SCORES ===")
for name, preds in [('LightGBM', lgb_oof), ('XGBoost', xgb_oof), ('CatBoost', cb_oof)]:
    f1 = f1_score(y, np.argmax(preds, axis=1), average='macro')
    print(f"  {name:<10}: Macro F1 = {f1:.5f}")

# Grid search / Optimization for ensemble blend weights
print("\n--- Optimizing Ensemble Blend Weights ---")
best_blend_f1 = 0
best_weights = None

# Grid search over simplex
weights_grid = []
for w1 in np.linspace(0.2, 0.8, 7):
    for w2 in np.linspace(0.1, 0.6, 6):
        w3 = 1.0 - w1 - w2
        if w3 >= 0.05:
            weights_grid.append((w1, w2, w3))

for w1, w2, w3 in weights_grid:
    blend_oof = w1 * lgb_oof + w2 * xgb_oof + w3 * cb_oof
    score = f1_score(y, np.argmax(blend_oof, axis=1), average='macro')
    if score > best_blend_f1:
        best_blend_f1 = score
        best_weights = (w1, w2, w3)

print(f"Best Blend Weights (LGB, XGB, CB): ({best_weights[0]:.2f}, {best_weights[1]:.2f}, {best_weights[2]:.2f})")
print(f"Blended OOF Macro F1 (raw argmax): {best_blend_f1:.5f}")

w1, w2, w3 = best_weights
ens_oof = w1 * lgb_oof + w2 * xgb_oof + w3 * cb_oof
ens_test = w1 * lgb_test + w2 * xgb_test + w3 * cb_test

# Multi-class threshold / class multiplier optimization
print("\n--- Optimizing Multi-Class Decision Multipliers for Macro F1 ---")
def obj_func(multipliers):
    m = np.array(multipliers)
    scaled_probs = ens_oof * m
    preds = np.argmax(scaled_probs, axis=1)
    return -f1_score(y, preds, average='macro')

# Optimization starting from unit weights
res = minimize(obj_func, [1.0, 1.0, 1.0, 1.0], method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-5})
opt_multipliers = res.x / np.max(res.x)
final_oof_preds = np.argmax(ens_oof * opt_multipliers, axis=1)
final_macro_f1 = f1_score(y, final_oof_preds, average='macro')

print(f"Optimized Multipliers [Avail, Benign, Recon, Spoof]: {opt_multipliers.round(4)}")
print(f"Final Calibrated Ensemble OOF Macro F1: {final_macro_f1:.5f}")

print("\n=== FINAL ENSEMBLE CLASSIFICATION REPORT ===")
print(classification_report(y, final_oof_preds, target_names=class_names, digits=5))

print("\n=== CONFUSION MATRIX ===")
cm = confusion_matrix(y, final_oof_preds)
cm_df = pd.DataFrame(cm, index=[f"Actual {c}" for c in class_names], columns=[f"Pred {c}" for c in class_names])
print(cm_df)

# Generate final test predictions
final_test_scaled = ens_test * opt_multipliers
final_test_classes = np.argmax(final_test_scaled, axis=1)
final_test_labels = [id_to_label[i] for i in final_test_classes]

# Verify submission format
sub = pd.DataFrame({
    'id': test['id'],
    'Target': final_test_labels
})

sub_path = os.path.join(data_dir, "submission.csv")
sub.to_csv(sub_path, index=False)
print(f"\nSaved final submission to: {sub_path}")

print("\n=== SUBMISSION VERIFICATION ===")
print(f"Shape: {sub.shape} (Expected: {sample_sub.shape})")
assert sub.shape == sample_sub.shape, "Shape mismatch!"
assert list(sub.columns) == ['id', 'Target'], "Columns mismatch!"
assert (sub['id'] == test['id']).all(), "ID alignment mismatch!"
assert (sub['id'] == sample_sub['id']).all(), "Sample submission ID mismatch!"
assert sub['Target'].isin(class_names).all(), "Invalid Target labels found!"
assert sub.isnull().sum().sum() == 0, "Null values found!"

print("All format checks passed successfully!")

print("\n=== FINAL TEST PREDICTION DISTRIBUTION ===")
print(sub['Target'].value_counts())
print("\nPercentage distribution:")
print(sub['Target'].value_counts(normalize=True) * 100)

print("\nFirst 10 submission rows:")
print(sub.head(10))
