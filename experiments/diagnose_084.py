import pandas as pd
import numpy as np
import os
from sklearn.metrics import f1_score

data_dir = r"C:\Users\moham\.gemini\antigravity\scratch\network_intrusion_detection"
sub = pd.read_csv(os.path.join(data_dir, "submission.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))

print("Current submission distribution:")
print(sub['Target'].value_counts())

# Let's inspect the test clusters
test['pred'] = sub['Target']

# Check how test is partitioned
# What are the major subsets?
# 1. Port 21 (10396 rows): all predicted Availability
# 2. Port 1883 (3181 rows): all predicted Benign (MQTT)
# 3. Novel ports (5854 rows): all predicted Reconnaissance
# 4. Port 5353 (2009 rows): 1920 predicted Spoofing, 77 Benign
# 5. Port 53 (6216 rows): 3572 Benign, 2640 Spoofing
# 6. Port 80 (2644 rows): 2465 Benign, 170 Spoofing
# 7. Port 443 (1834 rows): 1102 Benign, 704 Spoofing

print("\n--- Summary of the 7 test blocks ---")
blocks = {
    'Port 21': test['id.resp_p'] == 21,
    'Port 1883': test['id.resp_p'] == 1883,
    'Port 53': test['id.resp_p'] == 53,
    'Port 80': test['id.resp_p'] == 80,
    'Port 443': test['id.resp_p'] == 443,
    'Port 5353': test['id.resp_p'] == 5353,
    'Unseen Ports': ~test['id.resp_p'].isin([21, 1883, 53, 80, 443, 5353])
}

for name, mask in blocks.items():
    print(f"{name:<15} (n={mask.sum():<5}): {test.loc[mask, 'pred'].value_counts().to_dict()}")
