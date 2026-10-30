import pandas as pd
import numpy as np
import os

data_dir = r"C:\Users\moham\.gemini\antigravity\scratch\network_intrusion_detection"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))

print("=== BEHAVIOR PROFILES IN TRAIN ===")
# Define a behavior profile by key structural attributes
def get_profile(df):
    p = df.copy()
    p['proto_service'] = p['proto'] + "_" + p['service']
    p['resp_p_cat'] = p['id.resp_p'].apply(lambda x: str(int(x)) if x in [21, 53, 80, 443, 1883, 5353] else ('<1024' if x < 1024 else '>1024'))
    p['pkt_pattern'] = p['fwd_pkts_tot'].clip(upper=10).astype(str) + "_" + p['bwd_pkts_tot'].clip(upper=10).astype(str)
    p['win_pattern'] = (p['fwd_init_window_size'] > 0).astype(str) + "_" + (p['bwd_init_window_size'] > 0).astype(str)
    return p['proto_service'] + " | port:" + p['resp_p_cat'] + " | pkts:" + p['pkt_pattern'] + " | win:" + p['win_pattern']

train['profile'] = get_profile(train)
test['profile'] = get_profile(test)

train_profiles = train.groupby('profile')['Target'].value_counts().unstack().fillna(0).astype(int)
train_profiles['Total'] = train_profiles.sum(axis=1)
train_profiles = train_profiles.sort_values('Total', ascending=False)

print(f"Total distinct profiles in Train: {len(train_profiles)}")
print("\nTop 20 profiles in Train:")
print(train_profiles.head(20).to_string())

test_profiles = test['profile'].value_counts()
print(f"\nTotal distinct profiles in Test: {len(test_profiles)}")
print("\nTop 20 profiles in Test:")
print(test_profiles.head(20).to_string())

unseen_test_profiles = set(test_profiles.index) - set(train_profiles.index)
print(f"\nNovel test profiles not seen in train: {len(unseen_test_profiles)}")
unseen_count = test[test['profile'].isin(unseen_test_profiles)].shape[0]
print(f"Total test rows with novel profiles: {unseen_count} ({unseen_count/len(test)*100:.2f}%)")

print("\nTop novel profiles in Test:")
for prof in test_profiles.loc[list(unseen_test_profiles)].sort_values(ascending=False).head(15).index:
    cnt = test_profiles[prof]
    print(f"  [{cnt:5d}] {prof}")
