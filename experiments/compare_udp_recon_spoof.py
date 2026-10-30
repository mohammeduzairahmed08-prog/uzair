import pandas as pd
import numpy as np
import os

data_dir = r"C:\Users\moham\.gemini\antigravity\scratch\network_intrusion_detection"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))

recon_udp = train[(train['Target'] == 'Reconnaissance') & (train['proto'] == 'udp')]
print("Train Reconnaissance on UDP (n={}):".format(len(recon_udp)))
cols = ['flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot', 'fwd_pkts_payload.tot', 'bwd_pkts_payload.tot', 'fwd_header_size_tot', 'service']
print(recon_udp[cols].describe().T[['mean', 'std', 'min', '50%', 'max']])
print("Train Reconnaissance UDP services:\n", recon_udp['service'].value_counts())

print("\nCompare with Test 5353 (n=2009):")
test_5353 = test[test['id.resp_p'] == 5353]
print(test_5353[cols].describe().T[['mean', 'std', 'min', '50%', 'max']])
print("Test 5353 services:\n", test_5353['service'].value_counts())

print("\nCompare with Train Spoofing on UDP (n={}):".format(len(train[(train['Target'] == 'Spoofing') & (train['proto'] == 'udp')])))
spoof_udp = train[(train['Target'] == 'Spoofing') & (train['proto'] == 'udp')]
print(spoof_udp[cols].describe().T[['mean', 'std', 'min', '50%', 'max']])
print("Train Spoofing UDP services:\n", spoof_udp['service'].value_counts())
