import pandas as pd
import numpy as np
import os

data_dir = r"C:\Users\moham\.gemini\antigravity\scratch\network_intrusion_detection"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))

print("=== TARGET DISTRIBUTION & GROUP MEAN SUMMARY ===")
features_to_check = [
    'flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot', 'flow_pkts_per_sec',
    'fwd_pkts_payload.avg', 'bwd_pkts_payload.avg', 'payload_bytes_per_second',
    'flow_SYN_flag_count', 'flow_RST_flag_count', 'flow_ACK_flag_count', 'flow_FIN_flag_count',
    'fwd_init_window_size', 'bwd_init_window_size', 'fwd_last_window_size',
    'fwd_iat.avg', 'bwd_iat.avg', 'flow_iat.avg', 'active.avg', 'idle.avg'
]

summary = train.groupby('Target')[features_to_check].mean().round(4)
print(summary.T)

print("\n=== RECONNAISSANCE SAMPLES BREAKDOWN ===")
recon = train[train['Target'] == 'Reconnaissance']
print("Recon proto:\n", recon['proto'].value_counts())
print("Recon service:\n", recon['service'].value_counts())
print("Recon top 10 resp_p:\n", recon['id.resp_p'].value_counts().head(10))
print("Recon SYN count:\n", recon['flow_SYN_flag_count'].value_counts())
print("Recon RST count:\n", recon['flow_RST_flag_count'].value_counts())
print("Recon ACK count:\n", recon['flow_ACK_flag_count'].value_counts())
print("Recon fwd_pkts_tot:\n", recon['fwd_pkts_tot'].describe())

print("\n=== SPOOFING SAMPLES BREAKDOWN ===")
spoof = train[train['Target'] == 'Spoofing']
print("Spoof proto:\n", spoof['proto'].value_counts())
print("Spoof service:\n", spoof['service'].value_counts())
print("Spoof top 10 resp_p:\n", spoof['id.resp_p'].value_counts().head(10))

print("\n=== BENIGN SAMPLES BREAKDOWN ===")
benign = train[train['Target'] == 'Benign']
print("Benign proto:\n", benign['proto'].value_counts())
print("Benign service:\n", benign['service'].value_counts())
print("Benign top 10 resp_p:\n", benign['id.resp_p'].value_counts().head(10))

print("\n=== AVAILABILITY SAMPLES BREAKDOWN ===")
avail = train[train['Target'] == 'Availability']
print("Avail proto:\n", avail['proto'].value_counts())
print("Avail service:\n", avail['service'].value_counts())
print("Avail top 10 resp_p:\n", avail['id.resp_p'].value_counts().head(10))
non_21_avail = avail[avail['id.resp_p'] != 21.0]
print(f"Avail with resp_p != 21: {len(non_21_avail)}")
if len(non_21_avail) > 0:
    print(non_21_avail['id.resp_p'].value_counts().head(10))
    print(non_21_avail['service'].value_counts())
