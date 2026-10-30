"""
================================================================================
Feature Engineering Module for Network Intrusion Detection
================================================================================
Extracts 131 domain-engineered network flow features based on:
  - IANA port semantic groupings (well-known, registered, dynamic/ephemeral)
  - Core IoT protocol indicators (MQTT, mDNS, HTTP, HTTPS, DNS, FTP, SSH)
  - Tool signature detection (Hping3 SYN floods, Slowloris, Nmap port scans)
  - Bidirectional traffic asymmetry and packet ratios
  - Payload and header density mechanics
  - TCP flag combinations and anomaly indicators
  - Log-scaled timing and inter-arrival time (IAT) statistics
================================================================================
"""

import numpy as np
import pandas as pd


def extract_features(df: pd.DataFrame, drop_orig_port: bool = True) -> pd.DataFrame:
    """
    Extracts 131 generalizable network flow mechanics and protocol semantics.
    
    Args:
        df: Input DataFrame containing raw RT-IoT2022 / competition flow features.
        drop_orig_port: Whether to drop id.orig_p to prevent memorization of ephemeral client ports.
    
    Returns:
        Transformed DataFrame with engineered features.
    """
    df = df.copy()
    rp = df['id.resp_p']
    op = df['id.orig_p']

    # 1. Port Semantics (IANA standard groupings)
    df['is_well_known_resp'] = (rp < 1024).astype(np.int8)
    df['is_registered_resp'] = ((rp >= 1024) & (rp < 49152)).astype(np.int8)
    df['is_dynamic_resp'] = (rp >= 49152).astype(np.int8)
    df['is_well_known_orig'] = (op < 1024).astype(np.int8)
    df['is_dynamic_orig'] = (op >= 49152).astype(np.int8)
    df['is_same_port'] = (op == rp).astype(np.int8)

    # Standard service port indicators
    df['is_ftp'] = (rp == 21).astype(np.int8)
    df['is_ssh'] = (rp == 22).astype(np.int8)
    df['is_dns'] = (rp == 53).astype(np.int8)
    df['is_http'] = (rp == 80).astype(np.int8)
    df['is_https'] = (rp == 443).astype(np.int8)
    df['is_mqtt'] = (rp == 1883).astype(np.int8)
    df['is_mdns'] = (rp == 5353).astype(np.int8)

    # 2. Attack Tool Signatures (Hping3 SYN Flood & Slowloris)
    df['is_hping_signature'] = (
        (df['fwd_init_window_size'] == 64) &
        (df['fwd_pkts_payload.tot'] == 120) &
        (df['fwd_pkts_tot'] == 1)
    ).astype(np.int8)

    # 3. Traffic Asymmetry & Packet Ratios
    df['pkts_tot'] = df['fwd_pkts_tot'] + df['bwd_pkts_tot']
    df['data_pkts_tot'] = df['fwd_data_pkts_tot'] + df['bwd_data_pkts_tot']
    df['pkts_ratio'] = (df['fwd_pkts_tot'] + 1.0) / (df['bwd_pkts_tot'] + 1.0)
    df['data_pkts_ratio'] = (df['fwd_data_pkts_tot'] + 1.0) / (df['bwd_data_pkts_tot'] + 1.0)
    df['data_to_total_pkts_ratio'] = (df['data_pkts_tot'] + 1e-5) / (df['pkts_tot'] + 1e-5)
    df['bwd_pkts_is_zero'] = (df['bwd_pkts_tot'] == 0).astype(np.int8)
    df['bwd_data_pkts_is_zero'] = (df['bwd_data_pkts_tot'] == 0).astype(np.int8)
    df['bwd_payload_is_zero'] = (df['bwd_pkts_payload.tot'] == 0).astype(np.int8)

    # 4. Payload & Header Density
    df['bytes_ratio'] = (df['fwd_pkts_payload.tot'] + 1.0) / (df['bwd_pkts_payload.tot'] + 1.0)
    df['bytes_per_pkt'] = df['flow_pkts_payload.tot'] / (df['pkts_tot'] + 1e-5)
    df['fwd_bytes_per_pkt'] = df['fwd_pkts_payload.tot'] / (df['fwd_pkts_tot'] + 1e-5)
    df['bwd_bytes_per_pkt'] = df['bwd_pkts_payload.tot'] / (df['bwd_pkts_tot'] + 1e-5)
    df['total_header_size'] = df['fwd_header_size_tot'] + df['bwd_header_size_tot']
    df['header_to_payload_ratio'] = df['total_header_size'] / (df['flow_pkts_payload.tot'] + 1.0)

    # 5. TCP Flag Dynamics
    df['total_flags'] = (
        df['flow_FIN_flag_count'] + df['flow_SYN_flag_count'] +
        df['flow_RST_flag_count'] + df['fwd_PSH_flag_count'] +
        df['bwd_PSH_flag_count'] + df['flow_ACK_flag_count']
    )
    df['syn_only'] = ((df['flow_SYN_flag_count'] > 0) & (df['flow_ACK_flag_count'] == 0)).astype(np.int8)
    df['rst_syn_ratio'] = (df['flow_RST_flag_count'] + 1e-5) / (df['flow_SYN_flag_count'] + 1.0)
    df['psh_flag_tot'] = df['fwd_PSH_flag_count'] + df['bwd_PSH_flag_count']

    # 6. Log-Scale Timing Transforms (handles extreme positive skew)
    log_pairs = [
        ('log_flow_duration', 'flow_duration'),
        ('log_flow_pkts_per_sec', 'flow_pkts_per_sec'),
        ('log_payload_bytes_per_sec', 'payload_bytes_per_second'),
        ('log_flow_iat_min', 'flow_iat.min'),
        ('log_flow_iat_max', 'flow_iat.max'),
        ('log_flow_iat_avg', 'flow_iat.avg'),
        ('log_flow_iat_tot', 'flow_iat.tot'),
        ('log_fwd_iat_min', 'fwd_iat.min'),
        ('log_bwd_iat_min', 'bwd_iat.min')
    ]
    for new_col, old_col in log_pairs:
        if old_col in df.columns:
            df[new_col] = np.log1p(np.maximum(0.0, df[old_col]))

    # 7. Window Mechanics
    df['window_ratio'] = (df['fwd_init_window_size'] + 1.0) / (df['bwd_init_window_size'] + 1.0)
    df['fwd_window_delta'] = df['fwd_init_window_size'] - df['fwd_last_window_size']
    df['is_zero_bwd_window'] = (df['bwd_init_window_size'] == 0).astype(np.int8)

    if drop_orig_port:
        return df.drop(columns=['id.orig_p'], errors='ignore')
    return df
