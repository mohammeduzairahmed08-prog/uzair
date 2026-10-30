# Network Intrusion Detection

A machine learning solution for detecting and classifying network intrusions from IoT traffic flows into four categories:
- **Benign**: Normal IoT device communications and sensor telemetry
- **Availability**: Denial of Service attacks targeting service uptime (SYN floods, Slowloris)
- **Reconnaissance**: Network scanning, host discovery, and port probing (Nmap)
- **Spoofing**: Address impersonation and Man-in-the-Middle traffic (ARP cache poisoning)

**Evaluation Metric:** Macro-Averaged F1 Score

---

## Methodology

Network flow classification poses distinct challenges: heavy-tailed continuous distributions, high protocol cardinality, and subtle behavioral distinctions between attacks and legitimate services. This pipeline addresses them through a domain-driven machine learning workflow:

### 1. Domain Feature Engineering (`feature_engineering.py`)
Rather than relying purely on raw statistical aggregations, the feature pipeline extracts 131 protocol-aware features:
- **Port Semantics**: IANA categorization (well-known `<1024`, registered `1024-49151`, dynamic/ephemeral `>=49152`) and service-specific indicators (MQTT, mDNS, HTTP, HTTPS, DNS, FTP, SSH).
- **Traffic Asymmetry**: Forward-to-backward packet and byte ratios, payload-to-header size densities, and half-duplex flow flags.
- **TCP Flag Dynamics**: SYN-only flags, RST-to-SYN ratios, cumulative PSH/ACK dynamics, and abnormal flag combinations.
- **Timing Mechanics**: Log-scale inter-arrival time (IAT) and flow duration transforms (`np.log1p`) to normalize heavy-tailed timing distributions.

### 2. Leakage-Free Validation Strategy
Network datasets frequently contain repetitive flow patterns or duplicate packet bursts. A standard random K-Fold cross-validation leaks information between folds and produces overly optimistic validation scores. 

To ensure realistic generalization, we implement **5-Fold `StratifiedGroupKFold`** grouped on flow content hashes. This ensures that identical or retransmitted flow groups remain strictly within either train or validation sets.

### 3. Multi-Model GBDT Ensemble (`train_ensemble.py`)
- **Diverse Gradient Boosting**: Ensembles LightGBM, XGBoost, and CatBoost models trained across stratified group folds.
- **Data Augmentation**: Simulates packet-drop behavior under heavy server load to handle boundary cases where backward responses are suppressed.
- **Decision Threshold Optimization**: Applies coordinate-ascent weight tuning directly on out-of-fold probability distributions to maximize the competition metric (Macro F1).

### 4. Hybrid Flow Classification & Verification (`solution_pipeline.py`)
- Combines domain-specific structural invariants with calibrated ensemble probabilities.
- Runs an automated verification suite checking row integrity, label vocabulary, and ID alignment prior to generating outputs.

---

## Project Structure

```
gdgc/
├── main.py                  # Master entry point
├── solution_pipeline.py     # End-to-end classification pipeline & verification suite
├── train_ensemble.py        # 5-fold StratifiedGroupKFold GBDT training script
├── feature_engineering.py   # Protocol semantics and traffic asymmetry feature extractor
├── eda_analysis.py          # Data profiling and traffic inspection tool
├── submission.csv           # Final predicted test labels (33,161 rows)
├── requirements.txt         # Project dependencies
└── data/                    # Datasets (train.csv, test.csv, sample_submission.csv)
```

---

## Getting Started

### 1. Requirements
Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Generate Predictions
To run the full pipeline and generate the verified `submission.csv`:

```bash
python main.py
```

### 3. Model Training
To train the gradient boosting ensemble from scratch with 5-fold group cross-validation:

```bash
python main.py --mode train
```

### 4. Data Profiling
To inspect class distributions, core responder ports, and protocols:

```bash
python main.py --mode eda
```

---

## Performance Summary

| Class | Traffic Type | Predicted Distribution | Target F1 |
| :--- | :--- | :--- | :--- |
| **Availability** | SYN Floods, Slowloris HTTP DoS | 10,549 (31.81%) | > 0.99 |
| **Benign** | IoT Telemetry (MQTT, ThingSpeak, mDNS) | 10,428 (31.45%) | > 0.99 |
| **Reconnaissance** | Port Scans (UDP, TCP, XMAS, OS detection) | 6,887 (20.77%) | > 0.99 |
| **Spoofing** | ARP Poisoning & MITM Queries | 5,297 (15.97%) | > 0.99 |
| **Overall** | **Macro-Averaged F1** | **33,161 flows** | **> 0.99** |
