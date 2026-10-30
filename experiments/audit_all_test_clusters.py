import pandas as pd
import numpy as np
import os

data_dir = r"C:\Users\moham\.gemini\antigravity\scratch\network_intrusion_detection"
train = pd.read_csv(os.path.join(data_dir, "train.csv"))
test = pd.read_csv(os.path.join(data_dir, "test.csv"))
sub = pd.read_csv(os.path.join(data_dir, "submission.csv"))

test['Predicted'] = sub['Target']

# Group test by proto, service, id.resp_p, and check prediction distribution
grouped = test.groupby(['proto', 'service', 'id.resp_p'])['Predicted'].value_counts().unstack().fillna(0).astype(int)
grouped['Total'] = grouped.sum(axis=1)
grouped = grouped.sort_values('Total', ascending=False)

print("Top 25 test clusters (proto, service, resp_p):")
print(grouped.head(25).to_string())
