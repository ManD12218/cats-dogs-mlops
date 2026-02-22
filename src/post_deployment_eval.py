import csv
import pandas as pd

# Load prediction logs
log_file = "prediction_logs.csv"

try:
    df = pd.read_csv(log_file, header=None)
except FileNotFoundError:
    print("No prediction logs found.")
    exit()

df.columns = ["timestamp", "predicted_label", "confidence", "latency"]

# Simulated true labels (for demo purposes)
# Replace these with real labels if available
true_labels = ["dog", "cat", "dog", "dog", "cat"]

# Use only first N predictions to match true labels
df = df.head(len(true_labels))
df["true_label"] = true_labels[:len(df)]

# Calculate accuracy
correct = (df["predicted_label"] == df["true_label"]).sum()
accuracy = correct / len(df)

print("\nPost-Deployment Evaluation")
print("--------------------------")
print(f"Total evaluated samples: {len(df)}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.2f}")