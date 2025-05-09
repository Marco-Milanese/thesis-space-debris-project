import json
import os

def logLosses(logFile, epoch, trainLosses, valLosses):
    """
    Logs training and validation losses to a JSON file.

    Args:
        logFile (str): Path to the JSON log file.
        epoch (int): Current epoch number.
        trainLosses (dict): Dict of training losses.
        valLosses (dict): Dict of validation losses.
    """
    logData = {}

    # Load existing log file if it exists
    if os.path.exists(logFile):
        with open(logFile, 'r') as f:
            logData = json.load(f)

    # Add current epoch data
    logData[str(epoch)] = {
        "train": trainLosses,
        "val": valLosses
    }

    # Save updated log file
    with open(logFile, 'w') as f:
        json.dump(logData, f, indent=4)


import json
import pandas as pd
import matplotlib.pyplot as plt

# Load JSON
with open("training_logs.json", "r") as f:
    logs = json.load(f)

# Flatten the nested structure
flattened = []
for epoch, data in logs.items():
    row = {"epoch": int(epoch)}
    for split in ["train", "val"]:
        for key, value in data[split].items():
            row[f"{split}_{key}"] = value
    flattened.append(row)

# Create DataFrame
df = pd.DataFrame(flattened[1:]).sort_values("epoch").set_index("epoch")

# Plotting
plt.plot(df["train_total"], label="Train Total Loss")
plt.plot(df["val_total"], label="Val Total Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Over Time")
plt.grid(True)
plt.show()

