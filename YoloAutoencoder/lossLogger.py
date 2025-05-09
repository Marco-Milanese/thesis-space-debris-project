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

"""
import json
import pandas as pd
import matplotlib.pyplot as plt

with open("training_logs.json", "r") as f:
    logs = json.load(f)

df = pd.DataFrame.from_dict(logs, orient="index").sort_index()
df.index = df.index.astype(int)

plt.plot(df["train"]["total"], label="Train Total Loss")
plt.plot(df["val"]["total"], label="Val Total Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Over Time")
plt.grid(True)
plt.show()
"""