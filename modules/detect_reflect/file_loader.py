# pipeline/file_loader.py
import json
import pandas as pd
import os
import numpy as np
from datetime import datetime

def load_data(filepath):
    if os.path.isdir(filepath):
        result = {}
        for file in os.listdir(filepath):
            file_path = os.path.join(filepath, file)
            if os.path.isfile(file_path) and any(file_path.endswith(ext) for ext in [".json", ".csv", ".xlsx"]):
                try:
                    data = load_data(file_path)
                    result.update(data)
                except ValueError:
                    continue
        return result
    elif filepath.endswith(".json"):
        with open(filepath, "r") as f:
            return json.load(f)
    elif filepath.endswith(".csv"):
        df = pd.read_csv(filepath, low_memory=False)
        df = pd.read_csv(filepath, low_memory=False)
        filename = filepath.split("/")[-1].split(".")[0]
        return {filename: {"metadata": {}, "columns": {col: {"records": convert_to_serializable(values[:20])} for col, values in df.to_dict(orient="list").items()}}}
    
    elif filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath)
        filename = filepath.split("/")[-1].split(".")[0]
        return {filename: {"metadata": {}, "columns": {col: {"records": convert_to_serializable(values[:20])} for col, values in df.to_dict(orient="list").items()}}}
    
    else:
        raise ValueError("Unsupported file format")

def convert_to_serializable(values):
    """Convert pandas/numpy data types to JSON serializable types"""
    result = []
    for val in values:
        if isinstance(val, (np.integer, np.floating)):
            result.append(float(val) if isinstance(val, np.floating) else int(val))
        elif isinstance(val, np.bool_):
            result.append(bool(val))
        elif isinstance(val, (pd.Timestamp, np.datetime64)):
            result.append(val.isoformat() if hasattr(val, 'isoformat') else str(val))
        elif pd.isna(val):
            result.append(None)
        else:
            result.append(str(val))
    return result
