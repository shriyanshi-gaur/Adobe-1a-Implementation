import os
import json
import pandas as pd

# --- Configuration ---
GT_JSON_DIR = "../data/gt_json"
TRAINING_CSV_PATH = "../data/processed/training_data.csv"

def analyze_data():
    """
    Analyzes and compares the label distribution between the generated training CSV
    and the source ground truth JSON files.
    """
    # --- 1. Count labels in the training_data.csv file ---
    print(f"Analyzing {TRAINING_CSV_PATH}...")
    try:
        df = pd.read_csv(TRAINING_CSV_PATH)
        csv_counts = df['label'].value_counts().to_dict()
    except FileNotFoundError:
        print(f"Error: The file {TRAINING_CSV_PATH} was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return

    # --- 2. Count labels in all gt_json files ---
    print(f"\nAnalyzing ground truth files in {GT_JSON_DIR}...")
    gt_counts = {
        'title': 0, 'h1': 0, 'h2': 0, 'h3': 0, 'h4': 0, 'h5': 0, 'h6': 0
    }
    
    gt_files = [f for f in os.listdir(GT_JSON_DIR) if f.endswith('.json')]
    if not gt_files:
        print("No ground truth JSON files found.")
        return

    for filename in gt_files:
        gt_path = os.path.join(GT_JSON_DIR, filename)
        try:
            with open(gt_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data.get("title"):
                gt_counts['title'] += 1
            
            for item in data.get("outline", []):
                level = item.get("level", "").lower()
                if level in gt_counts:
                    gt_counts[level] += 1
        except Exception as e:
            print(f"Could not process {filename}: {e}")

    # Remove classes with zero counts from ground truth
    gt_counts = {k: v for k, v in gt_counts.items() if v > 0}

    # --- 3. Print the comparison ---
    print("\n--- Data Verification Report ---")
    print("\nCounts from training_data.csv:")
    for label, count in sorted(csv_counts.items()):
        if label != 'other':
            print(f"  - {label}: {count}")

    print("\nExpected counts from all gt_json files:")
    for label, count in sorted(gt_counts.items()):
        print(f"  - {label}: {count}")
    
    print("\n--- Analysis ---")
    discrepancy_found = False
    for label, gt_count in gt_counts.items():
        csv_count = csv_counts.get(label, 0)
        if csv_count != gt_count:
            discrepancy_found = True
            print(f"⚠️ Discrepancy found for '{label}': Expected {gt_count}, but found {csv_count} in the CSV.")
    
    if not discrepancy_found:
        print("✅ No discrepancies found. The CSV counts match the ground truth counts.")
    else:
        print("\nThis mismatch means the labeling process in 'extract_features_from_linewise_json.py' is failing for some labels.")


if __name__ == "__main__":
    analyze_data()