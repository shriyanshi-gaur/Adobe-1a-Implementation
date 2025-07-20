import os
import json
import pandas as pd
import time
# Import shared utility functions
from utils import (
    normalize, fuzzy_match, is_title_case, is_uppercase,
    ends_with_colon, starts_with_number, word_count, has_bullet_prefix
)

# Paths to folders
LINEWISE_DIR = "../data/linewise_json"
GT_DIR = "../data/gt_json"
OUTPUT_CSV = "../data/processed/training_data.csv"


def label_line(line_obj, page_number, gt_data):
    """Labels a single line based on fuzzy matching against ground truth data."""
    current_line_text = line_obj["text"].strip()

    # Match Title (only on page 1)
    if page_number == 1 and gt_data.get("title"):
        if fuzzy_match(gt_data["title"], current_line_text, threshold=0.85):
            return "title"

    # Match Outline Headings
    for item in gt_data.get("outline", []):
        gt_item_text = item.get("text", "")
        gt_item_page = item.get("page")

        if gt_item_page == page_number and fuzzy_match(gt_item_text, current_line_text, threshold=0.85):
            return item.get("level", "other").lower()

    return "other"


def extract_features(line_obj, page_number, page_width, page_height, next_line_y=None, prev_line_y_bottom=None):
    """Extracts a feature dictionary from a single line object."""
    spans = line_obj.get("spans", [])
    if not spans:
        return None

    first_span = spans[0]
    avg_font_size = sum(s["size"] for s in spans) / len(spans)
    is_bold = int(any("bold" in s["font"].lower() for s in spans))
    is_italic = int(any("italic" in s["font"].lower() for s in spans))

    y_pos = line_obj["y"]
    x_pos = line_obj["bbox"][0]
    line_height = line_obj["bbox"][3] - line_obj["bbox"][1]

    space_after_line = (next_line_y - (y_pos + line_height)) if next_line_y is not None else 0
    space_before_line = (y_pos - prev_line_y_bottom) if prev_line_y_bottom is not None else 0

    text_content = line_obj["text"]
    
    features = {
        "text": text_content,
        "font_size": avg_font_size,
        "is_bold": is_bold,
        "is_italic": is_italic,
        "y_pos": y_pos,
        "x_pos": x_pos,
        "line_height": max(0, line_height),
        "space_after_line": max(0, space_after_line),
        "space_before_line": max(0, space_before_line),
        "page": page_number,
        "is_title_case": int(is_title_case(text_content)),
        "is_upper": int(is_uppercase(text_content)),
        "ends_colon": int(ends_with_colon(text_content)),
        "starts_number": int(starts_with_number(text_content)),
        "word_count": word_count(text_content),
        "has_bullet_prefix": int(has_bullet_prefix(text_content)),
    }
    return features

def process_pair(linewise_path, gt_path):
    # This function remains largely the same as your original, but is simplified
    # for clarity and now calls the refactored extract_features.
    with open(linewise_path, "r", encoding="utf-8") as f:
        linewise_data = json.load(f)
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    rows = []
    for page_data in linewise_data.get("pages", []):
        page_number = page_data["page_number"]
        lines = page_data.get("lines", [])
        page_width = page_data.get("page_width", 595.0)
        page_height = page_data.get("page_height", 842.0)
        
        prev_line_y_bottom = None
        for i, line_obj in enumerate(lines):
            if not line_obj.get("spans") or not line_obj.get("bbox"):
                continue

            next_line_y = lines[i+1]["y"] if i + 1 < len(lines) else None
            features = extract_features(line_obj, page_number, page_width, page_height, next_line_y, prev_line_y_bottom)

            if features:
                label = label_line(line_obj, page_number, gt_data)
                features["label"] = label
                rows.append(features)
                prev_line_y_bottom = line_obj["y"] + features["line_height"]

    return rows

def main():
    # The main loop logic is correct and remains unchanged.
    all_rows = []
    print("Starting feature extraction for training data...")
    for file in os.listdir(LINEWISE_DIR):
        if file.endswith("_linewise.json"):
            base_filename = file.replace("_linewise.json", "")
            linewise_path = os.path.join(LINEWISE_DIR, file)
            gt_path = os.path.join(GT_DIR, f"{base_filename}.json")
            
            if os.path.exists(gt_path):
                print(f"Processing: {base_filename}")
                rows = process_pair(linewise_path, gt_path)
                all_rows.extend(rows)
            else:
                print(f"Warning: GT file not found for {file}. Skipping.")

    if not all_rows:
        print("No data processed. Exiting.")
        return

    df = pd.DataFrame(all_rows)
    df = df[df['text'].str.strip() != '']
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Training data successfully saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()