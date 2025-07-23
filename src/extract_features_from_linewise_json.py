import os
import json
import pandas as pd
import re
from utils import fuzzy_match, is_title_case, is_uppercase, ends_with_colon, starts_with_number, word_count, has_bullet_prefix

LINEWISE_DIR = "../data/linewise_json"
GT_DIR = "../data/gt_json"
OUTPUT_CSV = "../data/processed/training_data.csv"

def is_conventional_heading(text):
    """Checks for common heading patterns using domain knowledge."""
    text = text.strip().lower()
    if re.match(r'^(appendix\s+[a-z\d]+[:.]?|section\s+\d+[:.]?|chapter\s+\d+[:.]?|\d+\.\s+)', text):
        return 1
    return 0

def label_line(lines, current_index, page_number, gt_title, gt_outline, used_gt_indices):
    """
    Statefully labels a line, returning the label, lines consumed, and the index of the matched GT item.
    It will not match against GT items whose indices are in used_gt_indices.
    """
    current_line_obj = lines[current_index]
    current_line_text = current_line_obj["text"].strip()

    # --- Check for Two-line Matches First ---
    if current_index + 1 < len(lines):
        next_line_text = lines[current_index + 1]["text"].strip()
        combined_text = f"{current_line_text} {next_line_text}"
        
        # Check title first (special case, index -1)
        if gt_title and page_number == 1 and -1 not in used_gt_indices and fuzzy_match(gt_title, combined_text):
            return ("title", 2, -1)
        
        # Check outline headings
        for i, item in enumerate(gt_outline):
            if i in used_gt_indices: continue
            if item.get("page") == page_number and fuzzy_match(item.get("text", ""), combined_text):
                return (item.get("level", "other").lower(), 2, i)

    # --- If no two-line match, check for Single-line Matches ---
    if gt_title and page_number == 1 and -1 not in used_gt_indices and fuzzy_match(gt_title, current_line_text):
        return ("title", 1, -1)
        
    for i, item in enumerate(gt_outline):
        if i in used_gt_indices: continue
        if item.get("page") == page_number and fuzzy_match(item.get("text", ""), current_line_text):
            return (item.get("level", "other").lower(), 1, i)

    return ("other", 1, None)

def extract_features(line_obj, page_number, page_height, page_width, next_line_y=None, prev_line_y_bottom=None):
    """
    Extracts a complete set of features for a given line object.
    This function is consistent with the inference script.
    """
    spans = line_obj.get("spans", [])
    if not spans: return None
    text_content = line_obj["text"]
    avg_font_size = sum(s["size"] for s in spans) / len(spans)
    is_bold = int(any("bold" in s["font"].lower() for s in spans))
    is_italic = int(any("italic" in s["font"].lower() for s in spans))
    y_pos, x_pos = line_obj["y"], line_obj["bbox"][0]
    line_height = line_obj["bbox"][3] - line_obj["bbox"][1]
    space_after_line = (next_line_y - (y_pos + line_height)) if next_line_y is not None else 0
    space_before_line = (y_pos - prev_line_y_bottom) if prev_line_y_bottom is not None else 0
    
    return {
        "text": text_content, "font_size": avg_font_size, "is_bold": is_bold,
        "is_italic": is_italic, "x_pos": x_pos, "y_pos": y_pos,
        "line_height": max(0, line_height), "space_after_line": max(0, space_after_line),
        "space_before_line": max(0, space_before_line), "page": page_number,
        "y_pos_normalized": y_pos / page_height if page_height > 0 else 0,
        "x_pos_normalized": x_pos / page_width if page_width > 0 else 0,
        "is_left_aligned": int(x_pos < 100),
        "normalized_space_after_line": max(0, space_after_line) / avg_font_size if avg_font_size > 0 else 0,
        "normalized_space_before_line": max(0, space_before_line) / avg_font_size if avg_font_size > 0 else 0,
        "is_title_case": int(is_title_case(text_content)), "is_upper": int(is_uppercase(text_content)),
        "ends_colon": int(ends_with_colon(text_content)), "starts_number": int(starts_with_number(text_content)),
        "word_count": word_count(text_content), "has_bullet_prefix": int(has_bullet_prefix(text_content)),
        "is_conventional_heading": is_conventional_heading(text_content),
    }

def process_pair(linewise_path, gt_path):
    """
    UPDATED: Now manages a set of used ground truth indices to prevent reuse and overcounting.
    """
    with open(linewise_path, "r", encoding="utf-8") as f: linewise_data = json.load(f)
    with open(gt_path, "r", encoding="utf-8") as f: gt_data = json.load(f)
    
    rows = []
    used_gt_indices = set()
    gt_outline = gt_data.get("outline", [])
    gt_title = gt_data.get("title")

    for page_data in linewise_data.get("pages", []):
        page_number = page_data["page_number"]
        page_height = page_data.get("page_height", 1)
        page_width = page_data.get("page_width", 1)
        lines = page_data.get("lines", [])

        feature_list = []
        prev_line_y_bottom = None
        for i, line_obj in enumerate(lines):
            if not line_obj.get("spans") or not line_obj.get("bbox"):
                feature_list.append(None)
                continue
            next_line_y = lines[i+1]["y"] if i + 1 < len(lines) else None
            features = extract_features(line_obj, page_number, page_height, page_width, next_line_y, prev_line_y_bottom)
            feature_list.append(features)
            prev_line_y_bottom = line_obj["y"] + features["line_height"] if features else prev_line_y_bottom
            
        i = 0
        while i < len(lines):
            if feature_list[i] is None:
                i += 1
                continue

            label, lines_consumed, matched_index = label_line(lines, i, page_number, gt_title, gt_outline, used_gt_indices)
            
            if matched_index is not None:
                used_gt_indices.add(matched_index)
            
            for j in range(lines_consumed):
                line_index = i + j
                if line_index < len(lines) and feature_list[line_index] is not None:
                    feature_list[line_index]["label"] = label
            
            i += lines_consumed
            
        for features in feature_list:
            if features and "label" in features:
                rows.append(features)

    return rows

def main():
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
    df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    if not df.empty:
        df = df[df['text'].str.strip() != '']
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nTraining data successfully saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()