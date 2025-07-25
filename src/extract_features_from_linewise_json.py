import os
import json
import pandas as pd
import re
from utils import fuzzy_match, is_title_case, is_uppercase, ends_with_colon, starts_with_number, word_count, has_bullet_prefix, clean_extracted_text

LINEWISE_DIR = "../data/linewise_json"
GT_DIR = "../data/gt_json"
OUTPUT_CSV = "../data/processed/training_data.csv"

# Keywords for title and H1 detection
TITLE_KEYWORDS = ["table of contents", "contents", "introduction", "chapter", "section", "abstract", "summary", "acknowledgments", "references", "bibliography", "index", "appendix", "preface", "foreword", "dedication"]
H1_KEYWORDS = ["chapter", "section", "part", "introduction", "conclusion", "abstract"]

def is_conventional_heading(text):
    """Checks for common heading patterns using domain knowledge."""
    text = text.strip().lower()
    if re.match(r'^(appendix\s+[a-z\d]+[:.]?|section\s+\d+[:.]?|chapter\s+\d+[:.]?|\d+\.\s+)', text):
        return 1
    return 0

def is_centered(bbox, page_width, tolerance=0.05):
    """Checks if the text is horizontally centered within a given tolerance."""
    line_center = (bbox[0] + bbox[2]) / 2
    page_center = page_width / 2
    return abs(line_center - page_center) / page_width < tolerance

def has_keywords(text, keywords):
    """Checks if the text contains any of the specified keywords."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)

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

        for i, gt_item in enumerate(gt_outline):
            if i in used_gt_indices:
                continue
            if fuzzy_match(combined_text, gt_item["text"]):
                # Ensure it's on the same page, or GT has no page info
                if gt_item.get("page") is None or gt_item["page"] == page_number:
                    return gt_item["level"], 2, i

    # --- Check for Single-line Matches ---
    for i, gt_item in enumerate(gt_outline):
        if i in used_gt_indices:
            continue
        if fuzzy_match(current_line_text, gt_item["text"]):
            if gt_item.get("page") is None or gt_item["page"] == page_number:
                return gt_item["level"], 1, i

    # --- Check for Title Match ---
    if gt_title and fuzzy_match(current_line_text, gt_title):
        return "title", 1, None # Titles don't have an index in outline

    return "other", 1, None

def _extract_features_for_line(line_obj, page_width, page_height, page_number):
    """Extracts basic features for a single line. Some features need a second pass."""
    text = clean_extracted_text(line_obj["text"]) # Use clean_extracted_text from utils
    bbox = line_obj["bbox"]
    spans = line_obj["spans"]

    # Basic features
    font_sizes = [s["size"] for s in spans]
    # Ensure 'flags' is treated as a string before calling .lower()
    # It's better to check for the flag bits directly as in document_processor.py
    # is_bolds = [2 in s.get("flags", 0) for s in spans] # This is not correct for bitwise flags
    # is_italics = [4 in s.get("flags", 0) for s in spans] # This is not correct for bitwise flags

    # Correct way to check flags (assuming flags is an integer bitmask)
    is_bolds = [(s.get("flags", 0) & 2) != 0 for s in spans]
    is_italics = [(s.get("flags", 0) & 4) != 0 for s in spans]


    # Calculate average font size, is_bold, is_italic for the line
    font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
    is_bold = any(is_bolds)
    is_italic = any(is_italics)

    # Positional features (normalized)
    y_pos_normalized = bbox[1] / page_height
    x_pos_normalized = bbox[0] / page_width
    relative_y_pos = y_pos_normalized # This seems to be the same as y_pos_normalized
                                      # If it means something else (e.g., relative to previous heading),
                                      # it needs to be calculated in process_page

    # Line specific features
    line_height = bbox[3] - bbox[1]
    is_first_line = (bbox[1] < page_height * 0.05) # Heuristic for first line on page, needs context

    # Textual features
    wordcount = word_count(text)
    has_bullet = has_bullet_prefix(text)
    is_titlecase = is_title_case(text)
    is_upper = is_uppercase(text)
    ends_colon = ends_with_colon(text)
    starts_number_val = starts_with_number(text)
    is_conventional = is_conventional_heading(text)
    
    is_line_centered = is_centered(bbox, page_width) # Use the helper function
    has_title_kws = has_keywords(text, TITLE_KEYWORDS)
    has_h1_kws = has_keywords(text, H1_KEYWORDS)

    features = {
        "text": text,
        "font_size": font_size,
        "is_bold": is_bold,
        "is_italic": is_italic,
        "y_pos_normalized": y_pos_normalized,
        "x_pos_normalized": x_pos_normalized,
        "line_height": line_height,
        "space_after_line": 0, # Will be filled later
        "space_before_line": 0, # Will be filled later
        "normalized_space_after_line": 0, # Will be filled later
        "normalized_space_before_line": 0, # Will be filled later
        "is_left_aligned": (bbox[0] < page_width * 0.1), # Simple heuristic, might need refinement
        "page": page_number,
        "is_title_case": is_titlecase,
        "is_upper": is_upper,
        "ends_colon": ends_colon,
        "starts_number": starts_number_val,
        "word_count": wordcount,
        "has_bullet_prefix": has_bullet,
        "is_conventional_heading": is_conventional,
        "is_first_line_on_page": is_first_line,
        "is_centered": is_line_centered,
        "relative_y_pos": relative_y_pos, # This will be adjusted in the second pass if needed
        "has_title_keywords": has_title_kws,
        "has_h1_keywords": has_h1_kws,
        
        # Placeholder for previous line features
        "prev_line_font_size": 0,
        "prev_line_is_bold": False,
        "prev_line_is_upper": False,
        "prev_line_ends_colon": False,
        "prev_line_starts_number": False,
        "prev_line_word_count": 0,
        "prev_line_has_bullet_prefix": False,
        "prev_line_is_title_case": False,
        "is_largest_font_on_page": False # Will be determined in the second pass
    }
    return features

def process_page(page_data, gt_title, gt_outline):
    """Processes a single page to extract features and apply labels."""
    lines = page_data["lines"]
    page_width = page_data["page_width"]
    page_height = page_data["page_height"]
    page_number = page_data["page_number"]
    
    features_list = []
    
    # First pass: Extract initial features for all lines
    for i, line_obj in enumerate(lines):
        features_list.append(_extract_features_for_line(line_obj, page_width, page_height, page_number))

    # Calculate page-level features after all lines are processed
    page_font_sizes = [f["font_size"] for f in features_list]
    max_font_size_on_page = max(page_font_sizes) if page_font_sizes else 0

    # Second pass: Calculate line spacing, previous line features, and largest font
    for i in range(len(features_list)):
        # is_largest_font_on_page
        features_list[i]["is_largest_font_on_page"] = (features_list[i]["font_size"] == max_font_size_on_page and max_font_size_on_page > 0)
        
        # Line spacing
        if i > 0:
            prev_line_bbox = lines[i-1]["bbox"]
            current_line_bbox = lines[i]["bbox"]
            space_before = current_line_bbox[1] - prev_line_bbox[3]
            features_list[i]["space_before_line"] = space_before
            features_list[i]["normalized_space_before_line"] = space_before / page_height

            # Previous line features
            features_list[i]["prev_line_font_size"] = features_list[i-1]["font_size"]
            features_list[i]["prev_line_is_bold"] = features_list[i-1]["is_bold"]
            features_list[i]["prev_line_is_upper"] = features_list[i-1]["is_upper"]
            features_list[i]["prev_line_ends_colon"] = features_list[i-1]["ends_colon"]
            features_list[i]["prev_line_starts_number"] = features_list[i-1]["starts_number"]
            features_list[i]["prev_line_word_count"] = features_list[i-1]["word_count"]
            features_list[i]["prev_line_has_bullet_prefix"] = features_list[i-1]["has_bullet_prefix"]
            features_list[i]["prev_line_is_title_case"] = features_list[i-1]["is_title_case"]
        
        if i < len(features_list) - 1:
            current_line_bbox = lines[i]["bbox"]
            next_line_bbox = lines[i+1]["bbox"]
            space_after = next_line_bbox[1] - current_line_bbox[3]
            features_list[i]["space_after_line"] = space_after
            features_list[i]["normalized_space_after_line"] = space_after / page_height

    # Labeling pass
    used_gt_indices = set()
    rows_to_append = [] # Use a separate list for rows to append
    i = 0
    while i < len(features_list):
        label, lines_consumed, matched_index = label_line(lines, i, page_number, gt_title, gt_outline, used_gt_indices)
        
        if matched_index is not None:
            used_gt_indices.add(matched_index)
        
        for j in range(lines_consumed):
            line_index = i + j
            if line_index < len(features_list):
                # Ensure the feature dictionary exists and is not None
                if features_list[line_index] is not None:
                    features_list[line_index]["label"] = label
                    rows_to_append.append(features_list[line_index]) # Add to the final list after labeling
                else:
                    print(f"Warning: features_list[{line_index}] is None. Skipping.") # Debugging
            else:
                print(f"Warning: line_index {line_index} out of bounds for features_list of length {len(features_list)}") # Debugging
        
        i += lines_consumed
        
    return [row for row in rows_to_append if row['text'].strip() != ''] # Filter out empty text lines here too

def process_pair(linewise_path, gt_path):
    """Processes a pair of linewise JSON and ground truth JSON files."""
    with open(linewise_path, 'r', encoding='utf-8') as f:
        linewise_data = json.load(f)
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    gt_title = gt_data.get("title")
    gt_outline = gt_data.get("outline", [])

    rows = []
    for page_data in linewise_data["pages"]:
        rows.extend(process_page(page_data, gt_title, gt_outline))

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
        # The filter for empty text is now done within process_page for robustness.
        # However, a final check here doesn't hurt, but might be redundant.
        # df = df[df['text'].str.strip() != '']
        # Save to CSV
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Features extracted and saved to {OUTPUT_CSV}")
    else:
        print("No data to process or save.")

if __name__ == "__main__":
    main()