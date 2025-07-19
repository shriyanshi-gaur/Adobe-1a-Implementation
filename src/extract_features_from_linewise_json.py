import os
import json
import pandas as pd
import difflib
import re # Make sure re is imported

# Paths to folders
LINEWISE_DIR = "../data/linewise_json"
GT_DIR = "../data/gt_json"
OUTPUT_CSV = "../data/processed/training_data.csv"

def normalize(text):
    # Enhanced normalization to remove common heading punctuation and multiple spaces, then simplify
    text = str(text).strip() # Ensure text is string
    text = re.sub(r'[:;,.]\s*$', '', text) # Remove trailing punctuation
    text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with a single space
    return "".join(text.lower().split()) # Remove all spaces for fuzzy comparison

def fuzzy_match(text1, text2, threshold=0.75): # Adjusted threshold back to a more common value
    # Ensure both texts are normalized before comparison
    return difflib.SequenceMatcher(None, normalize(text1), normalize(text2)).ratio() >= threshold

# --- NLP-based Feature Functions (Copied from train_model.py, necessary for label_line) ---
def is_title_case(text):
    return str(text).istitle()

def is_uppercase(text):
    return str(text).isupper()

def ends_with_colon(text):
    return str(text).strip().endswith(":")

def starts_with_number(text): # <--- THIS FUNCTION IS ADDED
    return bool(re.match(r"^\d+(\.\d+)*", str(text).strip()))

def word_count(text):
    return len(str(text).split()) # Handle potential non-string text
# --- End of copied functions ---


def label_line(line_obj, page_number, gt_data, all_lines_on_page):
    """
    Labels a single line object based on ground truth, considering multi-line titles/headings.
    
    Args:
        line_obj (dict): The current line object from linewise_json.
        page_number (int): The current page number.
        gt_data (dict): The ground truth JSON data for the document.
        all_lines_on_page (list): All line objects for the current page, in order.
    Returns:
        str: The predicted label ('title', 'h1', 'h2', etc., or 'other').
    """
    current_line_text = line_obj["text"].strip()
    
    # --- Attempt to match Title ---
    if page_number == 1 and gt_data.get("title"):
        gt_title_norm = normalize(gt_data["title"])
        
        # Strategy 1: Direct fuzzy match on the line text (might work if title is single line or OCR is clean)
        if fuzzy_match(gt_title_norm, current_line_text):
            return "title"
        
        # Strategy 2: Heuristic based on font size and position for page 1 (for fragmented titles)
        # Check if this line is part of the original title text and is prominently formatted.
        # This is a heuristic to help label the training data for the 'title' class.
        if line_obj.get("spans") and line_obj["spans"]:
            current_font_size = line_obj["spans"][0]["size"]
            is_bold_line = "bold" in line_obj["spans"][0]["font"].lower()
            
            # If the line looks like a title component based on font/boldness and position
            if current_font_size > 15 and is_bold_line and line_obj["y"] < 350: # Adjust thresholds as needed
                # Check if its normalized text is a substring of the GT title (allowing for partial matches)
                if normalize(current_line_text) in gt_title_norm:
                    return "title"
                # For specific OCR issues in E0H1CM114, add direct text checks
                # These are very specific to the provided PDF's OCR artifacts.
                if "RFP: Request for Proposal" in current_line_text or \
                   "To Present a Proposal for Developing" in current_line_text or \
                   "the Business Plan for the Ontario" in current_line_text or \
                   "Digital Library" in current_line_text:
                    return "title"


    # --- Attempt to match Outline Headings ---
    for item in gt_data.get("outline", []):
        if item.get("page") != page_number:
            continue
        
        gt_item_text = item.get("text", "")
        gt_item_level = item.get("level", "other").lower()

        # Strategy 1: Direct fuzzy match
        if fuzzy_match(gt_item_text, current_line_text):
            return gt_item_level

        # Strategy 2: Handle fragmented headings - specifically for numbered headings
        # Look for lines that start with a number and approximately match a ground truth heading.
        if starts_with_number(current_line_text) and \
           normalize(gt_item_text).startswith(normalize(current_line_text).split(' ')[0]) and \
           fuzzy_match(gt_item_text, current_line_text, threshold=0.6): # Lower threshold for numbered headings
            return gt_item_level

    return "other" # Default if no match is found

def extract_features(line, page_number, page_height=None, next_line_y=None, prev_line_y_bottom=None):
    spans = line.get("spans", [])
    if not spans:
        # If no spans, it's not a text line we care about, return None so it's skipped
        return None

    # Use first span for basic features, but be careful with fragmented lines
    first_span = spans[0] 
    
    font_sizes = [s["size"] for s in spans]
    fonts = [s["font"].lower() for s in spans]
    
    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
    is_bold = int(any("bold" in f for f in fonts)) # Use int(bool) directly for consistency
    
    y_pos = line["y"] # Top y-coordinate of the line
    
    # x_pos of the first character of the line
    x_pos = first_span["bbox"][0] 

    # Calculate line_height: Max y_bottom - Min y_top across all spans in the line
    min_y_span = min(s["bbox"][1] for s in spans)
    max_y_span = max(s["bbox"][3] for s in spans)
    line_height = max_y_span - min_y_span
    if line_height < 0: line_height = 0 # Ensure non-negative

    # Calculate space_after_line
    space_after_line = 0
    line_bottom = y_pos + line_height
    if next_line_y is not None:
        space_after_line = next_line_y - line_bottom
        if space_after_line < 0: space_after_line = 0 # Cap at 0 if lines overlap

    # Calculate space_before_line
    space_before_line = 0
    if prev_line_y_bottom is not None:
        space_before_line = y_pos - prev_line_y_bottom
        if space_before_line < 0: space_before_line = 0 # Cap at 0 if lines overlap

    return {
        "text": line["text"],
        "font_size": avg_font_size,
        "is_bold": is_bold,
        "y_pos": y_pos,
        "x_pos": x_pos,
        "line_height": line_height,
        "space_after_line": space_after_line,
        "space_before_line": space_before_line,
        "page": page_number
    }

def process_pair(linewise_path, gt_path):
    with open(linewise_path, "r", encoding="utf-8") as f:
        linewise_data = json.load(f)
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    rows = []
    
    for page_data in linewise_data.get("pages", []):
        page_number = page_data.get("page_number", 1)
        lines = page_data.get("lines", [])
        
        processed_lines_with_features = []
        prev_line_y_bottom = None # Reset for each new page

        # First pass: Extract features for all lines
        for i, line_obj in enumerate(lines):
            # Calculate next_line_y for space_after_line feature
            next_line_y = lines[i+1]["y"] if i + 1 < len(lines) else None
            
            features = extract_features(line_obj, page_number, None, next_line_y, prev_line_y_bottom)
            
            if features:
                processed_lines_with_features.append((line_obj, features)) 
                # Update prev_line_y_bottom for the next iteration on the same page
                prev_line_y_bottom = line_obj["y"] + features["line_height"]
            else:
                # If a line yields no features (e.g., empty spans or text), ensure prev_line_y_bottom is not used for it
                # For a seamless flow, you might want to keep the last valid bottom,
                # or treat empty lines as having zero height and use their y_pos for next line calc.
                # For now, let's keep it None if the current line itself wasn't processed.
                prev_line_y_bottom = None


        # Second pass: Label the lines using the `label_line` function, now that features are ready
        for original_line_obj, features_dict in processed_lines_with_features:
            # Pass original line_obj and full page context to label_line
            label = label_line(original_line_obj, page_number, gt_data, lines) 
            features_dict["label"] = label # Add the label to the feature dictionary
            rows.append(features_dict)
            
    return rows


def main():
    all_rows = []
    
    # Iterate over all linewise JSON files in the LINEWISE_DIR
    for file in os.listdir(LINEWISE_DIR):
        if file.endswith("_linewise.json"):
            linewise_path = os.path.join(LINEWISE_DIR, file)
            # Derive the ground truth JSON filename from the linewise JSON filename
            base_filename = file.replace("_linewise.json", "")
            gt_path = os.path.join(GT_DIR, f"{base_filename}.json")
            
            # Check if the corresponding ground truth file exists
            if os.path.exists(gt_path):
                print(f"Processing pair: {file} and {os.path.basename(gt_path)}...")
                rows = process_pair(linewise_path, gt_path)
                all_rows.extend(rows)
            else:
                print(f"Warning: GT file {os.path.basename(gt_path)} not found for {file}. Skipping this pair.")

    if not all_rows:
        print("No data processed. Check your LINEWISE_DIR and GT_DIR paths and contents, and ensure GT files exist for linewise files.")
        return

    df = pd.DataFrame(all_rows)
    
    # --- Data Cleaning ---
    initial_rows = len(df)
    # Remove rows where 'text' is empty or contains only whitespace after stripping
    df = df[df['text'].str.strip() != '']
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"Removed {removed_rows} rows with empty or whitespace-only text.")

    # Ensure all numerical features are indeed numerical and handle potential NaNs that might arise from edge cases
    # For now, we trust the extract_features function to provide valid numbers or None.
    # If NaNs are still appearing after this, consider more aggressive fillna strategies.
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Training data saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    # It is assumed that generate_linewise_json.py has already been run
    # to populate '../data/linewise_json' directory before running this script.
    
    main()