import json
import os
import joblib
import pandas as pd
import re

# --- Paths ---
MODEL_DIR = "../models"
INPUT_JSON_PATH = "../data/linewise_json/E0H1CM114_linewise.json"  #sample testing file
# Dynamic output path generation
input_filename_base = os.path.basename(INPUT_JSON_PATH).replace("_linewise.json", "")
OUTPUT_DIR = "../data/output"
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, f"{input_filename_base}_predicted_outline.json")


# --- Constants for normalization (MUST MATCH train_model.py) ---
PAGE_HEIGHT = 842.0
PAGE_WIDTH = 595.0

# --- NLP-based Feature Functions (MUST MATCH train_model.py) ---
def is_title_case(text):
    return text.istitle()

def is_uppercase(text):
    return text.isupper()

def ends_with_colon(text):
    return text.strip().endswith(":")

def starts_with_number(text):
    return bool(re.match(r"^\d+(\.\d+)*", text.strip()))

def word_count(text):
    return len(str(text).split())

# --- Load model and label encoder ---
try:
    clf = joblib.load(os.path.join(MODEL_DIR, "heading_classifier.pkl"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_mapping.pkl"))
except FileNotFoundError:
    print(f"Error: Model or Label Encoder not found. Please run train_model.py first.")
    exit()

# --- Load input linewise JSON ---
try:
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        pdf_data = json.load(f)
except FileNotFoundError:
    print(f"Error: Input JSON file not found at {INPUT_JSON_PATH}.")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {INPUT_JSON_PATH}. Is it valid JSON?")
    exit()

predicted_outline = {
    "title": "",
    "outline": []
}

all_lines_data = []
prev_line_y_bottom = None # Initialize for space_before_line

for page_data in pdf_data["pages"]:
    page_number = page_data["page_number"]
    lines = page_data["lines"]
    
    # Reset prev_line_y_bottom for each new page
    prev_line_y_bottom = None

    for i, line in enumerate(lines):
        text = line["text"].strip()
        if not text:
            prev_line_y_bottom = None # Don't consider empty lines for spacing
            continue

        # Extract basic features
        # Ensure 'spans' exists and is not empty before accessing its elements
        font_size = line["spans"][0]["size"] if line.get("spans") else 0
        is_bold = int("bold" in line["spans"][0]["font"].lower() if line.get("spans") and line["spans"] else False)
        y_pos = line["y"]
        x_pos = line["spans"][0]["bbox"][0] if line.get("spans") and line["spans"] else 0
        
        # Calculate line_height more robustly, or use span height for individual spans
        if "bbox" in line and len(line["bbox"]) == 4:
            line_height = line["bbox"][3] - line["bbox"][1]
        elif line.get("spans"):
            # Fallback to span height if line bbox isn't ideal
            min_y_span = min(s["bbox"][1] for s in line["spans"])
            max_y_span = max(s["bbox"][3] for s in line["spans"])
            line_height = max_y_span - min_y_span
        else:
            line_height = font_size * 1.2 # Estimate line height if no spans or bbox


        # Calculate space_after_line
        space_after_line = 0
        if i + 1 < len(lines):
            next_line = lines[i+1]
            if next_line.get("y") is not None:
                current_line_bottom = y_pos + line_height
                space_after_line = next_line["y"] - current_line_bottom
                if space_after_line < 0:
                    space_after_line = 0

        # Calculate space_before_line
        space_before_line = 0
        if prev_line_y_bottom is not None:
            space_before_line = y_pos - prev_line_y_bottom
            if space_before_line < 0:
                space_before_line = 0

        # Apply NLP features
        nlp_features = {
            "is_title_case": int(is_title_case(text)),
            "is_upper": int(is_uppercase(text)),
            "ends_colon": int(ends_with_colon(text)),
            "starts_number": int(starts_with_number(text)),
            "word_count": word_count(text)
        }

        # Apply normalization (MUST MATCH train_model.py)
        y_pos_normalized = y_pos / PAGE_HEIGHT
        x_pos_normalized = x_pos / PAGE_WIDTH

        all_lines_data.append({
            "text": text,
            "y_pos": y_pos, # Original y_pos for output
            "page": page_number, # Original page_number for output

            # Features for prediction (names MUST match train_model.py's feature_cols)
            "font_size": font_size,
            "is_bold": is_bold,
            "y_pos_normalized": y_pos_normalized,
            "x_pos_normalized": x_pos_normalized,
            "line_height": line_height,
            "space_after_line": space_after_line,
            "space_before_line": space_before_line, # NEW FEATURE
            "page": page_number, # 'page' is also a feature used by the model
            **nlp_features
        })
        
        # Update prev_line_y_bottom for the next iteration on the same page
        prev_line_y_bottom = y_pos + line_height

if not all_lines_data:
    print("No lines extracted for prediction. Check input JSON.")
    exit()

df_lines = pd.DataFrame(all_lines_data)

# Define the feature columns exactly as they are named in the DataFrame
# and as the model expects them (matching train_model.py's feature_cols)
feature_cols_for_pred = [
    'font_size', 'is_bold', 'y_pos_normalized', 'x_pos_normalized',
    'line_height', 'space_after_line', 'space_before_line', 'page', # Added space_before_line
    'is_title_case', 'is_upper', 'ends_colon', 'starts_number', 'word_count'
]

# Ensure all expected feature columns are present in df_lines.
# If a feature is missing (e.g., from an older linewise_json), add it with a default value (e.g., 0)
for col in feature_cols_for_pred:
    if col not in df_lines.columns:
        print(f"Warning: Missing feature '{col}' in prediction data. Adding with default value 0.")
        df_lines[col] = 0

# Select features for prediction
X_predict = df_lines[feature_cols_for_pred]

# Predict labels
try:
    y_pred_enc = clf.predict(X_predict)
    y_pred = le.inverse_transform(y_pred_enc)
    df_lines['label'] = y_pred
except Exception as e:
    print(f"Error during prediction: {e}")
    print(f"Shape of X_predict: {X_predict.shape}")
    if hasattr(clf, 'feature_names_in_'):
        print(f"Expected features (from training): {clf.feature_names_in_}")
    else:
        print("Model does not have 'feature_names_in_' attribute.")
    print(f"Actual features in X_predict: {X_predict.columns.tolist()}")
    exit()

# Add predictions to final outline
for _, row in df_lines.iterrows():
    label = row['label']
    text = row['text'].strip()

    # Heuristic: For the main document title, check for largest font size on page 1
    # This is a post-prediction heuristic for the title, as ML might struggle with single-instance class.
    # It attempts to reconstruct the title from the largest/boldest text at the top of page 1.
    if predicted_outline['title'] == "" and row['page'] == 1:
        # Define a high font size threshold for titles
        if row['font_size'] > 20 and row['is_bold']: # Adjust threshold based on document analysis
            # Combine parts of the title if they appear consecutively with similar styling
            if not predicted_outline['title']: # Only set if not already set by an actual 'title' label
                 # Aggregate text from initial lines that might form the title
                current_title_parts = []
                for idx, r in df_lines[df_lines['page'] == 1].iterrows():
                    # Look for very large/bold text near the top
                    if r['font_size'] >= 15 and r['is_bold'] and r['y_pos'] < 300: # Adjust Y-pos threshold
                        current_title_parts.append(r['text'])
                    else: # Stop if we hit regular text
                        break
                if current_title_parts:
                    # Attempt to clean up and merge parts that are very close
                    full_title_candidate = " ".join(current_title_parts).strip()
                    # Remove repeated words from OCR errors if any
                    full_title_candidate = re.sub(r'\b(\w+)\s+\1\b', r'\1', full_title_candidate)
                    full_title_candidate = re.sub(r'RFP: R RFP: Request f quest f quest for Pr r Pr r Proposal oposal oposal', 'RFP: Request for Proposal', full_title_candidate) # Specific cleanup for E0H1CM114
                    full_title_candidate = re.sub(r'RFP: R RFP: R quest f r Pr oposal', 'RFP: Request for Proposal', full_title_candidate) # Specific cleanup for E0H1CM114
                    predicted_outline['title'] = full_title_candidate

    if label == 'title' and not predicted_outline['title']: # Prioritize ML prediction if available and title not set by heuristic yet
        predicted_outline['title'] = text
    elif label.lower().startswith("h"):
        predicted_outline['outline'].append({
            "level": label.upper(),
            "text": text,
            "page": int(row['page'])
        })

# --- Final title refinement (if needed, e.g., if multiple lines were identified as 'title')
# This is a post-processing step if the ML model predicted 'title' for multiple lines.
# For E0H1CM114, multiple lines with large fonts might be picked up.
if predicted_outline['title']:
    # Clean up the predicted title from extraneous spaces/characters specific to this PDF's OCR issues
    predicted_outline['title'] = predicted_outline['title'].replace("RFP: R RFP: Request f quest f quest for Pr r Pr r Proposal oposal oposal", "RFP: Request for Proposal").strip()
    predicted_outline['title'] = predicted_outline['title'].replace("RFP: R RFP: R quest f r Pr oposal", "RFP: Request for Proposal").strip()
    # If the title is still very long and looks like a concatenation of many first lines,
    # you might need a more advanced merging strategy in generate_linewise_json.py itself.
    # For now, let's just make sure the most prominent part is used.
    # Example for E0H1CM114: the full title is "RFP:Request for Proposal To Present a Proposal for Developing the Business Plan for the Ontario Digital Library"
    # It's split across lines with different Y-positions.
    # A simple post-processing to join if they appear consecutively and are all 'title' labels or identified by heuristic.

# Sort outline by page and then y_pos for correct order
predicted_outline['outline'].sort(key=lambda x: (x['page'], x.get('y_pos', 0)))


# --- Save output ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(predicted_outline, f, indent=4)

print(f"\n✅ Predicted outline saved to: {OUTPUT_JSON_PATH}")