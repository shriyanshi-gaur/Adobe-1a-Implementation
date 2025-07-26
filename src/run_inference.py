import os
import json
import fitz
import pandas as pd
import joblib
import argparse
import time
import re
from collections import defaultdict
from utils import (
    is_title_case, is_uppercase, ends_with_colon, starts_with_number,
    word_count, has_bullet_prefix, clean_extracted_text
)

# --- Load Model and Encoder ---
MODEL_DIR = "models" # Ensure this path is consistent with train_model.py
CONFIDENCE_THRESHOLD = 0.70 # You can adjust this threshold

try:
    clf = joblib.load(os.path.join(MODEL_DIR, "heading_classifier.pkl"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_mapping.pkl"))
    print(f"Successfully loaded model and label encoder from {MODEL_DIR}.")
except FileNotFoundError:
    print(f"Error: Model or Label Encoder not found in {MODEL_DIR}. Please run train_model.py first.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading models: {e}")
    exit()

def is_conventional_heading(text):
    """Checks for common heading patterns using domain knowledge."""
    text = text.strip().lower()
    # Pattern for "Appendix A:", "Section 1.", "Chapter 2", etc.
    if re.match(r'^(appendix\s+[a-z\d]+[:.]?|section\s+\d+[:.]?|chapter\s+\d+[:.]?|\d+\.\s+)', text):
        return 1
    return 0

def parse_pdf_to_lines(pdf_path):
    """
    Parses a PDF and extracts line-wise text and features.
    This logic should be consistent with extract_features_from_linewise_json.py.
    """
    doc = fitz.open(pdf_path)
    all_lines_data = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict", sort=True).get("blocks", [])
        lines_raw = defaultdict(list)

        for block in blocks:
            if block["type"] != 0: continue
            for line_block in block.get("lines", []):
                for span in line_block.get("spans", []):
                    # Filter out very small fonts if necessary, consistent with generate_linewise_json.py
                    # MIN_FONT_SIZE is 6 in generate_linewise_json.py
                    if span['size'] < 6: continue
                    y_top_rounded = round(span["bbox"][1], 1)
                    lines_raw[y_top_rounded].append(span)
        
        # Collect initial line data for the page
        current_page_lines_data = []
        for y_coord in sorted(lines_raw):
            sorted_spans = sorted(lines_raw[y_coord], key=lambda s: s["bbox"][0])
            full_line_text = " ".join(s["text"] for s in sorted_spans)
            cleaned_text = clean_extracted_text(full_line_text)

            if not cleaned_text.strip(): # Skip empty lines after cleaning
                continue

            # Calculate bbox for the entire line
            line_bbox = (
                min(s["bbox"][0] for s in sorted_spans),
                min(s["bbox"][1] for s in sorted_spans),
                max(s["bbox"][2] for s in sorted_spans),
                max(s["bbox"][3] for s in sorted_spans)
            )
            
            # Extract basic features for current line
            font_sizes = [s["size"] for s in sorted_spans]
            is_bolds = [(s.get("flags", 0) & 2) != 0 for s in sorted_spans]
            is_italics = [(s.get("flags", 0) & 4) != 0 for s in sorted_spans]

            font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
            is_bold = any(is_bolds)
            is_italic = any(is_italics)

            y_pos_normalized = line_bbox[1] / page.rect.height
            x_pos_normalized = line_bbox[0] / page.rect.width
            line_height = line_bbox[3] - line_bbox[1]
            is_left_aligned = (line_bbox[0] < page.rect.width * 0.1) # Calculation for is_left_aligned
            
            # Additional features for prediction
            wordcount = word_count(cleaned_text)
            has_bullet = has_bullet_prefix(cleaned_text)
            is_titlecase = is_title_case(cleaned_text)
            is_upper = is_uppercase(cleaned_text)
            ends_colon = ends_with_colon(cleaned_text)
            starts_number_val = starts_with_number(cleaned_text)
            is_conventional = is_conventional_heading(cleaned_text)

            current_page_lines_data.append({
                "text": cleaned_text,
                "font_size": font_size,
                "is_bold": is_bold,
                "is_italic": is_italic,
                "y_pos_normalized": y_pos_normalized,
                "x_pos_normalized": x_pos_normalized,
                "line_height": line_height,
                "bbox": line_bbox, # Keep bbox for spacing calculations
                "page": page_num + 1, # Page numbers are 1-indexed
                "is_title_case": is_titlecase,
                "is_upper": is_upper,
                "ends_colon": ends_colon,
                "starts_number": starts_number_val,
                "word_count": wordcount,
                "has_bullet_prefix": has_bullet,
                "is_conventional_heading": is_conventional,
                "is_left_aligned": is_left_aligned, # Ensure this is explicitly included
                
                # Placeholders for page-level & previous line features
                "is_largest_font_on_page": False,
                "is_first_line_on_page": False, # Will be set below
                "is_centered": False, # Needs page width context
                "relative_y_pos": y_pos_normalized, # Placeholder, can be refined
                "has_title_keywords": False, # Needs global keywords
                "has_h1_keywords": False, # Needs global keywords
                "prev_line_font_size": 0,
                "prev_line_is_bold": False,
                "prev_line_is_upper": False,
                "prev_line_ends_colon": False,
                "prev_line_starts_number": False,
                "prev_line_word_count": 0,
                "prev_line_has_bullet_prefix": False,
                "prev_line_is_title_case": False,
                "space_after_line": 0,
                "space_before_line": 0,
                "normalized_space_after_line": 0,
                "normalized_space_before_line": 0,
            })
        
        if not current_page_lines_data:
            continue

        # Second pass for page-level and previous-line features
        page_font_sizes = [d["font_size"] for d in current_page_lines_data]
        max_font_size_on_page = max(page_font_sizes) if page_font_sizes else 0
        
        page_width = page.rect.width
        page_height = page.rect.height

        for i in range(len(current_page_lines_data)):
            line = current_page_lines_data[i]
            
            # is_largest_font_on_page
            line["is_largest_font_on_page"] = (line["font_size"] == max_font_size_on_page and max_font_size_on_page > 0)
            
            # is_first_line_on_page
            line["is_first_line_on_page"] = (i == 0) # First line in the *processed* list for the page

            # is_centered (re-calculate using actual page_width)
            line_center = (line["bbox"][0] + line["bbox"][2]) / 2
            page_center = page_width / 2
            line["is_centered"] = abs(line_center - page_center) / page_width < 0.05 # Using a tolerance of 5%

            # Line spacing and previous line features
            if i > 0:
                prev_line = current_page_lines_data[i-1]
                
                space_before = line["bbox"][1] - prev_line["bbox"][3]
                line["space_before_line"] = space_before
                line["normalized_space_before_line"] = space_before / page_height

                line["prev_line_font_size"] = prev_line["font_size"]
                line["prev_line_is_bold"] = prev_line["is_bold"]
                line["prev_line_is_upper"] = prev_line["is_upper"]
                line["prev_line_ends_colon"] = prev_line["ends_colon"]
                line["prev_line_starts_number"] = prev_line["starts_number"]
                line["prev_line_word_count"] = prev_line["word_count"]
                line["prev_line_has_bullet_prefix"] = prev_line["has_bullet_prefix"]
                line["prev_line_is_title_case"] = prev_line["is_title_case"]
            
            if i < len(current_page_lines_data) - 1:
                next_line = current_page_lines_data[i+1]
                space_after = next_line["bbox"][1] - line["bbox"][3]
                line["space_after_line"] = space_after
                line["normalized_space_after_line"] = space_after / page_height
            
            all_lines_data.append(line)

    doc.close()
    return pd.DataFrame(all_lines_data)

def get_document_title(df_lines, confidence_threshold, preds_proba, label_encoder):
    """
    Attempts to identify the document title.
    Prioritizes 'title' label, then falls back to first H1 on page 1,
    then uses a heuristic for prominent text on page 1, including multi-line titles
    and special handling for "sparse" title-only pages.
    """
    # Try to find a line explicitly labeled as 'title' with high confidence
    titles = df_lines[
        (df_lines['label'].str.lower() == 'title') & 
        (df_lines['confidence'] >= confidence_threshold)
    ].sort_values(by='y_pos_normalized') # Sort to pick the top-most
    
    if not titles.empty:
        return titles.iloc[0]['text']
    
    # Fallback 1: Look for the first H1 on page 1 with high confidence
    h1_on_page_1 = df_lines[
        (df_lines['page'] == 1) & 
        (df_lines['label'].str.lower() == 'h1') &
        (df_lines['confidence'] >= confidence_threshold)
    ].sort_values(by='y_pos_normalized')

    if not h1_on_page_1.empty:
        return h1_on_page_1.iloc[0]['text']
        
    # Fallback 2: Heuristic for potential titles on page 1 if previous attempts failed
    page_1_lines = df_lines[df_lines['page'] == 1].copy()
    if not page_1_lines.empty:
        page_1_lines_with_original_idx = page_1_lines.reset_index() 
        
        label_to_idx = {name: idx for idx, name in enumerate(label_encoder.classes_)}
        heading_indices = [
            label_to_idx[lbl] for lbl in ['title', 'h1', 'h2', 'h3', 'h4']
            if lbl in label_to_idx # Ensure the label actually exists in the trained classes
        ]

        title_candidates_info = []
        
        # Calculate max and average font size on page 1 for comparison
        max_font_size_on_page_1 = page_1_lines_with_original_idx['font_size'].max()
        avg_font_size_on_page_1 = page_1_lines_with_original_idx['font_size'].mean()

        # Determine if it's a "sparse" first page (e.g., likely a title-only page)
        is_sparse_first_page = len(page_1_lines_with_original_idx) <= 5 # Few lines on the page

        # Iterate through page 1 lines, sorted by y_pos_normalized to maintain reading order
        for idx_in_page_lines, row in page_1_lines_with_original_idx.sort_values(by='y_pos_normalized', ascending=True).iterrows():
            line_proba = preds_proba[row['original_df_index']]
            max_heading_proba = 0
            if heading_indices:
                max_heading_proba = max(line_proba[idx] for idx in heading_indices)
            
            is_top_of_page = row['y_pos_normalized'] < 0.35 
            is_very_top = row['y_pos_normalized'] < 0.15 # Even stricter top area (e.g., for "Learn Acrobat")
            is_not_too_long = row['word_count'] < 20 # Avoid full paragraphs

            # A line is a strong candidate if:
            is_candidate = False
            if is_sparse_first_page:
                # For sparse pages, be more aggressive: first 3 lines at the very top, and prominent
                if idx_in_page_lines < 3 and is_very_top:
                    # If it's among the largest, or significantly larger than average
                    if (max_font_size_on_page_1 > 0 and row['font_size'] >= max_font_size_on_page_1 * 0.9) or \
                       (avg_font_size_on_page_1 > 0 and row['font_size'] > (avg_font_size_on_page_1 * 1.5)):
                        is_candidate = True
            elif is_top_of_page and is_not_too_long:
                # Original heuristic for non-sparse pages
                if (row['font_size'] == max_font_size_on_page_1) and (max_font_size_on_page_1 > 0):
                    is_candidate = True 
                elif idx_in_page_lines < 3 and (avg_font_size_on_page_1 > 0 and row['font_size'] > (avg_font_size_on_page_1 * 1.5)):
                    is_candidate = True
                elif (max_heading_proba >= (confidence_threshold * 0.6) and 
                      (row['is_bold'] or row['font_size'] > 16 or row['is_upper'])):
                    is_candidate = True

            if is_candidate:
                title_candidates_info.append(row)
        
        # Merge consecutive title candidates if they are close
        if title_candidates_info:
            merged_title_parts = [title_candidates_info[0]['text']]
            for i in range(1, len(title_candidates_info)):
                prev_candidate = title_candidates_info[i-1]
                current_candidate = title_candidates_info[i]

                vertical_gap = current_candidate['bbox'][1] - prev_candidate['bbox'][3]
                x_alignment_tolerance = 0.05 
                is_x_aligned = abs(current_candidate['x_pos_normalized'] - prev_candidate['x_pos_normalized']) < x_alignment_tolerance
                
                is_mergable = False
                # If it's a sparse page, use more generous merging for top lines
                if is_sparse_first_page:
                    # Merge if close AND aligned AND are both among the largest fonts on the page
                    if is_x_aligned and vertical_gap < (prev_candidate['line_height'] * 2.0) and \
                       (max_font_size_on_page_1 > 0 and current_candidate['font_size'] >= max_font_size_on_page_1 * 0.8) and \
                       (max_font_size_on_page_1 > 0 and prev_candidate['font_size'] >= max_font_size_on_page_1 * 0.8):
                        is_mergable = True
                else:
                    # Original merging conditions for non-sparse pages
                    if (vertical_gap < (prev_candidate['line_height'] * 1.5) and is_x_aligned):
                        is_mergable = True
                    elif current_candidate['y_pos_normalized'] < 0.2 and \
                         (current_candidate['font_size'] == max_font_size_on_page_1) and is_x_aligned:
                        is_mergable = True
                    elif current_candidate['y_pos_normalized'] < 0.2 and \
                         (avg_font_size_on_page_1 > 0 and current_candidate['font_size'] > (avg_font_size_on_page_1 * 1.5)) and is_x_aligned:
                        is_mergable = True

                if is_mergable:
                    merged_title_parts.append(current_candidate['text'])
                else:
                    break # Cannot merge further
            
            if merged_title_parts:
                return " ".join(merged_title_parts)
            
    return "Untitled Document"


def process_pdf(pdf_path, output_dir, filename):
    print(f"  - Processing {filename}...")
    start_time = time.time()
    
    # Store the original index from the DataFrame before any filtering
    df_lines = parse_pdf_to_lines(pdf_path).reset_index().rename(columns={'index': 'original_df_index'})

    if df_lines.empty:
        print(f"  ❌ No lines extracted or features generated for {filename}.")
        output_data = {"title": "Untitled Document", "outline": []}
        output_filename = os.path.splitext(filename)[0] + ".json"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"  ✅ Empty output saved for {filename} to {output_path}")
        return

    # Define feature columns - MUST match what the model was trained on
    feature_cols = [
        'font_size', 'is_bold', 'is_italic', 'y_pos_normalized', 'x_pos_normalized',
        'line_height', 'space_after_line', 'space_before_line',
        'normalized_space_after_line', 'normalized_space_before_line',
        'is_left_aligned', 'page', 'is_title_case', 'is_upper',
        'ends_colon', 'starts_number', 'word_count', 'has_bullet_prefix',
        'is_conventional_heading'
    ]

    # Verify that all required feature columns exist in the DataFrame
    missing_features = [col for col in feature_cols if col not in df_lines.columns]
    if missing_features:
        print(f"  ❌ Error: Missing features in parsed PDF data for {filename}: {missing_features}")
        return # Skip processing this file

    # Make predictions
    X_inference = df_lines[feature_cols]
    preds_encoded = clf.predict(X_inference)
    preds_proba = clf.predict_proba(X_inference) # Get probabilities for confidence
    
    # Get the confidence for the predicted class
    confidences = [preds_proba[i, pred_encoded] for i, pred_encoded in enumerate(preds_encoded)]

    df_lines['label'] = le.inverse_transform(preds_encoded)
    df_lines['confidence'] = confidences

    # Pass preds_proba and label_encoder to get_document_title for more robust title detection
    title = get_document_title(df_lines, CONFIDENCE_THRESHOLD, preds_proba, le)

    # Filter for headings (h1, h2, h3, h4 etc.) and apply confidence threshold
    # IMPORTANT: 'title' labels are explicitly EXCLUDED from the outline here, as they form the main document title.
    relevant_lines_for_outline = df_lines[
        (df_lines['label'].str.lower().str.startswith('h')) # Only include 'h' labels (h1, h2, h3, etc.)
    ].copy()

    # Apply confidence threshold
    confident_relevant_lines = relevant_lines_for_outline[relevant_lines_for_outline['confidence'] >= CONFIDENCE_THRESHOLD]
    
    # Sort for sequential merging
    confident_relevant_lines = confident_relevant_lines.sort_values(by=['page', 'y_pos_normalized']).reset_index(drop=True)

    merged_outline_items = []
    i = 0
    while i < len(confident_relevant_lines):
        current_line = confident_relevant_lines.iloc[i]
        
        merged_text = current_line['text']
        merged_level = current_line['label'].upper()
        merged_page = int(current_line['page'])
        first_line_y_pos = current_line['y_pos_normalized'] # Keep track of the first line's y-pos for sorting

        j = i + 1
        while j < len(confident_relevant_lines):
            next_line = confident_relevant_lines.iloc[j]

            is_same_page = (next_line['page'] == current_line['page'])
            is_same_level = (next_line['label'].lower() == current_line['label'].lower())
            
            vertical_gap = next_line["bbox"][1] - current_line["bbox"][3] # next_line_top - current_line_bottom
            
            # Dynamic threshold for vertical gap based on font size or line height
            max_vertical_gap = 1.5 * current_line['font_size'] # Can be adjusted
            
            # Check x-alignment (e.g., start positions are close)
            x_alignment_threshold = 0.02 # 2% of page width difference in normalized x-position. Can be adjusted.
            is_x_aligned = abs(next_line['x_pos_normalized'] - current_line['x_pos_normalized']) < x_alignment_threshold

            # Merge if conditions met
            if is_same_page and is_same_level and vertical_gap < max_vertical_gap and is_x_aligned:
                merged_text += " " + next_line['text'] # Combine text with a space
                j += 1
            else:
                break # Cannot merge further

        merged_outline_items.append({
            "level": merged_level,
            "text": merged_text,
            "page": merged_page,
            "y_pos_normalized": first_line_y_pos # Temporary for sorting
        })
        i = j # Move to the next unmerged line
    
    # Final sort based on the y_pos_normalized of the first line of the merged item
    outline = sorted(merged_outline_items, key=lambda x: (x['page'], x['y_pos_normalized']))

    # Remove the temporary 'y_pos_normalized' from the final output for consistency
    for item in outline:
        if 'y_pos_normalized' in item:
            del item['y_pos_normalized']

    output_data = {"title": title, "outline": outline}
    output_filename = os.path.splitext(filename)[0] + ".json"
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    end_time = time.time()
    print(f"  ✅ Finished in {end_time - start_time:.2f}s. Output saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract a structured outline from PDF files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input PDF files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output JSON files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {args.input_dir}. Exiting.")
        return

    print(f"\nStarting inference for {len(pdf_files)} PDF(s) in {args.input_dir}...")
    for filename in pdf_files:
        pdf_path = os.path.join(args.input_dir, filename)
        process_pdf(pdf_path, args.output_dir, filename)
    print("\nInference complete for all PDFs.")

if __name__ == "__main__":
    main()