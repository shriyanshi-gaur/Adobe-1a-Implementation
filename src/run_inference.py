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
MODEL_DIR = "../models"
CONFIDENCE_THRESHOLD = 0.70
try:
    clf = joblib.load(os.path.join(MODEL_DIR, "heading_classifier.pkl"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_mapping.pkl"))
except FileNotFoundError:
    print(f"Error: Model or Label Encoder not found in {MODEL_DIR}. Please run train_model.py first.")
    exit()

def is_conventional_heading(text):
    """Checks for common heading patterns using domain knowledge."""
    text = text.strip().lower()
    # Pattern for "Appendix A:", "Section 1.", "Chapter 2", etc.
    if re.match(r'^(appendix\s+[a-z\d]+[:.]?|section\s+\d+[:.]?|chapter\s+\d+[:.]?|\d+\.\s+)', text):
        return 1
    return 0

def parse_pdf_to_lines(pdf_path):
    """UPDATED: This parsing logic is now consistent with the training data generation."""
    doc = fitz.open(pdf_path)
    all_pages_data = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict", sort=True).get("blocks", [])
        lines_raw = defaultdict(list)
        for block in blocks:
            if block["type"] != 0: continue # Only process text blocks
            for line_block in block.get("lines", []):
                for span in line_block.get("spans", []):
                    # You may adjust MIN_FONT_SIZE if needed, but it's removed here for consistency
                    # if span['size'] < MIN_FONT_SIZE: continue
                    y_top_rounded = round(span["bbox"][1], 1)
                    lines_raw[y_top_rounded].append(span)

        line_items = []
        for y_coord in sorted(lines_raw):
            sorted_spans = sorted(lines_raw[y_coord], key=lambda s: s["bbox"][0])
            full_line_text = " ".join(s["text"] for s in sorted_spans)
            cleaned_text = clean_extracted_text(full_line_text)
            if not cleaned_text: continue
            bbox = (min(s["bbox"][0] for s in sorted_spans), min(s["bbox"][1] for s in sorted_spans),
                    max(s["bbox"][2] for s in sorted_spans), max(s["bbox"][3] for s in sorted_spans))
            line_items.append({"y": y_coord, "text": cleaned_text, "bbox": bbox, "spans": sorted_spans})
        
        all_pages_data.append({
            "page_number": page_num + 1, "lines": line_items,
            "page_height": page.rect.height, "page_width": page.rect.width
        })
    doc.close()
    return all_pages_data

def create_feature_dataframe(pages_data):
    """Creates a DataFrame with features for every line in the document."""
    all_lines_features = []
    for page_data in pages_data:
        lines, page_number, page_height, page_width = page_data["lines"], page_data["page_number"], page_data["page_height"], page_data["page_width"]
        prev_line_y_bottom = None
        for i, line_obj in enumerate(lines):
            spans = line_obj.get("spans", [])
            if not spans: continue
            text_content = line_obj["text"]
            avg_font_size = sum(s["size"] for s in spans) / len(spans) if spans else 0
            is_bold = int(any("bold" in s["font"].lower() for s in spans))
            is_italic = int(any("italic" in s["font"].lower() for s in spans))
            y_pos, x_pos, line_height = line_obj["y"], line_obj["bbox"][0], line_obj["bbox"][3] - line_obj["bbox"][1]
            next_line_y = lines[i + 1]["y"] if i + 1 < len(lines) else None
            space_after_line = (next_line_y - (y_pos + line_height)) if next_line_y else 0
            space_before_line = (y_pos - prev_line_y_bottom) if prev_line_y_bottom else 0
            features = {
                "text": text_content, "page": page_number, "y_pos": y_pos,
                "font_size": avg_font_size, "is_bold": is_bold, "is_italic": is_italic,
                "line_height": max(0, line_height), "space_after_line": max(0, space_after_line),
                "space_before_line": max(0, space_before_line),
                "y_pos_normalized": y_pos / page_height if page_height > 0 else 0,
                "x_pos_normalized": x_pos / page_width if page_width > 0 else 0,
                "is_left_aligned": int(x_pos < 100),
                "normalized_space_after_line": max(0, space_after_line) / avg_font_size if avg_font_size > 0 else 0,
                "normalized_space_before_line": max(0, space_before_line) / avg_font_size if avg_font_size > 0 else 0,
                "is_title_case": int(is_title_case(text_content)), "is_upper": int(is_uppercase(text_content)),
                "ends_colon": int(ends_with_colon(text_content)), "starts_number": int(starts_with_number(text_content)),
                "word_count": word_count(text_content), "has_bullet_prefix": int(has_bullet_prefix(text_content)),
                "is_conventional_heading": is_conventional_heading(text_content)
            }
            all_lines_features.append(features)
            prev_line_y_bottom = y_pos + line_height
    return pd.DataFrame(all_lines_features)

def get_document_title(df_lines):
    # This function is correct and remains unchanged
    page1_lines = df_lines[df_lines['page'] == 1].copy()
    if page1_lines.empty: return ""
    page1_lines['score'] = (
        page1_lines['font_size'] * 2 + page1_lines['is_bold'] * 5 - page1_lines['y_pos_normalized'] * 10
    )
    top_half_lines = page1_lines[page1_lines['y_pos_normalized'] < 0.4]
    if top_half_lines.empty: return page1_lines.iloc[0]['text'] if not page1_lines.empty else ""
    best_line_index = top_half_lines['score'].idxmax()
    best_line = top_half_lines.loc[best_line_index]
    title_lines = [best_line]
    current_index_loc = page1_lines.index.get_loc(best_line_index)
    for i in range(current_index_loc - 1, -1, -1):
        prev_line = page1_lines.iloc[i]
        if (best_line['y_pos'] - prev_line['y_pos'] < best_line['font_size'] * 2 and
            abs(best_line['font_size'] - prev_line['font_size']) < 2):
            title_lines.insert(0, prev_line)
        else: break
    for i in range(current_index_loc + 1, len(page1_lines)):
        next_line = page1_lines.iloc[i]
        if (next_line['y_pos'] - title_lines[-1]['y_pos'] < title_lines[-1]['font_size'] * 2 and
            abs(best_line['font_size'] - next_line['font_size']) < 2):
            title_lines.append(next_line)
        else: break
    full_title = " ".join(line['text'] for line in title_lines)
    return full_title.strip()

def process_document(pdf_path, output_dir):
    filename = os.path.basename(pdf_path)
    print(f"Processing: {filename}...")
    start_time = time.time()
    try:
        pages_data = parse_pdf_to_lines(pdf_path)
    except Exception as e:
        print(f"  Error parsing {filename}: {e}"); return
    if not pages_data:
        print(f"  Could not extract lines from {filename}."); return
    df_lines = create_feature_dataframe(pages_data)
    if df_lines.empty:
        print(f"  Could not create features for {filename}."); return
        
    # The feature list the model was trained on
    feature_cols = [
        'font_size', 'is_bold', 'is_italic', 'y_pos_normalized', 'x_pos_normalized',
        'line_height', 'space_after_line', 'space_before_line',
        'normalized_space_after_line', 'normalized_space_before_line',
        'is_left_aligned', 'page', 'is_title_case', 'is_upper',
        'ends_colon', 'starts_number', 'word_count', 'has_bullet_prefix',
        'is_conventional_heading'
    ]
    
    # Ensure all columns exist, fill with 0 if not (safer for inference)
    for col in feature_cols:
        if col not in df_lines.columns:
            df_lines[col] = 0
    X_predict = df_lines[feature_cols]
    
    y_pred_proba = clf.predict_proba(X_predict)
    y_pred_indices = y_pred_proba.argmax(axis=1)
    y_pred_confidence = y_pred_proba.max(axis=1)
    
    df_lines['label'] = le.inverse_transform(y_pred_indices)
    df_lines['confidence'] = y_pred_confidence

    title = get_document_title(df_lines)

    outline = []
    headings = df_lines[df_lines['label'].str.startswith('h')]
    confident_headings = headings[headings['confidence'] >= CONFIDENCE_THRESHOLD]
    
    for _, row in confident_headings.iterrows():
        outline.append({
            "level": row['label'].upper(), "text": row['text'], "page": int(row['page'])
        })

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
    for filename in os.listdir(args.input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(args.input_dir, filename)
            process_document(pdf_path, args.output_dir)

if __name__ == "__main__":
    main()