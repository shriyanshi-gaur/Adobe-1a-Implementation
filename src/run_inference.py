import os
import json
import fitz  # PyMuPDF
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
try:
    clf = joblib.load(os.path.join(MODEL_DIR, "heading_classifier.pkl"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_mapping.pkl"))
except FileNotFoundError:
    print(f"Error: Model or Label Encoder not found in {MODEL_DIR}. Please run train_model.py first.")
    exit()

def parse_pdf_to_lines(pdf_path):
    """
    Parses a PDF and extracts lines with detailed info, in memory.
    """
    doc = fitz.open(pdf_path)
    all_pages_data = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict", sort=True).get("blocks", [])
        lines_raw = defaultdict(list)
        for block in blocks:
            if block["type"] == 0:  # Text block
                for line in block.get("lines", []):
                    y0 = round(line["bbox"][1], 1)
                    lines_raw[y0].extend(line["spans"])

        line_items = []
        for y_coord in sorted(lines_raw.keys()):
            sorted_spans = sorted(lines_raw[y_coord], key=lambda s: s["bbox"][0])
            full_line_text = " ".join(s["text"] for s in sorted_spans)
            cleaned_text = clean_extracted_text(full_line_text)
            if not cleaned_text:
                continue
            
            bbox = (
                min(s["bbox"][0] for s in sorted_spans),
                min(s["bbox"][1] for s in sorted_spans),
                max(s["bbox"][2] for s in sorted_spans),
                max(s["bbox"][3] for s in sorted_spans),
            )
            line_items.append({"y": y_coord, "text": cleaned_text, "bbox": bbox, "spans": sorted_spans})
        
        all_pages_data.append({
            "page_number": page_num + 1,
            "lines": line_items,
            "page_height": page.rect.height,
            "page_width": page.rect.width
        })
    doc.close()
    return all_pages_data

def create_feature_dataframe(pages_data):
    """Creates a DataFrame with features for every line in the document."""
    all_lines_features = []
    for page_data in pages_data:
        lines = page_data["lines"]
        page_number = page_data["page_number"]
        page_height = page_data["page_height"]
        page_width = page_data["page_width"]
        
        prev_line_y_bottom = None
        for i, line_obj in enumerate(lines):
            spans = line_obj.get("spans", [])
            if not spans: continue

            text_content = line_obj["text"]
            avg_font_size = sum(s["size"] for s in spans) / len(spans) if spans else 0
            is_bold = int(any("bold" in s["font"].lower() for s in spans))
            is_italic = int(any("italic" in s["font"].lower() for s in spans))
            y_pos = line_obj["y"]
            x_pos = line_obj["bbox"][0]
            line_height = line_obj["bbox"][3] - line_obj["bbox"][1]

            next_line_y = lines[i + 1]["y"] if i + 1 < len(lines) else None
            space_after_line = (next_line_y - (y_pos + line_height)) if next_line_y else 0
            space_before_line = (y_pos - prev_line_y_bottom) if prev_line_y_bottom else 0
            
            features = {
                "text": text_content,
                "page": page_number,
                "y_pos": y_pos,
                "font_size": avg_font_size,
                "is_bold": is_bold,
                "is_italic": is_italic,
                "line_height": max(0, line_height),
                "space_after_line": max(0, space_after_line),
                "space_before_line": max(0, space_before_line),
                "y_pos_normalized": y_pos / page_height,
                "x_pos_normalized": x_pos / page_width,
                "is_left_aligned": int(x_pos < 100),
                "normalized_space_after_line": max(0, space_after_line) / avg_font_size if avg_font_size > 0 else 0,
                "normalized_space_before_line": max(0, space_before_line) / avg_font_size if avg_font_size > 0 else 0,
                "is_title_case": int(is_title_case(text_content)),
                "is_upper": int(is_uppercase(text_content)),
                "ends_colon": int(ends_with_colon(text_content)),
                "starts_number": int(starts_with_number(text_content)),
                "word_count": word_count(text_content),
                "has_bullet_prefix": int(has_bullet_prefix(text_content)),
            }
            all_lines_features.append(features)
            prev_line_y_bottom = y_pos + line_height
            
    return pd.DataFrame(all_lines_features)

def get_document_title(df_lines):
    """
    A refined heuristic to find the document title by identifying the most
    prominent text block on the first page.
    """
    page1_lines = df_lines[df_lines['page'] == 1].copy()
    if page1_lines.empty:
        return ""

    # --- Scoring ---
    # Calculate a "prominence score" for each line.
    # We prioritize large font size, bold text, and being high on the page.
    page1_lines['score'] = (
        page1_lines['font_size'] * 2 +
        page1_lines['is_bold'] * 5 -
        page1_lines['y_pos_normalized'] * 10
    )

    # --- Candidate Selection ---
    # Find the single most prominent line in the top 40% of the page
    top_half_lines = page1_lines[page1_lines['y_pos_normalized'] < 0.4]
    if top_half_lines.empty:
        return page1_lines.iloc[0]['text'] if not page1_lines.empty else ""

    best_line_index = top_half_lines['score'].idxmax()
    best_line = top_half_lines.loc[best_line_index]
    
    # --- Multi-line Title Logic ---
    # Check for neighboring lines that are also part of the title
    title_lines = [best_line]
    
    # Look upwards from the best line
    current_index_loc = page1_lines.index.get_loc(best_line_index)
    for i in range(current_index_loc - 1, -1, -1):
        prev_line = page1_lines.iloc[i]
        # Check if the line above is vertically close and has a similar font size
        if (best_line['y_pos'] - prev_line['y_pos'] < best_line['font_size'] * 2 and
            abs(best_line['font_size'] - prev_line['font_size']) < 2):
            title_lines.insert(0, prev_line)
        else:
            break  # Stop if there's a large gap or font size changes
    
    # Look downwards from the best line
    for i in range(current_index_loc + 1, len(page1_lines)):
        next_line = page1_lines.iloc[i]
        # Check if the line below is vertically close and has a similar font size
        if (next_line['y_pos'] - title_lines[-1]['y_pos'] < title_lines[-1]['font_size'] * 2 and
            abs(best_line['font_size'] - next_line['font_size']) < 2):
            title_lines.append(next_line)
        else:
            break

    # Combine the text from all identified title lines
    full_title = " ".join(line['text'] for line in title_lines)
    
    return full_title.strip()

def process_document(pdf_path, output_dir):
    """Main function to process a single PDF and generate the JSON output."""
    filename = os.path.basename(pdf_path)
    print(f"Processing: {filename}...")
    start_time = time.time()

    pages_data = parse_pdf_to_lines(pdf_path)
    if not pages_data:
        print(f"  Could not extract any lines from {filename}.")
        return

    df_lines = create_feature_dataframe(pages_data)
    if df_lines.empty:
        print(f"  Could not create features for {filename}.")
        return
        
    feature_cols = [col for col in clf.feature_names_in_ if col in df_lines.columns]
    X_predict = df_lines[feature_cols]
    
    y_pred_enc = clf.predict(X_predict)
    df_lines['label'] = le.inverse_transform(y_pred_enc)

    # Use the new, robust heuristic to get the title
    title = get_document_title(df_lines)

    outline = []
    headings = df_lines[df_lines['label'].str.startswith('h')]
    for _, row in headings.iterrows():
        text = row['text'].strip()
        
        # Post-processing filters to clean up noisy predictions
        if text.endswith('.'):
            continue
        if row['word_count'] > 15:
            continue
        if text and text[0].islower() and not row['starts_number']:
            continue
        if row['word_count'] < 10 and re.match(r'^(page\s*\d+|table\s\d+|figure\s\d+)', text, re.I):
            continue
        
        outline.append({
            "level": row['label'].upper(),
            "text": text,
            "page": int(row['page'])
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