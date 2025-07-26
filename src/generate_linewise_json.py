import os
import fitz
import json
from collections import defaultdict
import time
from utils import clean_extracted_text

RAW_PDF_DIR = "/data/raw_pdfs"
OUTPUT_DIR = "/data/linewise_json"
MIN_FONT_SIZE = 6

def extract_linewise_text(pdf_path):
    doc = fitz.open(pdf_path)
    document_data = { "filename": os.path.basename(pdf_path), "pages": [] }
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict", sort=True).get("blocks", [])
        lines_raw = defaultdict(list)
        for block in blocks:
            if block["type"] != 0: continue
            for line_block in block.get("lines", []):
                for span in line_block.get("spans", []):
                    if span['size'] < MIN_FONT_SIZE: continue
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
        document_data["pages"].append({
            "page_number": page_num + 1, "page_width": page.rect.width,
            "page_height": page.rect.height, "lines": line_items
        })
    doc.close()
    return document_data

def main():
    start_time_total = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processed_count = 0
    print("Starting PDF to Linewise JSON conversion...")
    for filename in os.listdir(RAW_PDF_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(RAW_PDF_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename.replace(".pdf", "_linewise.json"))
            print(f"Processing {filename}...")
            try:
                json_data = extract_linewise_text(pdf_path)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                processed_count += 1
            except Exception as e:
                print(f"❌ Failed on {filename}: {e}")
    end_time_total = time.time()
    print(f"\n--- Summary ---\nProcessed {processed_count} PDFs in {end_time_total - start_time_total:.2f} seconds.")
    print(f"Linewise JSON files saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()