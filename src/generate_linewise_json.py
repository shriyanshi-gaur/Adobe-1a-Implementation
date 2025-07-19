import os
import fitz  # PyMuPDF
import json
from collections import defaultdict

RAW_PDF_DIR = "../data/raw_pdfs"
OUTPUT_DIR = "../data/linewise_json"

def extract_linewise_text(pdf_path):
    doc = fitz.open(pdf_path)
    document_data = {
        "filename": os.path.basename(pdf_path),
        "pages": []
    }

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict", sort=True).get("blocks", [])

        lines = defaultdict(list)
        for block in blocks:
            if block["type"] != 0:
                continue  # skip images
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    y_top = round(span["bbox"][1], 1)
                    lines[y_top].append(span)

        # Group spans into lines
        line_items = []
        for y in sorted(lines):
            sorted_spans = sorted(lines[y], key=lambda s: s["bbox"][0])
            merged_text = " ".join([s["text"] for s in sorted_spans])
            line_items.append({
                "y": y,
                "text": merged_text,
                "spans": [
                    {
                        "text": s["text"],
                        "bbox": list(s["bbox"]),
                        "font": s["font"],
                        "size": s["size"],
                        "color": hex(s["color"])
                    } for s in sorted_spans
                ]
            })

        document_data["pages"].append({
            "page_number": page_num + 1,
            "lines": line_items
        })

    doc.close()
    return document_data

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename in os.listdir(RAW_PDF_DIR):
        if not filename.endswith(".pdf"):
            continue
        pdf_path = os.path.join(RAW_PDF_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".pdf", "_linewise.json"))
        print(f"🔍 Processing {filename}...")
        try:
            json_data = extract_linewise_text(pdf_path)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Saved to {output_path}")
        except Exception as e:
            print(f"❌ Failed on {filename}: {e}")

if __name__ == "__main__":
    main()
