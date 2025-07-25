# import re
# import difflib
# from thefuzz import fuzz

# def normalize(text):
#     text = str(text).strip()
#     text = re.sub(r'[:;,.]\s*$', '', text)
#     text = re.sub(r'\s+', ' ', text)
#     return text.lower()

# def fuzzy_match(text1, text2, threshold=85):
#     if not text1 or not text2:
#         return False
#     return fuzz.token_sort_ratio(normalize(text1), normalize(text2)) >= threshold

# def is_title_case(text):
#     return str(text).istitle()

# def is_uppercase(text):
#     return str(text).isupper()

# def ends_with_colon(text):
#     return str(text).strip().endswith(":")

# def starts_with_number(text):
#     return bool(re.match(r"^\d+(\.\d+)*", str(text).strip()))

# def word_count(text):
#     return len(str(text).split())

# def has_bullet_prefix(text):
#     return bool(re.match(r"^[\s•\-*—]+\s*", str(text).strip()))

# def clean_extracted_text(text):
#     text = str(text)
#     text = re.sub(r'\s*\(cid:\d+\)\s*', '', text)
#     text = re.sub(r'\s+', ' ', text)
#     text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
#     text = text.replace("ΤΡΑΜΡOLINE", "TRAMPOLINE")
#     return text.strip()


import re
import difflib
from thefuzz import fuzz
from collections import defaultdict

def normalize(text):
    text = str(text).strip()
    text = re.sub(r'[:;,.]\s*$', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

def fuzzy_match(text1, text2, threshold=85):
    if not text1 or not text2:
        return False
    return fuzz.token_sort_ratio(normalize(text1), normalize(text2)) >= threshold

def is_title_case(text):
    # A stricter title case: first letter of each word capitalized, others lowercase
    # Allows for common exceptions like "of", "and", "the" etc.
    words = str(text).split()
    if not words:
        return False
    return all(word.istitle() or (word.lower() in ['a', 'an', 'the', 'and', 'or', 'but', 'for', 'nor', 'on', 'at', 'to', 'from', 'by', 'of', 'in', 'with']) for word in words)


def is_uppercase(text):
    return str(text).isupper() and len(str(text).strip()) > 1 # Avoid single chars being true

def ends_with_colon(text):
    return str(text).strip().endswith(":")

def starts_with_number(text):
    return bool(re.match(r"^\d+(\.\d+)*(\s+|$)", str(text).strip()))

def word_count(text):
    return len(str(text).split())

def has_bullet_prefix(text):
    return bool(re.match(r"^[•*—\-–•]+\s*", str(text).strip()))

def clean_extracted_text(text):
    text = str(text)
    text = re.sub(r'\s*\(cid:\d+\)\s*', '', text) # Remove (cid:xxxx)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text) # Handle hyphenated words across lines
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces/newlines with single space
    return text

# --- NEW UTILITY FUNCTIONS FOR FEATURE ENGINEERING ---

def is_conventional_heading(text):
    """Checks for common heading patterns using domain knowledge, enhanced."""
    text = text.strip().lower()
    # Expanded patterns for H1s and similar structures
    patterns = [
        r'^(appendix\s+[a-z\d]+[:.]?|section\s+\d+[:.]?|chapter\s+\d+[:.]?|part\s+[ivxlcdm]+[:.]?|\d+\.\s+)', # Existing
        r'^introduction[:.]?$',
        r'^conclusion[:.]?$',
        r'^summary[:.]?$',
        r'^table of contents$',
        r'^list of figures$',
        r'^list of tables$',
        r'^abstract$',
        r'^acknowledgments$',
        r'^preface$',
        r'^foreword$',
        r'^bibliography$',
        r'^references$',
        r'^index$',
        r'^glossary$',
        r'^(chapter|section|part)\s+\w+(\s+.*)?$', # e.g., "Chapter One Introduction"
    ]
    for pattern in patterns:
        if re.match(pattern, text):
            return 1
    return 0

def is_centered(bbox, page_width, tolerance_ratio=0.05):
    """Checks if the text block is horizontally centered on the page."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    text_width = bbox[2] - bbox[0] # x1 - x0
    center_of_text = bbox[0] + text_width / 2
    page_center = page_width / 2
    
    # Check if the center of the text is within a tolerance range of the page center
    return abs(center_of_text - page_center) <= (page_width * tolerance_ratio)

def get_page_font_sizes(page_lines):
    """Collects all unique font sizes and their counts on a given page."""
    font_sizes = defaultdict(int)
    for line in page_lines:
        # Assuming 'spans' contain font information in linewise JSON
        if 'spans' in line:
            for span in line['spans']:
                font_sizes[round(span['size'], 1)] += 1
    return font_sizes

def get_page_line_data(page_lines):
    """Gathers essential data for all lines on a page for relative feature calculation."""
    line_data = []
    for line in page_lines:
        if line and 'spans' in line and line['spans']:
            # Take the first span's font size as representative for the line
            font_size = round(line['spans'][0]['size'], 1) if line['spans'] else 0
            line_data.append({
                'text': line['text'],
                'y_pos': line['bbox'][1], # y0
                'font_size': font_size,
                # FIX: Ensure 'flags' is treated as a string before calling .lower()
                'is_bold': 'bold' in str(line['spans'][0].get('flags', '')).lower() if line['spans'] else False 
            })
    return line_data

def has_keywords(text, keywords):
    """Checks if any of the given keywords are present in the text."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)

# Specific keywords for titles and H1s (can be expanded based on data)
TITLE_KEYWORDS = ["title", "document", "report", "overview", "summary", "manual", "guide", "handbook", "book", "chapters", "table of contents"]
H1_KEYWORDS = ["chapter", "section", "part", "introduction", "conclusion", "summary", "abstract", "appendix", "bibliography", "references"]