import re
import difflib

# --- Text Normalization and Matching ---
def normalize(text):
    """Removes trailing punctuation and extra spaces, and converts to lowercase."""
    text = str(text).strip()
    text = re.sub(r'[:;,.]\s*$', '', text)
    text = re.sub(r'\s+', ' ', text)
    return "".join(text.lower().split())

def fuzzy_match(text1, text2, threshold=0.80):
    """Performs a fuzzy string match between two texts."""
    if not text1 or not text2:
        return False
    return difflib.SequenceMatcher(None, normalize(text1), normalize(text2)).ratio() >= threshold

# --- NLP & Pattern-based Feature Functions ---
def is_title_case(text):
    """Checks if a string is in title case."""
    return str(text).istitle()

def is_uppercase(text):
    """Checks if a string is all uppercase."""
    return str(text).isupper()

def ends_with_colon(text):
    """Checks if a string ends with a colon."""
    return str(text).strip().endswith(":")

def starts_with_number(text):
    """Checks if a string starts with a number (e.g., 1., 2.1, 3)."""
    return bool(re.match(r"^\d+(\.\d+)*", str(text).strip()))

def word_count(text):
    """Counts the number of words in a string."""
    return len(str(text).split())

def has_bullet_prefix(text):
    """Checks for common bullet point prefixes."""
    return bool(re.match(r"^[\s•\-*—]+\s*", str(text).strip()))

def clean_extracted_text(text):
    """Applies common text cleaning rules for OCR output."""
    text = str(text)
    # Remove common OCR artifacts like (cid:xxxx)
    text = re.sub(r'\s*\(cid:\d+\)\s*', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # FIX: The 'text' argument was missing in the line below
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)

    # Handle specific unicode misinterpretations
    text = text.replace("ΤΡΑΜΡOLINE", "TRAMPOLINE")
    return text.strip()