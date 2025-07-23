import re
import difflib
from thefuzz import fuzz

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
    return str(text).istitle()

def is_uppercase(text):
    return str(text).isupper()

def ends_with_colon(text):
    return str(text).strip().endswith(":")

def starts_with_number(text):
    return bool(re.match(r"^\d+(\.\d+)*", str(text).strip()))

def word_count(text):
    return len(str(text).split())

def has_bullet_prefix(text):
    return bool(re.match(r"^[\s•\-*—]+\s*", str(text).strip()))

def clean_extracted_text(text):
    text = str(text)
    text = re.sub(r'\s*\(cid:\d+\)\s*', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
    text = text.replace("ΤΡΑΜΡOLINE", "TRAMPOLINE")
    return text.strip()