import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import re
from imblearn.over_sampling import SMOTE # New import

# Paths
INPUT_CSV = "../data/processed/training_data.csv"
MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "heading_classifier.pkl")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load training data
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"Error: Training data CSV not found at {INPUT_CSV}.")
    print("Please ensure extract_features_from_linewise_json.py has been run successfully.")
    exit()

if df.empty:
    print("Error: Training data CSV is empty. Cannot train model.")
    exit()

# --- Feature Engineering ---
# Add new NLP-based features (existing)
def is_title_case(text):
    return str(text).istitle()

def is_uppercase(text):
    return str(text).isupper()

def ends_with_colon(text):
    return str(text).strip().endswith(":")

def starts_with_number(text):
    # Corrected function call and robust string conversion
    return bool(re.match(r"^\d+(\.\d+)*", str(text).strip()))

def word_count(text):
    return len(str(text).split()) # Handle potential non-string text

df['is_title_case'] = df['text'].fillna('').apply(is_title_case).astype(int)
df['is_upper'] = df['text'].fillna('').apply(is_uppercase).astype(int)
df['ends_colon'] = df['text'].fillna('').apply(ends_with_colon).astype(int)
df['starts_number'] = df['text'].fillna('').apply(starts_with_number).astype(int)
df['word_count'] = df['text'].fillna('').apply(word_count)

# --- New Structural Features ---
# Constants for normalization (MUST MATCH predict_headings.py)
PAGE_HEIGHT = 842.0 # Standard A4 height in points
PAGE_WIDTH = 595.0  # Standard A4 width in points (for x_pos normalization)

df['y_pos_normalized'] = df['y_pos'] / PAGE_HEIGHT
df['x_pos_normalized'] = df['x_pos'] / PAGE_WIDTH
df['line_height'] = df['line_height']
df['space_after_line'] = df['space_after_line']
df['space_before_line'] = df['space_before_line'] # NEW FEATURE

# Encode labels
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])

# Save label mapping for later use
label_map_path = os.path.join(MODEL_DIR, "label_mapping.pkl")
joblib.dump(le, label_map_path)

# Feature columns (updated to include all new features)
feature_cols = [
    'font_size', 'is_bold', 'y_pos_normalized', 'x_pos_normalized',
    'line_height', 'space_after_line', 'space_before_line', 'page', # Added space_before_line
    'is_title_case', 'is_upper', 'ends_colon', 'starts_number', 'word_count'
]

# Ensure all feature columns exist in the DataFrame before selecting them
missing_cols = [col for col in feature_cols if col not in df.columns]
if missing_cols:
    print(f"Error: Missing feature columns in training data: {missing_cols}")
    print("Please re-run extract_features_from_linewise_json.py with the latest code.")
    exit()

X = df[feature_cols].copy()
X['is_bold'] = X['is_bold'].astype(int)  # ensure boolean is int
y = df['label_enc']

# Check number of unique classes in the full dataset
if len(y.unique()) < 2:
    print(f"Error: The dataset contains only 1 class ({le.inverse_transform(y.unique())[0]}). Cannot train a classifier.")
    print("Please provide training data with at least two different classes.")
    exit()

# Split data with stratification
# Handle cases where stratification might fail due to extremely low support in a class
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
except ValueError as e:
    print(f"Stratified split failed: {e}.")
    # Detailed check for why stratification might fail
    class_counts_full = y.value_counts()
    problem_classes_full = class_counts_full[class_counts_full <= 1].index
    if len(problem_classes_full) > 0:
        print(f"Warning: Classes {le.inverse_transform(problem_classes_full.tolist())} have 1 or fewer samples in the full dataset.")
        print("Stratified split cannot guarantee representation of these classes in both train/test sets.")
        print("Falling back to non-stratified split, or consider removing these rare classes for training if they are noise.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_state=42) # Removed stratify for fallback
    else:
        # If not the 1-sample issue, re-raise the original error
        raise e

print(f"Original training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Classes in y_train (counts before SMOTE):")
# Corrected printing of value counts using .items()
class_counts_before_smote = y_train.value_counts().sort_index()
for encoded_label, count in class_counts_before_smote.items():
    original_label = le.inverse_transform([encoded_label])[0]
    print(f"  {encoded_label}: {count} ({original_label})")


# Check if y_train still has multiple classes after splitting, before SMOTE
if len(y_train.unique()) < 2:
    print(f"Error: y_train has only 1 class after splitting: {le.inverse_transform(y_train.unique())[0]}. SMOTE cannot be applied.")
    print("Consider: 1. More diverse training data. 2. Adjusting test_size. 3. Removing extremely rare classes from dataset before splitting.")
    exit()

# Determine k_neighbors for SMOTE dynamically based on the smallest class in y_train
min_class_samples = y_train.value_counts().min()
smote_k_neighbors = min(5, min_class_samples - 1) # Default 5, but cap at min_class_samples - 1

if smote_k_neighbors < 1:
    print(f"Warning: Smallest class in y_train has {min_class_samples} sample(s). SMOTE might not apply to all minority classes.")
    # If k_neighbors is 0, SMOTE will error if it tries to oversample a class with 1 sample.
    # Set it to 0 only if strictly necessary and only if we explicitly want to skip oversampling for classes with 1 sample.
    # A common robust strategy is to set k_neighbors to 1 if min_samples_in_class > 1.
    sm = SMOTE(random_state=42, k_neighbors=1 if min_class_samples > 1 else 0)
else:
    sm = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)

if sm.k_neighbors == 0 and len(y_train.unique()) > 1: # If SMOTE can't create samples and multiple classes exist
    print("Skipping SMOTE application because smallest class in training data has only 1 sample (k_neighbors=0).")
    X_train_res, y_train_res = X_train, y_train # Skip SMOTE if k_neighbors is problematic
elif X_train.empty: # Handle empty training set
    print("Warning: Training set is empty. Skipping SMOTE.")
    X_train_res, y_train_res = X_train, y_train
else:
    try:
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        print(f"Resampled training set shape: X_train_res={X_train_res.shape}, y_train_res={y_train_res.shape}")
        print(f"Classes in y_train_res (counts after SMOTE):")
        class_counts_after_smote = y_train_res.value_counts().sort_index()
        for encoded_label, count in class_counts_after_smote.items():
            original_label = le.inverse_transform([encoded_label])[0]
            print(f"  {encoded_label}: {count} ({original_label})")

    except ValueError as smote_error:
        print(f"Error during SMOTE resampling: {smote_error}")
        print("This often means k_neighbors is too high for the smallest class, or a class has only 1 sample.")
        print("Proceeding with original (non-resampled) training data.")
        X_train_res, y_train_res = X_train, y_train


# Train model with class_weight='balanced'
clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', max_depth=15, min_samples_leaf=1)
clf.fit(X_train_res, y_train_res)

# Save model
joblib.dump(clf, MODEL_PATH)

# Evaluate
y_pred = clf.predict(X_test)

# Map numeric predictions back to original labels for report
y_test_labels = le.inverse_transform(y_test)
y_pred_labels = le.inverse_transform(y_pred)


print("\n✅ Classification Report:\n")
# Ensure all target names are present, even if not in current split's y_test
all_target_names = sorted(le.classes_) # Sort to ensure consistent order for display
print(classification_report(y_test_labels, y_pred_labels,
                            labels=all_target_names, # Use all known labels
                            target_names=all_target_names,
                            zero_division=0)) # Explicitly handle zero division
print("\n✅ Confusion Matrix:\n")
# Ensure confusion matrix labels are ordered consistently for display
cm_display_labels = all_target_names
print(confusion_matrix(y_test_labels, y_pred_labels, labels=cm_display_labels))
print(f"Confusion Matrix Labels Order: {cm_display_labels}")