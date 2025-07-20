import pandas as pd
import os
import joblib
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Paths
INPUT_CSV = "../data/processed/training_data.csv"
MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "heading_classifier.pkl")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_mapping.pkl")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load training data
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"Error: Training data CSV not found at {INPUT_CSV}.")
    exit()

# --- Feature Engineering ---
DEFAULT_PAGE_HEIGHT = 842.0
DEFAULT_PAGE_WIDTH = 595.0

df['y_pos_normalized'] = df['y_pos'] / DEFAULT_PAGE_HEIGHT
df['x_pos_normalized'] = df['x_pos'] / DEFAULT_PAGE_WIDTH
df['is_left_aligned'] = (df['x_pos'] < 100).astype(int)
df['normalized_space_after_line'] = df['space_after_line'] / df['font_size'].replace(0, 1)
df['normalized_space_before_line'] = df['space_before_line'] / df['font_size'].replace(0, 1)

# Feature columns for the model
feature_cols = [
    'font_size', 'is_bold', 'is_italic',
    'y_pos_normalized', 'x_pos_normalized',
    'line_height', 'space_after_line', 'space_before_line',
    'normalized_space_after_line', 'normalized_space_before_line',
    'is_left_aligned', 'page', 'is_title_case', 'is_upper',
    'ends_colon', 'starts_number', 'word_count', 'has_bullet_prefix'
]

X = df[feature_cols]
y = df['label']

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
joblib.dump(le, LABEL_MAP_PATH)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# --- Hyperparameter Tuning with GridSearchCV ---

# 1. Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5]
}

# 2. Initialize GridSearchCV
# We use 'f1_macro' as the scoring metric because it's better for imbalanced datasets.
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid=param_grid,
    cv=5,
    n_jobs=1,
    scoring='f1_macro',
    verbose=2
)

# 3. Fit the grid search to the data
print("Starting hyperparameter tuning with GridSearchCV...")
start_time = time.time()
grid_search.fit(X_train_res, y_train_res)
end_time = time.time()
print(f"GridSearchCV took {end_time - start_time:.2f} seconds.")

# 4. Get the best model and its parameters
print("\nBest parameters found by GridSearchCV:")
print(grid_search.best_params_)
best_clf = grid_search.best_estimator_

# Save the best model found by the grid search
joblib.dump(best_clf, MODEL_PATH)
print(f"\n✅ Best model saved to: {MODEL_PATH}")

# Evaluate the best model on the test set
y_pred = best_clf.predict(X_test)
print("\n✅ Classification Report on Test Set (using the best model):\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))