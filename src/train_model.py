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
INPUT_CSV = "/data/processed/training_data.csv"
MODEL_DIR = "/models"
MODEL_PATH = os.path.join(MODEL_DIR, "heading_classifier.pkl")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_mapping.pkl")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Load training data
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"Error: Training data CSV not found at {INPUT_CSV}. Please run extract_features_from_linewise_json.py first.")
    exit()

# Use a fixed, final list of features for training
feature_cols = [
    'font_size', 'is_bold', 'is_italic', 'y_pos_normalized', 'x_pos_normalized',
    'line_height', 'space_after_line', 'space_before_line',
    'normalized_space_after_line', 'normalized_space_before_line',
    'is_left_aligned', 'page', 'is_title_case', 'is_upper',
    'ends_colon', 'starts_number', 'word_count', 'has_bullet_prefix',
    'is_conventional_heading'
]

# Ensure all feature columns exist in the DataFrame
missing_features = [col for col in feature_cols if col not in df.columns]
if missing_features:
    print(f"Error: Missing features in training_data.csv: {missing_features}")
    print("Please ensure extract_features_from_linewise_json.py generates all required features.")
    exit()

X = df[feature_cols]
y = df['label']

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Save the label encoder
joblib.dump(le, LABEL_MAP_PATH)
print(f"Label encoder saved to {LABEL_MAP_PATH}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Apply SMOTE to handle class imbalance
print("Class distribution before SMOTE:", pd.Series(le.inverse_transform(y_train)).value_counts().to_dict())

# --- FIX: Set k_neighbors to a value smaller than the smallest class size - 1 ---
# Since n_samples_fit was 4, k_neighbors must be <= 3.
smote = SMOTE(random_state=42, k_neighbors=3) 

X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Class distribution after SMOTE:", pd.Series(le.inverse_transform(y_train_res)).value_counts().to_dict())

# --- Hyperparameter Tuning with GridSearchCV ---
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [15, 20],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring='f1_macro',
    verbose=2
)

print("\nStarting hyperparameter tuning with GridSearchCV...")
start_time = time.time()
grid_search.fit(X_train_res, y_train_res)
end_time = time.time()
print(f"GridSearchCV took {end_time - start_time:.2f} seconds.")

best_clf = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation F1-macro score: {grid_search.best_score_:.4f}")

# Evaluate the best model on the test set
y_pred = best_clf.predict(X_test)
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save the trained model
joblib.dump(best_clf, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

print("\nTraining complete.")