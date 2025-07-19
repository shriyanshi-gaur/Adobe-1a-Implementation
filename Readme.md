# PDF Document Outline Extractor

This project provides a machine learning-based solution to extract structured outlines (Title, H1, H2, H3, H4 headings with their respective page numbers) from PDF documents. The solution is designed to be modular and efficient, leveraging layout and text-based features to classify document lines.

## 🚀 Project Overview

The core idea is to convert unstructured PDF content into a structured format, extract rich features, train a classification model, and then use this model to predict and generate a hierarchical outline of any new PDF.

The pipeline consists of the following steps:

1.  **PDF to Linewise JSON Conversion**: Converts raw PDF files into a line-by-line structured JSON format, preserving essential layout information like font size, font type, bold status, bounding box, and page number.
      * **Example Line Data**:
        ```json
        {
          "y": 72.1,
          "text": "The Ontario Digital Library",
          "spans": [
            {"text": "The Ontario Digital Library", "bbox": [95.4, 72.1, 238.6, 94.6], "font": "Arial-Black", "size": 15.96, "color": "0x0"}
          ]
        }
        ```
2.  **Feature Extraction and Ground Truth Labeling**: Each line from the generated linewise JSON is enriched with various features (both style-based and NLP-based). These lines are then labeled using provided ground truth JSONs, employing fuzzy matching to identify titles and various heading levels (h1, h2, h3, h4). All other lines are labeled as "other". The results are stored in a CSV file (`training_data.csv`).
3.  **Machine Learning Model Training**: A Random Forest Classifier is trained on the extracted features and labels. To handle class imbalance (where 'other' lines significantly outnumber headings), techniques like `SMOTE` (Synthetic Minority Over-sampling Technique) and `class_weight='balanced'` are employed. The trained model and a `LabelEncoder` (for mapping numerical predictions back to human-readable labels) are saved for future predictions.
4.  **Outline Prediction**: For a new input PDF (first converted to linewise JSON), the trained model predicts the label for each line. These predictions are then assembled into the final structured JSON outline format.

## 📂 Project Structure

```
.
├── data/
│   ├── gt_json/               # Ground Truth JSON files
│   │   └── E0H1CM114.json     # Example GT for E0H1CM114.pdf
│   ├── linewise_json/         # Intermediate linewise JSON files from PDFs
│   │   └── E0H1CM114_linewise.json # Example linewise JSON for E0H1CM114.pdf
│   ├── processed/             # Processed data, including training_data.csv
│   │   └── training_data.csv  # Generated training data
│   └── raw_pdfs/              # Original raw PDF files
│       └── E0H1CM114.pdf      # Example input PDF
├── models/                    # Trained ML models and label encoders
│   ├── heading_classifier.pkl # Trained Random Forest model
│   └── label_mapping.pkl      # LabelEncoder mapping
├── output/                    # Predicted outline JSONs
│   └── E0H1CM114_predicted_outline.json # Example predicted output
└── src/                       # Source code for the pipeline
    ├── generate_linewise_json.py
    ├── extract_features_from_linewise_json.py
    ├── train_model.py
    └── predict_headings.py
```

## ✨ Features

  * **Multimodal Features**: Combines both visual/layout features (font size, bold status, y-position, x-position, line height, spacing before/after line, page number) and NLP-based textual features (title case, uppercase, ends with colon, starts with number, word count).
  * **Robust Labeling**: Uses fuzzy matching and heuristics to accurately label lines based on ground truth, even for fragmented or inconsistently formatted titles/headings.
  * **Class Imbalance Handling**: Employs `SMOTE` and `class_weight` to improve the model's ability to learn and predict minority classes (headings and titles).
  * **Structured JSON Output**: Generates a clean, hierarchical JSON output with `title` and `outline` (levels H1, H2, H3, H4).
  * **Modularity**: The pipeline is broken down into distinct scripts, allowing for easy understanding, maintenance, and reuse.

## ⚙️ Setup and Installation

### Prerequisites

  * Python 3.8+
  * `pip` (Python package installer)

### Install Dependencies

It's highly recommended to use a virtual environment to manage project dependencies:

```bash
# Navigate to your project's root directory (where src/ is)
python -m venv venv
.\venv\Scripts\activate # On Windows
# source venv/bin/activate # On macOS/Linux

pip install pandas PyMuPDF scikit-learn joblib imbalanced-learn
```

## 🚀 How to Run the Solution

Ensure you have your raw PDF files in `data/raw_pdfs/` and their corresponding ground truth JSONs in `data/gt_json/`. For example, `E0H1CM114.pdf` in `raw_pdfs` should have `E0H1CM114.json` in `gt_json`.

Execute the scripts in the following order from the `src/` directory:

1.  **Generate Linewise JSONs**:
    This step processes raw PDFs and creates a line-by-line JSON representation for each PDF in `data/linewise_json/`.

    ```bash
    python generate_linewise_json.py
    ```

2.  **Extract Features and Create Training Data**:
    This script reads the linewise JSONs and ground truth JSONs, extracts features, and labels each line. The resulting training data is saved to `data/processed/training_data.csv`.

    ```bash
    python extract_features_from_linewise_json.py
    ```

3.  **Train the ML Model**:
    This step trains the RandomForestClassifier using the generated `training_data.csv`. It saves the trained model and label encoder to the `models/` directory. It will also print a classification report and confusion matrix, detailing the model's performance on the test set.

    ```bash
    python train_model.py
    ```

      * **Note**: If your dataset is very small or has extremely rare classes (e.g., only 1 sample for a specific heading level), `SMOTE` might be skipped for those classes, and the stratification in `train_test_split` might warn or fall back to a non-stratified split. This is normal behavior for very sparse data.

4.  **Predict Headings and Generate Outline**:
    This script takes a specific linewise JSON file (e.g., `E0H1CM114_linewise.json`) from `data/linewise_json/`, runs it through the trained model, and generates the final structured outline JSON in `data/output/`. You can change the `INPUT_JSON_PATH` variable in this script to process different linewise JSONs.

    ```bash
    python predict_headings.py
    ```

## 🎯 Solution Constraints Adherence

  * **Execution time**: The modular design and use of efficient libraries (PyMuPDF, scikit-learn) aim to keep execution time low. For a 50-page PDF, the total pipeline should complete within reasonable limits (expected $\\le 10$ seconds).
  * **Model size**: RandomForest models are generally compact, aiming to be within the $\\le 200$MB limit.
  * **Network**: No internet access or external API calls are required once dependencies are installed. The solution runs entirely offline.
  * **Runtime**: The solution is built on CPU-friendly libraries (PyMuPDF, scikit-learn, pandas) and should run on standard CPU configurations (8 CPUs, 16 GB RAM).

## 📈 Performance and Accuracy

The model's performance, particularly for minority heading classes, has shown improvement through:

  * **Comprehensive Feature Engineering**: Utilizing both stylistic (font size, bold, position, spacing) and semantic (NLP-based) features.
  * **Class Imbalance Handling**: Employment of `SMOTE` and `class_weight='balanced'` in the RandomForest Classifier helps the model learn from underrepresented heading types.
  * **Heuristics for Difficult Cases**: Specific heuristics are applied in the labeling phase for training data generation, particularly for fragmented titles and numbered headings, to ensure the ground truth is accurately reflected in the features.

While overall accuracy remains high due to the dominant 'other' class, the `macro avg` F1-score is a better indicator of balanced performance across all classes, and it has shown positive trends. Continuous improvement can be achieved by:

  * Collecting more diverse labeled data, especially for rare heading types.
  * Further refining feature engineering, potentially adding context-aware features like relative x-position, indentation levels, or analyzing the *flow* of text around headings.
  * More exhaustive hyperparameter tuning of the RandomForest model.

## 🤝 Contributing

Feel free to fork the repository, open issues, and submit pull requests.

## 📄 License

(Consider adding your chosen license, e.g., MIT, Apache 2.0, if applicable)