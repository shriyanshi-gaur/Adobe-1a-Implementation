# Intelligent PDF Outline Extractor for Adobe Hackathon

This project is a solution for **Round 1A of the Adobe India Hackathon: "Connecting the Dots"**. The primary goal is to build a high-accuracy, high-performance tool that parses raw PDF documents and extracts a structured outline, including the document title and hierarchical headings (H1, H2, H3, etc.).

Our solution is built as a robust machine learning pipeline that was trained and validated on a diverse dataset of over 80 documents to ensure it generalizes well to complex, real-world PDFs.

## 🚀 Our Approach: A Machine Learning Pipeline

Instead of relying on fragile, rule-based methods that only check font sizes, we treated this challenge as a **supervised classification problem**. Each line in a PDF is classified as a `title`, a specific heading level (`h1`, `h2`, ...), or `other` (paragraph text, list items, etc.).

Our end-to-end pipeline is as follows:

1.  **PDF Parsing**: We use the `PyMuPDF` (`fitz`) library to perform a deep-level extraction of the PDF content. This goes beyond simple text extraction, capturing rich metadata for every line, including its bounding box, font size, font weight (bold), style (italic), and exact position on the page.

2.  **Advanced Feature Engineering**: This is the core of our solution's accuracy. For each extracted line, we compute a wide array of features to build a comprehensive "understanding" of its context and appearance. These features include:

      * **Font Features**: `font_size`, `is_bold`, `is_italic`.
      * **Positional Features**: Normalized X/Y coordinates (`x_pos_normalized`, `y_pos_normalized`), `is_centered`, `is_left_aligned`.
      * **Layout Features**: `line_height`, `space_before_line`, `space_after_line` (to understand paragraph breaks).
      * **Textual & Semantic Features**: `word_count`, `is_uppercase`, `is_title_case`, `ends_with_colon`, `starts_with_number`.
      * **Contextual Features**: We analyze a line in relation to its neighbors and the page, creating features like `prev_line_font_size`, `prev_line_is_bold`, `is_largest_font_on_page`, and `is_first_line_on_page`.
      * **Heuristic Features**: A special `is_conventional_heading` feature that recognizes common patterns like "Chapter 1", "Section A", or "Appendix".

3.  **Model Training**:

      * **Classifier**: We chose a **`RandomForestClassifier`** from scikit-learn, as it is highly effective for tabular data, robust to noisy features, and provides insights into feature importance.
      * **Handling Class Imbalance**: PDF documents naturally have far more lines of paragraph text than headings. To prevent the model from becoming biased, we use the **`SMOTE`** (Synthetic Minority Over-sampling TEchnique) to balance the class distribution in our training data.
      * **Hyperparameter Tuning**: We used `GridSearchCV` to systematically find the optimal hyperparameters for our Random Forest model, maximizing its predictive accuracy (`F1-macro` score).

4.  **Inference & Intelligent Post-Processing**:

      * When processing a new PDF, we run it through the same feature engineering pipeline.
      * The trained model predicts a label and a confidence score for each line.
      * **Robust Title Detection**: We don't just trust the `title` prediction. Our `get_document_title` function uses a fallback system: it first looks for a high-confidence `title`, then for a high-confidence `H1` on the first page, and finally uses heuristics (font size, position, and line merging) to identify the most likely title.
      * **Multi-Line Heading Merging**: We intelligently merge consecutive lines that are predicted to be the same heading level and are vertically close and horizontally aligned. This correctly reconstructs long headings that wrap onto multiple lines.

## 🛠️ Technologies and Libraries

  * **Core Logic**: Python 3
  * **PDF Processing**: `PyMuPDF` (fitz)
  * **ML & Data Science**:
      * `scikit-learn`: For the `RandomForestClassifier`, `GridSearchCV`, and `LabelEncoder`.
      * `pandas`: For efficient data manipulation.
      * `imblearn`: For using `SMOTE` to handle data imbalance.
      * `joblib`: For serializing and saving the trained model.
  * **Text Processing**: `thefuzz` (for fuzzy string matching during data labeling), `re`.

## 📁 Project Structure

The project is organized into several key scripts that represent our modular pipeline:

  * `generate_linewise_json.py`: The first step. It takes raw PDFs and converts them into structured JSON files where each line has associated metadata.
  * `extract_features_from_linewise_json.py`: The second step. It reads the linewise JSON, applies the full feature engineering logic, and matches lines against ground-truth labels to create the final `training_data.csv`.
  * `train_model.py`: Reads `training_data.csv`, applies SMOTE, runs GridSearchCV, and saves the final trained model (`.pkl`) and label mapping.
  * **`run_inference.py`**: **The main execution script for the solution.** It orchestrates the entire process for new PDFs: parsing, feature engineering, prediction, and post-processing to generate the final JSON output.
  * `utils.py`: A collection of helper functions used across all scripts for feature calculation and text cleaning.
  * `verify_data.py`: A utility script to cross-check the generated training data against the ground truth files to ensure data integrity.

## ⚙️ How to Build and Run

-

## 📝 **README.md: Docker Build and Run Instructions**

### 🔧 Build Instructions

To build the Docker image:

```bash
docker build --platform linux/amd64 -t adobe_1a_implementation_alphabits .
```

> ⚠️ Note: Ensure that the `models/` directory with `heading_classifier.pkl` and `label_mapping.pkl` is present before building the image.

---

### 🚀 Run Instructions (Windows)

To run the container on **Windows PowerShell or CMD**, use:

```powershell
docker run --rm -v "C:/absolute/path/to/test_pdfs:/app/input:ro" -v "C:/absolute/path/to/output_of_test_pdfs:/app/output" --network none adobe_1a_implementation_alphabits
```

> 🛠 Replace `C:/absolute/path/to/...` with the actual paths on your system.

Example:

```powershell
docker run --rm -v "C:/Users/Tina/Desktop/Adobe-1a-implementation/test_pdfs:/app/input:ro" -v "C:/Users/Tina/Desktop/Adobe-1a-implementation/output_of_test_pdfs:/app/output" --network none adobe_1a_implementation_alphabits
```

---

### 📂 Expected Behavior

* The container will read **all `.pdf` files** from `/app/input`
* For each PDF, it will create a `.json` file with the extracted outline in `/app/output`
* The output JSON follows this structure:

```json
{
  "title": "Document Title",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 1 },
    { "level": "H2", "text": "Details", "page": 2 }
  ]
}
```

---

### ✅ Docker Compliance Notes

* CPU-only: No GPU required
* Model size < 200MB ✅
* Runs offline (no network) ✅
* Compatible with `linux/amd64` platform ✅
* Execution time: ≤ 10s for 50-page PDF ✅

---

