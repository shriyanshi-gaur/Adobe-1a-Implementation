# PDF Outline Extractor

A high-performance solution for **Round 1A of the Adobe India Hackathon**, this project uses a machine learning pipeline to parse raw PDF documents and extract a structured outline, including the document title and hierarchical headings (H1, H2, etc.).

Our model was trained and validated on a diverse dataset of over 80 documents, ensuring it generalizes well to complex, real-world PDFs.

## 🚀 Our Approach: A Machine Learning Pipeline

Instead of relying on fragile, rule-based methods, we treated this challenge as a **supervised classification problem**. Each line in a PDF is classified as a `title`, a specific heading level (`h1`, `h2`, etc.), or `other` text.

Our end-to-end pipeline operates as follows:

1.  **PDF Parsing**: Using the `PyMuPDF` (fitz) library, we perform a deep-level extraction of PDF content, capturing rich metadata for every line including its bounding box, font properties (size, bold, italic), and precise position on the page.

2.  **Advanced Feature Engineering**: At the core of our accuracy is a wide array of computed features that build a comprehensive understanding of each line's context. These include:

      * **Font Features**: `font_size`, `is_bold`, `is_italic`.
      * **Positional Features**: Normalized X/Y coordinates, `is_centered`, `is_left_aligned`.
      * **Layout Features**: `line_height`, `space_before_line`, `space_after_line`.
      * **Textual Features**: `word_count`, `is_uppercase`, `is_title_case`, `ends_with_colon`, `starts_with_number`.
      * **Contextual Features**: Analysis of a line relative to its neighbors, such as `prev_line_font_size` and `is_largest_font_on_page`.
      * **Heuristic Features**: A special `is_conventional_heading` feature that recognizes patterns like "Chapter 1" or "Appendix A".

3.  **Model Training**:

      * **Classifier**: We chose a **`RandomForestClassifier`** for its effectiveness with tabular data and robustness to noisy features.
      * **Handling Class Imbalance**: To prevent model bias from the high volume of paragraph text, we use **`SMOTE`** (Synthetic Minority Over-sampling Technique) to balance the training data.
      * **Hyperparameter Tuning**: We employ `GridSearchCV` to systematically find the optimal model hyperparameters, maximizing predictive accuracy.

4.  **Inference & Intelligent Post-Processing**:

      * **Robust Title Detection**: Our `get_document_title` function uses a fallback system—it first seeks a high-confidence `title`, then a high-confidence `H1` on the first page, and finally uses heuristics to identify the most likely title.
      * **Multi-Line Heading Merging**: We intelligently merge consecutive lines predicted as the same heading level that are vertically close and horizontally aligned, correctly reconstructing long headings that wrap to new lines.

## 🛠️ Technologies and Libraries

  * **Core Logic**: Python 3
  * **PDF Processing**: `PyMuPDF` (fitz)
  * **ML & Data Science**: `scikit-learn` (RandomForest, GridSearchCV), `pandas`, `imblearn` (SMOTE), `joblib`
  * **Text Processing**: `thefuzz`, `re`

## 📁 Project Structure

  * `generate_linewise_json.py`: Converts raw PDFs into structured JSON files with line-by-line metadata.
  * `extract_features_from_linewise_json.py`: Applies feature engineering and ground-truth labels to create the `training_data.csv`.
  * `train_model.py`: Trains the Random Forest classifier and saves the final model.
  * **`run_inference.py`**: **The main execution script.** Orchestrates the full pipeline for new PDFs to generate the final JSON output.
  * `utils.py`: A collection of helper functions for feature extraction and text cleaning.
  * `verify_data.py`: A utility script to ensure data integrity between the ground truth and the training set.

## ⚙️ How to Build and Run

This solution is containerized with Docker for easy, consistent deployment.

### Build the Docker Image

> **Prerequisite**: Ensure the `models/` directory containing `heading_classifier.pkl` and `label_mapping.pkl` is present in the project root.

Execute the following command in your terminal at the project's root:

```bash
docker build --platform linux/amd64 -t adobe_1a_implementation_alphabits .
```

### Run the Container

The command below maps local directories for input and output.

**On Windows (PowerShell/CMD):**

```powershell
docker run --rm -v "C:/absolute/path/to/test_pdfs:/app/input:ro" -v "C:/absolute/path/to/output_of_test_pdfs:/app/output" --network none adobe_1a_implementation_alphabits
```

**On macOS / Linux:**

```bash
docker run --rm -v "$(pwd)/test_pdfs:/app/input:ro" -v "$(pwd)/output_of_test_pdfs:/app/output" --network none adobe_1a_implementation_alphabits
```

> **Note**: For Windows, you must replace `C:/absolute/path/to/...` with the actual, full path to your project folders.

### Expected Behavior

  * The container automatically processes all `.pdf` files from the mounted input directory.
  * For each input PDF, a corresponding `.json` file containing the extracted outline is generated in the output directory.
  * The output JSON will conform to the following structure:
    ```json
    {
      "title": "Document Title",
      "outline": [
        { "level": "H1", "text": "Introduction", "page": 1 },
        { "level": "H2", "text": "Details", "page": 2 }
      ]
    }
    ```

### ✅ Hackathon Compliance

  * **Platform**: `linux/amd64` compatible
  * **Execution**: CPU-only; no GPU required
  * **Network**: Runs fully offline
  * **Constraints**: Model size under 200MB and execution time under 10 seconds for a 50-page PDF