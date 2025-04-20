# OULAD MOOC Course Recommendation System

## Project Overview

This project aims to build a course recommendation system using the Open University Learning Analytics Dataset (OULAD). The goal is to recommend relevant course presentations (`code_module` + `code_presentation`) to students based on their historical interactions, demographics, and characteristics of the courses themselves.

We will explore various recommendation techniques, including:
- Baselines (Popularity, Item-based CF)
- Collaborative Filtering (Matrix Factorization via ALS/BPR, Neural Collaborative Filtering - NCF)
- Content-Based Features (derived from VLE interactions and course structure)
- Hybrid Models combining collaborative and content information

## Dataset

- **Source:** Open University Learning Analytics Dataset (OULAD)
- **Files Used:** `assessments.csv`, `courses.csv`, `studentAssessment.csv`, `studentInfo.csv`, `studentRegistration.csv`, `studentVle.csv`, `vle.csv`
- **Location:** Raw CSV files should be placed in the `data/raw/` directory.

## Project Structure

```
recsys_project/
│
├── data/
│   ├── raw/              # Original OULAD CSVs
│   └── processed/        # Processed data (Parquet files)
│
├── notebooks/            # Jupyter notebooks for EDA, development, experimentation
│
├── src/                  # Source code for the project
│   ├── config.py         # Configuration (paths, DB URI, parameters)
│   ├── data/             # Data loading, preprocessing, feature engineering, PyTorch Datasets
│   ├── database/         # Database schema (SQLAlchemy), connection, loading scripts
│   ├── evaluation/       # Evaluation metrics and protocols
│   ├── models/           # Recommender model implementations
│   ├── pipelines/        # End-to-end scripts (preprocessing, training, evaluation)
│   └── common_utils.py   # General utility functions
│
├── scripts/              # Helper shell scripts (optional)
│
├── reports/              # EDA summary, project reports
│
├── saved_models/         # Trained model artifacts
├── tests/                # Unit and integration tests
│
├── .env                  # Local environment variables (DB credentials - DO NOT COMMIT)
├── .env.example          # Template for .env
├── .gitignore            # Specifies intentionally untracked files that Git should ignore
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd recsys_final
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up PostgreSQL Database:**
    - Ensure you have PostgreSQL installed and running.
    - Create a database (e.g., `oulad_recsys`).
    - Create a database user and grant privileges.

5.  **Configure Database Connection:**
    - Copy the `.env.example` file to `.env`:
      ```bash
      cp .env.example .env
      ```
    - Edit the `.env` file and fill in your actual PostgreSQL `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, and `DB_NAME`.

6.  **Download Data:**
    - Obtain the OULAD dataset CSV files.
    - Place all required CSV files (`assessments.csv`, `courses.csv`, etc.) into the `data/raw/` directory.

7.  **Verify Setup:**
    - Run the configuration script to check paths and DB connection (optional, prints info):
      ```bash
      python src/config.py
      ```
    - Check if the raw data files are correctly detected.

## Usage

The project follows these main phases:

### Phase 1: Exploratory Data Analysis (EDA)

1.  Start Jupyter Lab:
    ```bash
    jupyter lab
    ```
2.  Navigate to the `notebooks/` directory.
3.  Run `01_eda.ipynb` to explore the raw data and understand its characteristics. Review the generated `reports/eda_summary.md`.

### Phase 2: Data Preprocessing

1.  Ensure raw OULAD CSV files are placed in the `data/raw/` directory.
2.  Run the preprocessing script. This cleans the data, performs feature engineering, filters based on interaction counts, and aggregates interactions.
    ```bash
    python src/pipelines/run_preprocessing.py
    ```
3.  Verify that the following files are created in the `data/processed/` directory:
    *   `interactions_final.parquet`: Aggregated user-item interaction data with implicit feedback scores.
    *   `users_final.parquet`: Processed user features.
    *   `items_final.parquet`: Processed item (presentation) features.

### Phase 3: Database Setup & Loading (Optional but Recommended)

This step loads the processed data into a PostgreSQL database, which can be useful for querying or potential future API interaction.

1.  Make sure your PostgreSQL server is running and the connection details in the `.env` file are correct.
2.  **(First Time Only)** Setup the database schema (tables for users, presentations, aggregated interactions).
    *   **Warning:** Check the `DROP_EXISTING_TABLES` variable inside `src/pipelines/setup_database.py` before running. Set it to `False` unless you intend to delete all existing data in the target tables.
    ```bash
    python src/pipelines/setup_database.py
    ```
3.  Load the processed data generated in Phase 2 into the database tables.
    ```bash
    python src/database/load_to_db.py
    ```

### Phase 4: Model Training

Train different recommendation models using the processed interaction data. Model artifacts (e.g., `.pkl` or `.pt` files) will be saved in the `saved_models/` directory.

*   **Train Popularity Model:**
    ```bash
    python src/pipelines/train.py --model-name Popularity
    ```
*   **Train ItemCF Model:**
    ```bash
    python src/pipelines/train.py --model-name ItemCF
    ```
*   **Train ALS Model (Example with custom factors):**
    ```bash
    python src/pipelines/train.py --model-name ALS --factors 100 --iterations 30
    ```
*   **Train NCF Model (Example with specific epochs/LR):**
    ```bash
    python src/pipelines/train.py --model-name NCF --epochs 15 --lr 0.0005 --batch-size 2048
    ```
*   **Train Hybrid Model (Requires Item Features):**
    ```bash
    # This model automatically uses data/processed/items_final.parquet
    python src/pipelines/train.py --model-name Hybrid --epochs 15 --lr 0.0005 --batch-size 512
    ```
*   Check the `saved_models/` directory for the resulting model files (names include model type and key hyperparameters).

### Phase 5: Model Evaluation

Evaluate a previously trained and saved model artifact using the time-based split evaluation protocol.

*   **Evaluate a saved model:**
    ```bash
    # Replace MODEL_FILENAME.pkl or .pt with your actual saved model file
    python src/pipelines/evaluate.py --model-path saved_models/MODEL_FILENAME.pt --k 10 --neg-samples 100
    ```
    *   `--k`: The 'k' value for metrics@k (e.g., Precision@10).
    *   `--neg-samples`: Number of negative items to sample for evaluation (100 is common for faster evaluation; use 0 or omit for full evaluation over all items, which can be very slow).
*   **Save evaluation metrics to a file:**
    ```bash
    python src/pipelines/evaluate.py --model-path saved_models/MODEL_FILENAME.pkl --metrics-output-path reports/evaluation/model_results.json
    ```

### Phase 6: Review Reports

*   Review the generated reports in the `reports/` directory:
    *   `eda_summary.md`: Summary of initial data exploration.
    *   `interim_report.md`: Report after implementing baseline models.
    *   `final_report.md`: Comprehensive report comparing all implemented models.
    *   `reports/evaluation/`: Directory to store saved JSON metrics files.