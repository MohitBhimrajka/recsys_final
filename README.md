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

## Usage (Phase 1 - EDA)

1.  **Start Jupyter Lab:**
    ```bash
    jupyter lab
    ```
2.  **Navigate** to the `notebooks/` directory in the Jupyter Lab interface.
3.  **Open and run** `01_eda.ipynb` to explore the raw data.
4.  **Review** the generated plots and summary statistics.
5.  **Update** `reports/eda_summary.md` with your findings.

*(More sections on running preprocessing, training, evaluation will be added later)*