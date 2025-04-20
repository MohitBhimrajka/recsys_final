# Interim Report - OULAD Course Recommendation System

**Date:** April 20, 2025 (Update as needed)

## 1. Introduction

This report outlines the progress made on the OULAD Course Recommendation System project. The primary goal is to build a system recommending relevant course presentations (`code_module` + `code_presentation`) to students based on interaction history, demographics, and course features. This interim report covers the initial phases: Exploratory Data Analysis (EDA), Data Preprocessing, and the implementation and evaluation of baseline recommendation models.

## 2. EDA Summary

An initial Exploratory Data Analysis was performed on the raw OULAD CSV files (`assessments`, `courses`, `studentAssessment`, `studentInfo`, `studentRegistration`, `studentVle`, `vle`). Key findings include:

*   **Data Size:** The dataset is substantial, especially `studentVle` (over 10 million interaction logs).
*   **Student Info:** Diverse demographics (region, education, IMD band), with `imd_band` having some missing values (handled by imputation/categorization). Age is skewed towards '0-35'. A significant portion of students withdraw ('Withdrawn' final result).
*   **Course Info:** 7 unique modules offered across 22 presentations, typically lasting around 260 days.
*   **Interactions (`studentVle`):** VLE interactions (`sum_click`) are highly skewed, suggesting log transformation or careful handling for implicit feedback. Interaction frequency correlates positively with final outcomes (Pass/Distinction).
*   **Registrations:** Students often register well before the official start date. Unregistration data is sparse (~31% recorded), suggesting many students simply stop engaging.
*   **Assessments:** TMA (Tutor Marked) is the most common assessment type. Scores are generally high among submitted assessments. Submission often occurs close to or slightly before the deadline.

(Refer to `reports/eda_summary.md` for full details and visualizations.)

## 3. Data Preprocessing

A pipeline was implemented (`src/pipelines/run_preprocessing.py`) to prepare the data for modeling:

1.  **Loading:** All raw CSVs loaded.
2.  **Cleaning:** Handled missing values (e.g., 'Missing' category for `imd_band`, dropping rows with missing critical dates/scores), converted data types. `presentation_id` created by combining `code_module` and `code_presentation`.
3.  **Interaction Filtering:** VLE interactions (`studentVle`) were filtered to only include events occurring within a student's active registration period (after registration, before unregistration if applicable).
4.  **Sparsity Filtering:** An iterative filtering process was applied to the detailed interactions, removing users with fewer than 5 interactions and items (presentations) with fewer than 5 unique interacting users (`MIN_INTERACTIONS_PER_USER=5`, `MIN_USERS_PER_ITEM=5` from `config.py`). This removed 1192 interaction records.
5.  **Aggregation:** Filtered interactions were aggregated per student per presentation to create core features: `total_clicks`, `interaction_days`, `first_interaction_date`, `last_interaction_date`.
6.  **Implicit Feedback:** An `implicit_feedback` score was calculated as `log1p(total_clicks)`.
7.  **Feature Generation:**
    *   `users_final.parquet`: Created user features (demographics mapped to numerical, credits, previous attempts) for the 25,364 unique users remaining after filtering.
    *   `items_final.parquet`: Created item features (presentation length, VLE activity type proportions) for the 22 unique presentations remaining after filtering.
8.  **Final Data:** The resulting processed dataframes are:
    *   `interactions_final.parquet`: (28466, 7) - The core data for training/evaluation.
    *   `users_final.parquet`: (25364, 9) - User features (index: `id_student`).
    *   `items_final.parquet`: (22, 22) - Item features (index: `presentation_id`).

## 4. Baseline Model Evaluation

Three baseline models were trained and evaluated using a time-based split (`last_interaction_date <= 250` for train) and ranking metrics calculated @ K=10. Evaluation used the test set (731 users, 13 items shared with train) with 100 negative samples per positive instance for efficiency.

| Model        | Precision@10 | Recall@10 | NDCG@10 |
| :----------- | :----------- | :-------- | :------ |
| Popularity   | 0.0621       | 0.6156    | 0.2153  |
| ItemCF       | 0.0988       | 0.9781    | 0.6153  |
| ALS (f=50)   | 0.0685       | 0.6778    | 0.3844  |

**Observations:**

*   **ItemCF** significantly outperformed Popularity and ALS on all metrics, especially Recall and NDCG. This suggests that recent item-item relationships are strongly predictive in this dataset. Its high recall indicates it's good at finding most of the relevant items within the top 10.
*   **Popularity** performs poorly on precision and NDCG, indicating simply recommending popular courses isn't effective for personalization. The high recall suggests relevant items *are* generally popular, but the ranking is poor.
*   **ALS** performance was modest, better than Popularity in NDCG but worse than ItemCF across the board. This configuration might need further hyperparameter tuning.

## 5. Current Status & Challenges

*   EDA, preprocessing, and baseline model (Pop, ItemCF, ALS) implementation/evaluation are complete.
*   Data sparsity was addressed through filtering, resulting in a dataset size suitable for modeling.
*   The time-based split was successfully implemented.
*   A key challenge is the relatively small number of unique items (22 course presentations) remaining after filtering, which might limit the diversity of recommendations.

## 6. Next Steps

*   Implement and evaluate Neural Collaborative Filtering (NCF).
*   Implement and evaluate a Hybrid NCF model incorporating item content features.
*   Compare the performance of all models.
*   Write the final report summarizing the project and findings.
*   (Future) Develop an API and frontend for demonstration.