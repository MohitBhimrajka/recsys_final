# Exploratory Data Analysis (EDA) Summary - OULAD Dataset

**Date:** April 20, 2025

## 1. Overview

This document summarizes the initial findings from exploring the Open University Learning Analytics Dataset (OULAD). The analysis covers the structure, distributions, missing values, and basic relationships within the key data tables (`studentInfo`, `courses`, `studentRegistration`, `studentVle`, `vle`, `assessments`, `studentAssessment`). This EDA aimed to understand the data characteristics to inform preprocessing strategies and model selection for the course presentation recommendation task.

## 2. Data Files Summary & Key Findings

*   **`assessments.csv` (206 rows, 6 columns):** Defines assessments (TMA, CMA, Exam). Contains `date` (deadline, 11 missing), and `weight`. Many assessments have 0 weight, likely formative.
    *   *Implication:* Missing deadlines might affect analysis based on submission timing. Zero-weight assessments might be less critical signals than weighted ones.
*   **`courses.csv` (22 rows, 3 columns):** Defines 7 unique course modules across 22 presentations (`code_module` + `code_presentation`). Presentation lengths mostly ~260 days.
    *   *Implication:* The target item space (`presentation_id`) is relatively small (22). This is a key constraint for recommendation diversity.
*   **`studentAssessment.csv` (173,912 rows, 5 columns):** Records student submissions. `score` has 173 missing values. Scores are generally high (mean ~75.8) among submitted assessments.
    *   *Implication:* Missing scores were dropped during preprocessing as they represent a small fraction. Assessment scores could potentially be used as a stronger implicit signal if incorporated later.
*   **`studentInfo.csv` (32,593 rows, 12 columns):** Demographic and outcome data per student-presentation.
    *   `imd_band`: 1111 missing values (~3.4%). *Decision:* Imputed with a dedicated 'Missing' category during cleaning (`clean_student_info`) to retain information.
    *   `final_result`: Significant 'Withdrawn' category (10,156 students), indicating user churn is common.
    *   Distributions: Reasonably balanced gender; varied regions; 'A Level' most common education; age skewed to '0-35'; few previous attempts (`num_of_prev_attempts` mostly 0); `studied_credits` peaks at 60/120.
    *   *Implication:* Demographic features offer potential for personalization if models can leverage them effectively (e.g., Hybrid models). The high withdrawal rate highlights the challenge of retaining engagement.
*   **`studentRegistration.csv` (32,593 rows, 5 columns):**
    *   `date_registration`: 45 missing values. *Decision:* Rows dropped during cleaning (`clean_registrations`). Registration often occurs significantly before day 0 (mean ~ -69).
    *   `date_unregistration`: ~69% missing. *Decision:* Missing values indicate the student *did not* formally unregister. An `is_unregistered` flag was created, and NaNs in the date column were filled for processing (`clean_registrations`).
    *   *Implication:* Registration and unregistration dates are crucial for filtering VLE activity to the relevant period (`filter_interactions_by_registration`).
*   **`studentVle.csv` (10,655,280 rows, 6 columns):** Core interaction logs.
    *   Volume: Very large, representing the main source of interaction data.
    *   `sum_click`: Highly right-skewed (mean 3.7, median 2, max ~7k). *Decision:* Log transformation (`log1p`) applied during aggregation (`create_interaction_features`) to create the `implicit_feedback` score, mitigating skewness and emphasizing initial interactions more than massive click counts. *Visualization:* The log-scaled histogram of `sum_click` (notebook 01) showed a more manageable distribution after transformation.
    *   Interaction Volume per User: Also skewed (mean 409 records, median 270). *Decision:* Motivated the interaction count filtering step (`apply_interaction_count_filters`) to remove users with minimal activity (e.g., < 5 interactions) that might add noise or instability to CF models.
    *   Interaction Timing: Activity peaks early and decreases. *Decision:* A time-based split (`last_interaction_date`) was chosen for evaluation to reflect this temporal nature.
*   **`vle.csv` (6,364 rows, 6 columns):** VLE item metadata.
    *   `activity_type`: Diverse types, dominated by 'resource', 'subpage', 'oucontent'. *Decision:* Proportions of these activity types were used as content features for items (`generate_item_features`).
    *   `week_from`/`week_to`: ~82% missing. *Implication:* Weekly structure information is unreliable for most VLE items and was not used as a primary feature.

## 3. Initial Relationship Insights

*   **Interaction vs. Outcome:** A clear positive correlation was observed between VLE interaction volume (`total_clicks`, `total_interactions`) and `final_result`. Students passing or achieving distinction showed significantly higher engagement (visualized via boxplots in notebook 01).
    *   *Implication:* Validates using VLE interaction counts (specifically `log1p(total_clicks)`) as a meaningful proxy for user interest/engagement (i.e., `implicit_feedback`).
*   **Demographics vs. Score:** Higher education levels correlated with higher average assessment scores.
    *   *Implication:* User features might hold predictive power, motivating exploration beyond pure CF models (like Hybrid).

## 4. Key Takeaways for Preprocessing & Modeling

1.  **Implicit Feedback:** `log1p(total_clicks)` aggregated per `(id_student, presentation_id)` serves as the primary interaction signal, balancing engagement volume while mitigating extreme skew.
2.  **Time-Based Filtering & Splitting:** Filtering interactions based on registration dates and using a time-based split (`last_interaction_date <= 250` chosen based on EDA percentile analysis in notebook 03) are crucial for realistic modeling and evaluation.
3.  **Sparsity Handling:** Filtering users and items with low interaction counts (thresholds `MIN_INTERACTIONS_PER_USER=5`, `MIN_USERS_PER_ITEM=5`) is necessary for model stability, but significantly reduces the dataset size (final: 25k users, 22 items). This impacts recommendation diversity.
4.  **Feature Engineering:**
    *   User features derived from `studentInfo` (demographics, credits) provide context.
    *   Item features derived from `courses` (length) and `vle` (activity type proportions) provide content information.
5.  **Missing Data:** Specific strategies were applied (imputation for `imd_band`, dropping sparse critical data like `date_registration`, creating flags for `date_unregistration`).
6.  **Item Space Constraint:** The final set of 22 unique course presentations after filtering is a major factor influencing model complexity choices and expected recommendation diversity.