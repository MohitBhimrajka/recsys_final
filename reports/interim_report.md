# Interim Report - OULAD Course Recommendation System

**Date:** April 20, 2025

## 1. Introduction

This report details the initial phases of the OULAD Course Recommendation System project. The objective is to recommend relevant course presentations (`code_module` + `code_presentation`) to students using the OULAD dataset. This report covers the work completed up to the implementation and evaluation of baseline recommendation models, encompassing:

*   Exploratory Data Analysis (EDA) to understand data characteristics.
*   Data Preprocessing to clean, filter, and structure the data for modeling.
*   Implementation and evaluation of Popularity, Item-Based Collaborative Filtering (ItemCF), and Alternating Least Squares (ALS) baseline models.

The findings from these initial stages provide a foundation and motivation for exploring more advanced modeling techniques in the subsequent project phases.

## 2. Exploratory Data Analysis (Key Insights)

The EDA (detailed in `reports/eda_summary.md`) revealed several critical characteristics of the OULAD dataset that directly informed the preprocessing strategy:

*   **Interaction Skewness:** Raw VLE interaction counts (`sum_click`) were heavily right-skewed. This motivated the use of a log transformation (`log1p`) to create a more balanced `implicit_feedback` score, preventing highly active users/items from dominating disproportionately.
*   **Data Sparsity:** Many users had very few interactions. To ensure model stability and focus on more engaged users/items, interaction count filtering was deemed necessary.
*   **Temporal Nature:** VLE activity showed temporal patterns (peaking early). This supported the use of a time-based split for evaluation to simulate predicting future interactions based on past behavior.
*   **Implicit Signals:** The observed positive correlation between VLE interaction volume and positive student outcomes (Pass/Distinction) validated the approach of using interaction data as the primary signal for recommendations.
*   **Dataset Constraints:** The relatively small number of unique course modules (7) and presentations (22) highlighted a potential limitation in recommendation diversity from the outset.
*   **Missing Data:** Key columns like `imd_band` and `date_unregistration` had missing values, requiring specific handling strategies (imputation, flag creation).

## 3. Data Preprocessing Pipeline

Based on the EDA, a standardized preprocessing pipeline (`src/pipelines/run_preprocessing.py`) was implemented with the following key steps and rationale:

1.  **Cleaning & Structuring:** Raw CSVs were loaded, data types corrected, missing values handled (e.g., `imd_band` imputed with 'Missing', critical missing dates/scores dropped), and a consistent `presentation_id` (e.g., 'AAA_2013J') was created.
2.  **Registration Filtering:** VLE interactions (`studentVle`) were filtered to include only activity occurring within a student's valid registration period using `filter_interactions_by_registration`, ensuring relevance.
3.  **Interaction Count Filtering:** To address data sparsity and improve model robustness, users with fewer than 5 interactions and items (presentations) interacted with by fewer than 5 users were iteratively removed using `apply_interaction_count_filters` (thresholds set in `config.py`). This significantly reduced the dataset size but focused the models on denser parts of the interaction graph. **Impact:** The initial ~10.6M VLE interactions were reduced, ultimately resulting in a final aggregated interaction dataset covering **25,364 unique users** and **22 unique items (presentations)**.
4.  **Aggregation & Implicit Feedback:** The filtered VLE interactions were aggregated per `(id_student, presentation_id)`. The core engagement signal, `implicit_feedback`, was calculated as **`log1p(total_clicks)`** using `create_interaction_features`. This score represents user interest.
5.  **Feature Generation:** User demographic features (`users_final.parquet`) and item content features (`items_final.parquet`, e.g., presentation length, VLE activity proportions) were generated for the *filtered* set of users and items, ready for use in potential hybrid models.
6.  **Time-Based Split:** The final aggregated interaction data (`interactions_final.parquet`) was split into training and testing sets using `preprocess.time_based_split`. Based on EDA distribution analysis (aiming for roughly an 80/20 split while respecting time), a threshold of **`last_interaction_date <= 250`** was used to define the training set. Interactions after day 250 formed the initial test set, which was then filtered to only include users and items present in the training set.

## 4. Baseline Model Implementation & Evaluation

Three baseline models were implemented and evaluated to establish initial performance benchmarks. Evaluation used the time-based split (training on interactions <= day 250) and calculated Precision@10, Recall@10, and NDCG@10, using 100 negative samples per positive test instance for efficiency.

### 4.1. Model Descriptions

*   **Popularity:** A non-personalized baseline that ranks items based solely on their aggregate `implicit_feedback` score across all users in the training data. Implemented in `src/models/popularity.py`.
*   **Item-Based Collaborative Filtering (ItemCF):** Calculates the cosine similarity between items based on users' interaction patterns (`implicit_feedback` scores) in the training data. Predicts a user's score for an item by computing a weighted average of the similarities between that item and items the user previously interacted with. Implemented in `src/models/item_cf.py`.
*   **Alternating Least Squares (ALS):** A matrix factorization technique that learns latent vector representations (embeddings) for users and items by minimizing reconstruction error on the user-item interaction matrix. Implemented using the `implicit` library wrapper (`src/models/matrix_factorization.py`) with parameters: `factors=50`, `regularization=0.05`, `iterations=25`.

### 4.2. Evaluation Results (K=10, Neg Samples=100)

| Model        | Precision@10 | Recall@10 | NDCG@10 |
| :----------- | :----------- | :-------- | :------ |
| Popularity   | 0.0621       | 0.6156    | 0.2153  |
| **ItemCF**   | **0.0988**   | **0.9781**| **0.6153** |
| ALS (f=50)   | 0.0685       | 0.6778    | 0.3844  |

### 4.3. Analysis of Baseline Performance

*   **ItemCF Dominance:** ItemCF clearly outperformed both Popularity and ALS on all metrics. Its significantly higher Precision and NDCG indicate it not only finds relevant items but also ranks them effectively. The near-perfect Recall suggests that for users in the test set, ItemCF almost always placed their relevant future interactions within the top 10 recommendations. This points to strong, predictive co-interaction patterns within the OULAD VLE data (i.e., students interacting with item A are highly likely to interact with similar items B and C).
*   **Popularity Limitations:** The Popularity baseline, while achieving decent recall (meaning relevant items *are* generally popular), performed poorly on precision and NDCG. This confirms that simply recommending the most globally popular courses is insufficient for providing relevant, personalized recommendations.
*   **ALS Performance:** The ALS implementation, with default library parameters, showed modest performance, improving over Popularity but lagging significantly behind ItemCF. This could indicate that either the default hyperparameters are suboptimal for this specific dataset/implicit feedback type, or that basic latent factor models struggle to capture the specific co-interaction signals as effectively as ItemCF does in this context. Further tuning might improve ALS, but its initial performance was underwhelming compared to the simpler ItemCF.

## 5. Motivation for Advanced Models

The baseline evaluation highlights the effectiveness of collaborative filtering, particularly item-based methods, on this dataset. However, limitations remain:

1.  **Content Blindness:** ItemCF and ALS rely solely on the interaction matrix and do not utilize the available item content features (presentation length, VLE activity types). Incorporating these features could potentially improve recommendations, especially for less popular items or help differentiate between items with similar interaction patterns but different content structures.
2.  **Linearity Assumption (Implicit):** While effective, traditional CF methods like ItemCF (cosine similarity) and ALS (dot product of factors) primarily model linear relationships. Neural network approaches might capture more complex, non-linear user-item interaction patterns.

Therefore, the next logical steps involve exploring models designed to address these points:

*   **Neural Collaborative Filtering (NCF):** To investigate if deep learning models can capture non-linear interaction patterns more effectively than ALS or ItemCF.
*   **Hybrid NCF:** To explicitly combine the strengths of collaborative filtering with item content features using a neural architecture, potentially leading to more nuanced and robust recommendations.

These advanced models will be implemented and evaluated using the same rigorous time-based split protocol to allow for direct comparison against the established ItemCF benchmark.