# Final Report - OULAD Course Recommendation System

**Date:** April 20, 2025 (Update as needed)
**Author:** Mohit Bhimrajka

## 1. Abstract

This project focused on developing a course recommendation system for the Open University Learning Analytics Dataset (OULAD). The goal was to recommend relevant course presentations to students based on their historical interactions and course characteristics. Various recommendation techniques were explored, including popularity baselines, collaborative filtering (ItemCF, ALS), and neural network approaches (NCF, Hybrid NCF). Data was preprocessed to handle missing values, filter sparse interactions, and generate user/item features. Models were evaluated using a time-based split and standard ranking metrics (Precision@10, Recall@10, NDCG@10). The Item-based Collaborative Filtering (ItemCF) model demonstrated the strongest performance, particularly in recall and NDCG, suggesting that item co-interaction patterns are highly predictive in this dataset.

## 2. Introduction

Online learning platforms generate vast amounts of interaction data. Utilizing this data effectively to provide personalized recommendations can significantly enhance the student experience, potentially improving engagement and success rates. The Open University Learning Analytics Dataset (OULAD) provides a rich collection of anonymized data about students, courses, and their interactions within a Virtual Learning Environment (VLE).

This project aimed to leverage the OULAD data to build a course recommendation system. Specifically, the objective was to recommend course presentations (`code_module` + `code_presentation`) to students who might find them relevant, based on their past VLE interactions, assessment performance (implicitly), demographics, and the content characteristics of the courses themselves.

## 3. Methodology

### 3.1. Dataset

The OULAD dataset was used, comprising information from seven tables: `assessments`, `courses`, `studentAssessment`, `studentInfo`, `studentRegistration`, `studentVle`, and `vle`.

### 3.2. Data Preprocessing

A standardized preprocessing pipeline was implemented (`src/pipelines/run_preprocessing.py`):
1.  **Cleaning:** Handled missing data (imputation/categorization), corrected types, created unique `presentation_id`.
2.  **Filtering:** Removed VLE interactions outside student registration periods. Iteratively filtered users with < 5 interactions and items with < 5 interacting users to mitigate sparsity.
3.  **Aggregation:** Grouped filtered VLE interactions by `(id_student, presentation_id)` to calculate `total_clicks`, `interaction_days`, first/last interaction dates.
4.  **Implicit Feedback:** Defined the primary interaction score as `implicit_feedback = log1p(total_clicks)`.
5.  **Feature Engineering:** Generated separate feature tables for users (`users_final.parquet` - mapped demographics, credits, etc.) and items (`items_final.parquet` - presentation length, VLE activity proportions).
The final aggregated interaction dataset contained 28,466 interactions from 25,364 users across 22 unique items (presentations).

### 3.3. Models Implemented

The following models were implemented, inheriting from a common `BaseRecommender` class:

1.  **Popularity:** Recommends items based on the global sum of `implicit_feedback` scores in the training data. Non-personalized.
2.  **Item-based Collaborative Filtering (ItemCF):** Computes item-item similarity (cosine) based on the user-item interaction matrix. Predicts scores for a user based on the similarity of candidate items to items the user previously interacted with.
3.  **Alternating Least Squares (ALS):** Matrix factorization technique (from the `implicit` library) that decomposes the user-item matrix into latent user and item factors. Used default hyperparameters from the library wrapper (factors=50, regularization=0.05, iterations=25).
4.  **Neural Collaborative Filtering (NCF):** A neural network model combining Generalized Matrix Factorization (GMF) and a Multi-Layer Perceptron (MLP) path to learn user-item interaction patterns. Trained using BCEWithLogitsLoss and negative sampling. (Epochs=2 used in dev notebook run).
5.  **Hybrid NCF:** Extends NCF by incorporating item content features. An MLP (`ContentEncoder`) first encodes item features (length, VLE proportions) into an embedding, which is then concatenated with CF user/item embeddings before being passed through a final MLP for prediction. (Epochs=10 used in dev notebook run).

### 3.4. Evaluation Protocol

*   **Split:** A time-based split was used on the aggregated interactions (`interactions_final.parquet`). Interactions where `last_interaction_date <= 250` formed the training set, and later interactions formed the test set.
*   **Filtering:** Users and items present in the test set but not the training set were filtered out to simulate a realistic prediction scenario.
*   **Metrics:** Standard ranking metrics were calculated at K=10:
    *   Precision@10: Proportion of top-10 recommendations that are relevant.
    *   Recall@10: Proportion of all relevant items found in the top-10 recommendations.
    *   NDCG@10: Precision/Recall measure discounted by rank, accounting for the position of relevant items.
*   **Sampling:** For efficiency during development and pipeline runs, evaluation was typically performed using 100 randomly sampled negative items per user alongside their positive test items. (The Hybrid model evaluation used 20 negatives in the provided log).

## 4. Results & Discussion

The performance of the implemented models on the OULAD test set (K=10) is summarized below.

| Model                     | Precision@10 | Recall@10 | NDCG@10 | Notes                                     |
| :------------------------ | :----------- | :-------- | :------ | :---------------------------------------- |
| Popularity                | 0.0621       | 0.6156    | 0.2153  | Neg Samples=100                           |
| ItemCF                    | **0.0988**   | **0.9781**| **0.6153** | Neg Samples=100                           |
| ALS (f=50, it=25)         | 0.0685       | 0.6778    | 0.3844  | Neg Samples=100                           |
| NCF (e=2, default dims)   | 0.0707       | 0.7011    | 0.5855  | Neg Samples=100                           |
| Hybrid (e=10, default dims)| 0.0900       | 0.8912    | 0.4698  | Neg Samples=20 (*Note difference*)       |

**Discussion:**

*   **Best Performer:** ItemCF stands out as the best performing model, achieving the highest Precision, Recall, and NDCG. This strongly indicates that patterns of item co-interaction (which courses students tend to interact with together or sequentially in this aggregated view) are very predictive for this dataset. The extremely high recall suggests that if a student interacted with a relevant course in the test set, ItemCF was very likely to place it in the top 10 recommendations.
*   **Neural Models:** NCF achieved good NDCG, slightly below ItemCF, suggesting it captured useful interaction patterns. The Hybrid model, while having higher precision than NCF/ALS/Pop, had lower NDCG than ItemCF and NCF in this run (though evaluated with fewer negative samples, potentially inflating precision slightly and affecting NDCG comparison). The content features (VLE proportions, length) might not have added significant value beyond the CF signal or might require different encoding/integration. Training these models for more epochs or tuning hyperparameters could potentially improve their performance further.
*   **ALS:** The standard ALS implementation performed relatively poorly compared to ItemCF and NCF, suggesting basic matrix factorization might not capture the nuances as well, or requires more tuning.
*   **Popularity:** As expected, the non-personalized Popularity baseline performed worst, highlighting the need for personalization.

## 5. Challenges & Limitations

*   **Dataset Size & Sparsity:** While large initially, filtering resulted in a manageable but potentially less diverse interaction set.
*   **Limited Items:** Only 22 unique course presentations remained after filtering, limiting recommendation diversity.
*   **Implicit Feedback:** Using `log1p(clicks)` is a common heuristic, but doesn't fully capture complex engagement nuances.
*   **Cold Start:** The models evaluated cannot recommend to new users or recommend new items not seen in training.
*   **Evaluation Sampling:** Using negative sampling speeds up evaluation but provides an approximation of the true ranking performance across all items. The Hybrid model was evaluated with fewer samples, making direct comparison slightly less reliable.

## 6. Conclusion

This project successfully implemented and evaluated several recommendation algorithms on the OULAD dataset. Item-based Collaborative Filtering (ItemCF) emerged as the most effective model for predicting relevant course presentations based on aggregated student interaction data, demonstrating high recall and NDCG@10. Neural network models (NCF, Hybrid) showed promise but did not surpass the simpler ItemCF in this evaluation setup. The results suggest that collaborative patterns are the dominant signal in this processed dataset.

## 7. Future Work

*   **API & Frontend:** Develop a FastAPI backend and a Typescript frontend to demonstrate the recommender system interactively.
*   **Hyperparameter Tuning:** Systematically tune hyperparameters for ALS, NCF, and Hybrid models (e.g., using Optuna) to optimize performance.
*   **Raw Interaction Models:** Explore models that directly use the raw VLE interaction sequences (e.g., RNNs, Transformers) if loading raw data becomes feasible/desirable.
*   **Feature Engineering:** Experiment with more sophisticated user and item features.
*   **Cold-Start Handling:** Investigate strategies to address recommendations for new users/items.