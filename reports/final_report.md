# Final Report - OULAD Course Recommendation System

**Date:** April 20, 2025
**Author:** Mohit Bhimrajka

## 1. Abstract

This project developed and evaluated a course recommendation system for the Open University Learning Analytics Dataset (OULAD), aiming to personalize suggestions of course presentations (`code_module` + `code_presentation`). A comprehensive data preprocessing pipeline was established, including cleaning, interaction filtering based on registration and activity levels (resulting in 25k users and 22 items), and implicit feedback generation (`log1p(total_clicks)`). Five distinct recommendation models were implemented and compared: Popularity, Item-Based Collaborative Filtering (ItemCF), Alternating Least Squares (ALS), Neural Collaborative Filtering (NCF), and a Hybrid NCF model incorporating item content features. Models were evaluated using a rigorous time-based split (`last_interaction_date <= 250`) and standard ranking metrics (Precision@10, Recall@10, NDCG@10), employing negative sampling for efficiency. Results demonstrated that **Item-Based Collaborative Filtering (ItemCF) significantly outperformed all other models**, achieving the highest NDCG@10 (0.6153) and Recall@10 (0.9781), indicating that direct item co-interaction patterns are the most potent predictive signal within this processed dataset context.

## 2. Introduction

The proliferation of online learning platforms like those used by the Open University creates opportunities to leverage student interaction data for enhancing the learning experience. Personalized course recommendations can help students discover relevant materials, navigate course offerings effectively, and potentially improve engagement and completion rates. The OULAD dataset provides a rich, real-world source of anonymized student VLE interactions, demographics, and course information suitable for building such a system.

This project details the development process, from initial data exploration and preprocessing to the implementation and comparative evaluation of various recommendation algorithms. The primary goal was to predict and recommend relevant course presentations to individual students based on their historical engagement patterns and available user/item features.

## 3. Methodology

### 3.1. Dataset & Preprocessing

The project utilized the standard OULAD CSV files. A critical preprocessing pipeline (`src/pipelines/run_preprocessing.py`) was implemented:

1.  **Cleaning:** Addressed missing values (e.g., `imd_band` imputed), corrected types, created `presentation_id`.
2.  **Temporal Filtering:** VLE interactions were restricted to periods within active student registrations.
3.  **Sparsity Filtering:** Iteratively removed users with < 5 interactions and items (presentations) with < 5 unique interacting users. This was crucial for model stability but reduced the dataset significantly.
    *   **Final Size:** The process resulted in an aggregated interaction dataset (`interactions_final.parquet`) containing **28,466 interactions** from **25,364 unique users** across only **22 unique items (presentations)**.
4.  **Implicit Feedback:** The core interaction score (`implicit_feedback`) was calculated as `log1p(total_clicks)` from aggregated VLE interactions per student-presentation pair.
5.  **Feature Engineering:** Generated user demographic features (`users_final.parquet`) and item content features (`items_final.parquet`, including presentation length and VLE activity type proportions) for the filtered user/item sets.

### 3.2. Models Implemented

Five distinct recommendation models were implemented, inheriting from `src.models.base.BaseRecommender`:

1.  **Popularity:** Non-personalized baseline ranking items by the global sum of `implicit_feedback`.
2.  **ItemCF:** Standard Item-Based Collaborative Filtering using cosine similarity on the user-item `implicit_feedback` matrix.
3.  **ALS:** Matrix Factorization via `implicit.als.AlternatingLeastSquares` with `factors=50`, `regularization=0.05`, `iterations=25`.
4.  **NCF:** Neural Collaborative Filtering combining GMF and MLP pathways. Trained with `mf_dim=32`, `mlp_embedding_dim=32`, `mlp_layers=[64,32,16,8]`, `epochs=2`, `lr=0.001`, `batch_size=1024`, `num_negatives=4`.
5.  **Hybrid NCF:** Extended NCF incorporating item features via a `ContentEncoder` MLP (`hidden_dims=[32,16]`, `output_dim=16`). Combined CF and content embeddings fed into a final MLP (`layers=[64,32,16]`). Trained with `cf_embedding_dim=32`, `content_embedding_dim=16`, `epochs=10`, `lr=0.001`, `batch_size=512`, `num_negatives=4`.

*(Note: NCF/Hybrid hyperparameters based on development notebook runs; systematic tuning was outside the scope)*

### 3.3. Evaluation Protocol

*   **Split:** A strict time-based split was used. Training data included all interactions where `last_interaction_date <= 250`. Test data included interactions where `last_interaction_date > 250`.
*   **Filtering:** The test set was filtered to only include users and items present in the training set, simulating recommending known items to known users based on past data.
*   **Metrics:** Precision@10, Recall@10, and NDCG@10 were calculated.
*   **Sampling:** Due to the potentially large number of candidate items (~22), evaluation was performed using **negative sampling** for efficiency. For each user in the test set, scores were predicted for their actual positive interactions *plus* a specified number of randomly sampled negative items (items not interacted with by the user in train or test, but known to the model). The reported results used **`n_neg_samples=100`** for Pop, ItemCF, ALS, NCF, and **`n_neg_samples=20`** for the Hybrid model run shown in the notebook logs.

## 4. Results & Discussion

### 4.1. Comparative Performance (K=10)

| Model                     | `n_neg_samples` | Precision@10 | Recall@10 | NDCG@10 |
| :------------------------ | :-------------- | :----------- | :-------- | :------ |
| Popularity                | 100             | 0.0621       | 0.6156    | 0.2153  |
| ALS (f=50, it=25)         | 100             | 0.0685       | 0.6778    | 0.3844  |
| NCF (e=2, default dims)   | 100             | 0.0707       | 0.7011    | 0.5855  |
| Hybrid (e=10, default dims)| **20**          | 0.0900       | 0.8912    | 0.4698  |
| **ItemCF**                | **100**         | **0.0988**   | **0.9781**| **0.6153** |

*(Note: Hybrid model evaluated with fewer negative samples, potentially affecting direct metric comparison, especially NDCG).*

### 4.2. Discussion

*   **ItemCF Superiority:** ItemCF consistently demonstrated the best performance across all key metrics when compared using the same evaluation setup (n=100 negatives). Its near-perfect recall (0.9781) indicates an exceptional ability to retrieve relevant items within the top 10. Its leading NDCG (0.6153) shows it ranks these relevant items effectively. This strongly suggests that **direct item co-interaction patterns are the most dominant and predictive signal** within this filtered OULAD interaction data. The simplicity and effectiveness of cosine similarity on the implicit feedback matrix proved highly suitable.

*   **Neural Models Performance:**
    *   **NCF:** Showed decent performance, particularly in NDCG (0.5855), outperforming ALS and Popularity. This indicates its ability to capture valuable, potentially non-linear, interaction patterns beyond simple matrix factorization. However, it did not surpass ItemCF in this setup (potentially limited by only 2 training epochs in the dev run).
    *   **Hybrid NCF:** Achieved good precision and recall, notably higher than NCF. However, its NDCG (0.4698) was lower than both ItemCF and NCF. It's crucial to note this was evaluated with only **20 negative samples**. Fewer negatives make it easier to rank positive items highly, potentially inflating Precision/Recall but making NDCG less comparable to runs with 100 negatives. While the inclusion of content features seems to help identify relevant items (higher P/R than NCF), the ranking quality (NDCG) didn't improve over ItemCF/NCF in this specific run. This could be due to insufficient training (10 epochs), suboptimal feature encoding, or the limited predictive power of the engineered VLE features for this task compared to the strong CF signal.

*   **ALS & Popularity:** ALS, with default parameters, offered only marginal improvement over the non-personalized Popularity baseline. This highlights that basic matrix factorization might not be the best fit or requires significant tuning for this type of implicit data. Popularity served its purpose as the lowest benchmark.

*   **Impact of Small Item Space:** The preprocessing pipeline, while necessary for handling sparsity, resulted in only 22 unique course presentations. This severely limits the potential diversity and serendipity of recommendations. Models might be effectively learning patterns within a very constrained item set, which could inflate recall metrics (as finding the few relevant items from a small pool is easier).

*   **Impact of Negative Sampling:** Evaluating with 100 (or 20) negative samples provides a computationally feasible approximation of ranking performance. However, these metrics are likely higher than what would be achieved when ranking against the *entire* item catalog (all 21 other items). The relative performance *between* models evaluated with the *same* number of negative samples is still informative, but absolute values should be interpreted with caution. The difference in `n_neg_samples` for the Hybrid model makes direct NDCG comparison difficult.

## 5. Challenges & Limitations

*   **Data Sparsity & Filtering:** The initial dataset was large but sparse. Aggressive filtering (min 5 interactions) was necessary but drastically reduced the user/item pool, especially the number of unique items (22), limiting real-world applicability and recommendation diversity.
*   **Implicit Feedback Quality:** `log1p(total_clicks)` captures engagement intensity but lacks nuance. It doesn't differentiate between productive interaction, confusion, or brief browsing. This simplification might obscure more complex learning patterns.
*   **Cold Start:** None of the evaluated models inherently handle new users or new items (presentations) without retraining or specific fallback strategies (e.g., content-based or popularity for new users/items).
*   **Limited Item Features:** The engineered item features (length, VLE proportions) might not be sufficiently discriminative or were not optimally integrated by the Hybrid model. More granular content analysis could be beneficial.
*   **Evaluation Protocol:**
    *   *Single Time Split:* Performance is based on one specific train/test split point (day 250). Results might vary with different split points.
    *   *Negative Sampling:* Provides an efficient but approximate evaluation. Full ranking evaluation would be more definitive but computationally expensive. Inconsistent sample sizes (Hybrid vs. others) hinder direct comparison.
*   **Static Dataset:** The models were trained on a fixed snapshot. A production system would require periodic retraining to incorporate new data and adapt to evolving interaction patterns.

## 6. Conclusion

This project successfully developed and evaluated a range of recommendation models for the OULAD dataset. Despite the challenges of data sparsity and the limitations of implicit feedback, the **Item-Based Collaborative Filtering (ItemCF) model emerged as the most effective** approach within the constraints of this project, demonstrating superior performance in retrieving and ranking relevant course presentations based on user co-interaction patterns. While neural models like NCF and Hybrid NCF showed potential, they did not surpass the simpler, highly effective ItemCF in this specific evaluation context, possibly due to limited training, hyperparameter tuning, or the dominance of the collaborative signal over the engineered content features. The study underscores the importance of robust preprocessing and highlights that even relatively simple CF methods can be powerful when strong co-interaction patterns exist, but also emphasizes the limitations imposed by a small final item space.

## 7. Future Work

*   **Hyperparameter Optimization:** Systematically tune NCF and Hybrid models (learning rate, embedding dimensions, layer structures, dropout) using tools like Optuna or Ray Tune.
*   **Advanced Feature Engineering:** Explore more sophisticated item features (e.g., assessment types/difficulty, textual descriptions if available) and user features (e.g., performance trajectories).
*   **Alternative Implicit Signals:** Experiment with different implicit feedback formulations (e.g., incorporating interaction duration, distinct days, assessment scores).
*   **Cold-Start Strategies:** Implement fallback mechanisms (e.g., content-based recommendations using item features for new users, popularity for new items) to address the cold-start problem.
*   **Sequence-Aware Models:** If processing power allows, explore models that leverage the *temporal sequence* of VLE interactions (e.g., RNNs, Transformers like SASRec) on less aggregated data, which might capture learning pathways more effectively.
*   **Full Ranking Evaluation:** If feasible, perform evaluation by ranking against all candidate items (not just sampled negatives) for a more accurate assessment of top-K performance.
*   **API/Frontend Enhancement:** Fully integrate other trained models (selectable) into the API and frontend for direct comparison in the demo.