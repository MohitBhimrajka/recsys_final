# src/evaluation/metrics.py

import numpy as np
import math

def precision_at_k(recommended_items: list, relevant_items: list, k: int) -> float:
    """
    Calculates Precision@K.

    Args:
        recommended_items (list): List of recommended item IDs, ordered by descending relevance.
        relevant_items (list): List of actual relevant item IDs (ground truth).
        k (int): The number of top recommendations to consider.

    Returns:
        float: Precision@K score.
    """
    if k <= 0:
        return 0.0
    if not recommended_items or not relevant_items:
        return 0.0

    # Consider only the top K recommendations
    top_k_recommendations = recommended_items[:k]

    # Convert relevant items to a set for faster lookup
    relevant_set = set(relevant_items)

    # Count how many of the top K recommendations are relevant
    hits = sum(1 for item_id in top_k_recommendations if item_id in relevant_set)

    # Precision = (Number of relevant items in top K) / K
    return hits / k

def recall_at_k(recommended_items: list, relevant_items: list, k: int) -> float:
    """
    Calculates Recall@K.

    Args:
        recommended_items (list): List of recommended item IDs, ordered by descending relevance.
        relevant_items (list): List of actual relevant item IDs (ground truth).
        k (int): The number of top recommendations to consider.

    Returns:
        float: Recall@K score.
    """
    if not relevant_items: # If there are no relevant items, recall is undefined or 0. Let's return 0.
        return 0.0
    if k <= 0 or not recommended_items:
        return 0.0

    # Consider only the top K recommendations
    top_k_recommendations = recommended_items[:k]

    # Convert relevant items to a set for faster lookup
    relevant_set = set(relevant_items)

    # Count how many of the top K recommendations are relevant
    hits = sum(1 for item_id in top_k_recommendations if item_id in relevant_set)

    # Recall = (Number of relevant items in top K) / (Total number of relevant items)
    return hits / len(relevant_set)


def dcg_at_k(recommended_items: list, relevant_items: list, k: int) -> float:
    """
    Calculates Discounted Cumulative Gain (DCG)@K.

    Args:
        recommended_items (list): List of recommended item IDs, ordered by descending relevance.
        relevant_items (list): List of actual relevant item IDs (ground truth).
        k (int): The number of top recommendations to consider.

    Returns:
        float: DCG@K score.
    """
    if k <= 0:
        return 0.0
    if not recommended_items or not relevant_items:
        return 0.0

    top_k_recommendations = recommended_items[:k]
    relevant_set = set(relevant_items)
    dcg = 0.0

    for i, item_id in enumerate(top_k_recommendations):
        if item_id in relevant_set:
            # Gain is 1 if relevant, 0 otherwise (binary relevance)
            gain = 1.0
            # Discount factor: log2(rank + 1), where rank starts from 1
            discount = math.log2(i + 2) # i starts from 0, so rank is i+1. log2(rank+1) = log2(i+2)
            dcg += gain / discount

    return dcg

def ndcg_at_k(recommended_items: list, relevant_items: list, k: int) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG)@K.

    Args:
        recommended_items (list): List of recommended item IDs, ordered by descending relevance.
        relevant_items (list): List of actual relevant item IDs (ground truth).
        k (int): The number of top recommendations to consider.

    Returns:
        float: NDCG@K score.
    """
    if not relevant_items:
        return 0.0 # If no relevant items, NDCG is 0

    # Calculate actual DCG
    actual_dcg = dcg_at_k(recommended_items, relevant_items, k)

    # Calculate Ideal DCG (IDCG)
    # Ideal ranking puts all relevant items at the top
    ideal_recommendations = sorted(relevant_items, key=lambda x: 1, reverse=True) # Creates a list of relevant items
    ideal_dcg = dcg_at_k(ideal_recommendations, relevant_items, k)

    if ideal_dcg == 0:
        # This happens if there are no relevant items within the top K of the *ideal* ranking,
        # which means either no relevant items exist at all, or k is smaller than the number of relevant items.
        # If actual_dcg is also 0, result is 0. If actual_dcg is > 0 (shouldn't happen if idcg=0), result is undefined, return 0.
        return 0.0
    else:
        return actual_dcg / ideal_dcg

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    recs = ['item_A', 'item_B', 'item_C', 'item_D', 'item_E', 'item_F', 'item_G']
    rels = ['item_C', 'item_A', 'item_F', 'item_X', 'item_Y'] # True relevant items
    k_val = 5

    p_at_k = precision_at_k(recs, rels, k_val)
    r_at_k = recall_at_k(recs, rels, k_val)
    ndcg_at_k_val = ndcg_at_k(recs, rels, k_val)

    print(f"Example Recommendations: {recs}")
    print(f"Example Relevant Items:  {rels}")
    print(f"K = {k_val}")
    print("-" * 20)
    print(f"Precision@{k_val}: {p_at_k:.4f}") # Expected: 3 relevant items (A,C,F) in top 5 recs -> 3/5 = 0.6
    print(f"Recall@{k_val}:    {r_at_k:.4f}") # Expected: 3 relevant items found / 5 total relevant items -> 3/5 = 0.6
    print(f"NDCG@{k_val}:      {ndcg_at_k_val:.4f}")

    # Example from Wikipedia NDCG page (simplified binary relevance)
    recs_wiki = [1, 2, 3, 4, 5] # Item IDs
    rels_wiki = [3, 2]          # Relevant Item IDs
    k_wiki = 5
    dcg_wiki = dcg_at_k(recs_wiki, rels_wiki, k_wiki) # 1/log2(2+1) + 1/log2(3+1) = 1/log2(3) + 1/log2(4) = 1/1.585 + 1/2 = 0.6309 + 0.5 = 1.1309
    ideal_recs_wiki = [3, 2, 1, 4, 5] # Ideal order
    idcg_wiki = dcg_at_k(ideal_recs_wiki, rels_wiki, k_wiki) # 1/log2(1+1) + 1/log2(2+1) = 1/log2(2) + 1/log2(3) = 1/1 + 1/1.585 = 1 + 0.6309 = 1.6309
    ndcg_wiki = ndcg_at_k(recs_wiki, rels_wiki, k_wiki) # DCG / IDCG = 1.1309 / 1.6309 = 0.6934
    print("\n--- Wikipedia Example ---")
    print(f"DCG@{k_wiki}:  {dcg_wiki:.4f}")
    print(f"IDCG@{k_wiki}: {idcg_wiki:.4f}")
    print(f"NDCG@{k_wiki}: {ndcg_wiki:.4f}")