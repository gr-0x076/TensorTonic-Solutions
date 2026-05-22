def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    top_k = recommended[:k]
    intersection = set(top_k) & set(relevant)

    precision = len(intersection) / k if k > 0 else 0
    recall = len(intersection) / len(relevant) if len(relevant) > 0 else 0

    return [precision, recall]

