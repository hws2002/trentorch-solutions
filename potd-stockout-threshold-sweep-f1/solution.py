import numpy as np 

def compute_F1(TP, FP, FN):
    if TP + FP == 0 or TP + FN == 0:
        return -np.inf
    return 2 * TP / (2 * TP + FP + FN)
    
def best_f1_threshold(scores: list[float], y: list[int]) -> tuple[float, float]:
    """
    The score threshold that maximizes F1 (predict positive if score >= t).

    scores: n predicted scores. y: n true labels, each 0 or 1.

    For each distinct value in scores, using it as threshold t, compute
    F1(t) over the whole dataset. Return (best_threshold, best_f1).

    Ties for the best F1 are broken by the LARGER threshold. A threshold
    where precision or recall is undefined (a zero denominator) can never
    win. Sort once and sweep with running TP/FP counts, O(n log n), not
    a from-scratch recompute per threshold.
    """
    # TODO: sort by score descending, group equal scores together (they
    # share one threshold), and only update the best on a STRICT improvement.
    scores = np.array(scores)
    idx = np.argsort(scores)[::-1]
    scores = scores[idx]
    y = np.array(y)[idx]
    best_t = 0
    best_F1 = -1
    TP, FP, FN = 0, 0, np.sum(y==1)
    best_t = -1
    best_F1 = compute_F1(TP,FP,FN)
    i = 0
    while i < len(scores):
        t = scores[i]
        while i < len(scores) and scores[i] == t:
            TP += y[i] == 1
            FP += y[i] == 0
            FN -= y[i] == 1
            i+=1
        
        F1 = compute_F1(TP,FP,FN)
        if best_F1 < F1:
            best_F1 = F1
            best_t = t
    return best_t, best_F1
