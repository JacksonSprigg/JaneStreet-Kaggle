import numpy as np

def r2_score_weighted(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray) -> float:
    
    # R² = 1 - (∑wi(yi-yi^)²) / (∑wiy²i)

    numerator = np.sum(sample_weight * (y_true - y_pred) ** 2)    # ∑wi(yi-yi^)²
    denominator = np.sum(sample_weight * y_true ** 2)             # ∑wiy²i
    
    # Add small constant to prevent division by zero
    denominator = denominator + 1e-38
    
    r2 = 1 - numerator / denominator                             # 1 - (∑wi(yi-yi^)²)/(∑wiy²i)
    return r2
