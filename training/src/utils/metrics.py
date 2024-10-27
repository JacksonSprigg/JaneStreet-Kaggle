import numpy as np

def r2_score_weighted(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray) -> float:
    
    # R² = 1 - (∑wi(yi-yi^)²) / (∑wiy²i)

    print("\nR² Score Calculation Debug:")
    print(f"Array Shapes:")
    print(f"y_true: {y_true.shape}, y_pred: {y_pred.shape}, weights: {sample_weight.shape}")
    
    print("\nFirst 5 samples:")
    print("True      | Predicted | Weight")
    print("-" * 40)
    for i in range(5):
        print(f"{y_true[i]:9.6f} | {y_pred[i]:9.6f} | {sample_weight[i]:6.6f}")
    
    print("\nLast 5 samples:")
    print("True      | Predicted | Weight")
    print("-" * 40)
    for i in range(-5, 0):
        print(f"{y_true[i]:9.6f} | {y_pred[i]:9.6f} | {sample_weight[i]:6.6f}")
    
    print("\nArray Statistics:")
    print(f"y_true     - mean: {np.mean(y_true):9.6f}, std: {np.std(y_true):9.6f}")
    print(f"y_pred     - mean: {np.mean(y_pred):9.6f}, std: {np.std(y_pred):9.6f}")
    print(f"weights    - mean: {np.mean(sample_weight):9.6f}, std: {np.std(sample_weight):9.6f}")

    numerator = np.sum(sample_weight * (y_true - y_pred) ** 2)    # ∑wi(yi-yi^)²
    denominator = np.sum(sample_weight * y_true ** 2)             # ∑wiy²i
    
    # Add small constant to prevent division by zero
    denominator = denominator + 1e-38

    print("\nCalculation Components:")
    print(f"Sum of weighted squared errors: {numerator:9.6f}")
    print(f"Sum of weighted true squared  : {denominator:9.6f}")
    
    r2 = 1 - numerator / denominator                             # 1 - (∑wi(yi-yi^)²)/(∑wiy²i)

    print(f"\nFinal R² Score: {r2:9.6f}")
    return r2
