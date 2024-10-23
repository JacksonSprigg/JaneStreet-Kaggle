import numpy as np

def r2_score_weighted(y_true, y_pred, sample_weight):
    """Base R² calculation function"""
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    return r2

def r2_lgb_eval(y_true, y_pred):
    """LightGBM custom eval metric for sklearn API"""
    return 'r2', r2_score_weighted(y_true, y_pred, None), True