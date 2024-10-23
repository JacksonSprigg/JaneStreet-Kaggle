import os
import polars as pl
import numpy as np
import pandas as pd
import wandb

from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from typing import Tuple

# Configuration
class Config:
    PROJECT_NAME = "jane-street-market"
    RUN_NAME = "lgbm_baseline_v1"
    
    TARGET = "responder_6"
    OFFLINE_START_DATE = 500
    RANDOM_STATE = 42
    
    # LightGBM Parameters
    LGBM_PARAMS = {
        "device": "gpu",
        "objective": "regression_l2",
        "n_estimators": 1000,
        "max_depth": 10,
        "learning_rate": 0.03,
        "colsample_bytree": 0.55,
        "subsample": 0.80,
        "random_state": RANDOM_STATE,
        "reg_lambda": 1.25,
        "reg_alpha": 0.001,
        "verbosity": -1,
    }

def load_and_prepare_data(data_path: str, offline_start_date: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """
    Load and prepare the training data, splitting into train and validation sets
    """
    # Initialize wandb run
    wandb.init(
        project=Config.PROJECT_NAME,
        name=Config.RUN_NAME,
        config=Config.LGBM_PARAMS,
        tags=["baseline", "lightgbm", "offline-training"]
    )
    
    # Log data loading start
    wandb.log({"data_loading_start": True})
    
    # Load training data
    train = pl.scan_parquet(os.path.join(data_path, "train.parquet")).\
        select(
            pl.int_range(pl.len(), dtype=pl.UInt64).alias("id"),
            pl.all(),
        )
    
    # Get columns to use
    all_cols = train.collect_schema().names()
    feature_cols = [c for c in all_cols if c.startswith('feature_')]
    exclude_cols = ['id', 'date_id', 'time_id', 'partition_id'] + \
                  [c for c in all_cols if 'responder' in c and c != Config.TARGET]
    
    # Log feature information
    wandb.config.update({
        "num_features": len(feature_cols),
        "target": Config.TARGET,
        "offline_start_date": offline_start_date
    })
    
    # Prepare training data
    train_data = train.filter(pl.col("date_id") > offline_start_date)
    train_df = train_data.collect().to_pandas()
    
    # Split features and target
    X = train_df.drop(exclude_cols + [Config.TARGET], axis=1)
    y = train_df[Config.TARGET]
    weights = train_df['weight'].values
    
    # Split into train/validation based on the last portion of data
    train_size = int(len(X) * 0.8)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    w_train = weights[:train_size]
    
    X_val = X[train_size:]
    y_val = y[train_size:]
    w_val = weights[train_size:]
    
    # Log dataset sizes
    wandb.log({
        "train_size": len(X_train),
        "val_size": len(X_val),
        "data_loading_complete": True
    })
    
    return X_train, X_val, y_train, y_val, w_train, w_val

class CustomLGBMCallback:
    """Custom callback for LightGBM to log metrics to wandb"""
    def __init__(self, logging_interval=10):
        self.logging_interval = logging_interval
        
    def __call__(self, env):
        if env.iteration % self.logging_interval == 0:
            wandb.log({
                "iteration": env.iteration,
                "train_r2": env.evaluation_result_list[0][2],
                "val_r2": env.evaluation_result_list[1][2]
            })

def custom_r2_score(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """Calculate weighted R² score"""
    score = r2_score(y_true, y_pred, sample_weight=weights)
    wandb.log({"r2_score": score})
    return score

# Main execution
if __name__ == "__main__":
    try:
        # Setup paths
        data_path = os.path.join(os.path.dirname(os.getcwd()), 
                                "jane-street-real-time-market-data-forecasting")
        
        # Load and prepare data
        X_train, X_val, y_train, y_val, w_train, w_val = load_and_prepare_data(
            data_path, Config.OFFLINE_START_DATE
        )
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        
    except Exception as e:
        wandb.alert(
            title="Training Failed",
            text=f"Error occurred: {str(e)}"
        )
        raise e
    
    finally:
        wandb.finish()