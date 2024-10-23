import os
import joblib
import wandb
import time

from lightgbm import LGBMRegressor, early_stopping
from tqdm import tqdm

from config import config
from src.utils.metrics import r2_lgb_eval

class JaneStreetLGBM:
    def __init__(self):
        self.model = LGBMRegressor(
            **config.LGBM_PARAMS,
            disable_default_eval_metric=True
        )
    
    def train(self, X_train, y_train, w_train, X_val, y_val, w_val, callback):
        print("\n🚀 Starting LightGBM training...")
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        
        start_time = time.time()
        
        self.model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_sample_weight=[w_train, w_val],
            eval_metric=r2_lgb_eval,
            callbacks=[
                callback,
                early_stopping(
                    stopping_rounds=100,
                    verbose=True
                )
            ]
        )
        
        training_time = time.time() - start_time
        
        print(f"\n✨ Training completed in {training_time:.2f} seconds")
        print(f"Best iteration: {self.model.best_iteration_}")
        print(f"Best score: {self.model.best_score_}")

class CustomLGBMCallback:
    def __init__(self, logging_interval: int = 10):
        self.logging_interval = logging_interval
        self.start_time = time.time()
        self.best_score = float('-inf')  # For R², higher is better
        self.pbar = None
        
    def __call__(self, env):
        if self.pbar is None:
            self.pbar = tqdm(
                total=env.end_iteration,
                desc="Training LightGBM",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            
        self.pbar.update(1)
        
        if env.iteration % self.logging_interval == 0:
                
            train_r2 = env.evaluation_result_list[1][2]
            val_r2 = env.evaluation_result_list[3][2]
            
            if val_r2 > self.best_score:
                self.best_score = val_r2
                improved = "🔥"
            else:
                improved = "  "
            
            elapsed = time.time() - self.start_time
            
            self.pbar.set_postfix({
                'train_r2': f"{train_r2:.9f}",
                'val_r2': f"{val_r2:.9f}",
                'best': f"{self.best_score:.9f}"
            })
            
            wandb.log({
                "iteration": env.iteration,
                "train_r2": train_r2,
                "val_r2": val_r2,
                "best_val_r2": self.best_score,
                "elapsed_time": elapsed
            })
            
            print(f"\nIteration {env.iteration:4d} {improved} | "
                    f"Train R²: {train_r2:.9f} | "
                    f"Val R²: {val_r2:.9f} | "
                    f"Best: {self.best_score:.9f} | "
                    f"Time: {elapsed:.1f}s")
    
    def __del__(self):
        if self.pbar is not None:
            self.pbar.close()