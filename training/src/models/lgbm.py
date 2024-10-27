import os
import json
import time
from datetime import datetime
from dataclasses import asdict
from typing import Optional, Dict, Any

import wandb
import numpy as np
from lightgbm import LGBMRegressor, Booster, early_stopping
from tqdm import tqdm

from config import config
from src.utils.metrics import r2_score_weighted

class ModelManager:
    """Handles model saving, loading, and metadata management."""
    
    def __init__(self, base_dir: str = 'trained_models'):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
    def save_model(self, 
                  model: Booster, 
                  metadata: Dict[str, Any],
                  model_type: str = 'checkpoint',
                  custom_name: Optional[str] = None) -> str:
        """
        Save model and metadata using LightGBM's native format.
        
        Args:
            model: LightGBM model to save
            metadata: Dictionary of metadata to save with model
            model_type: Either 'checkpoint' or 'final'
            custom_name: Optional custom name for the model file
            
        Returns:
            Path where model was saved
        """
        # Create timestamp-based version
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_name = f"{custom_name}.txt"
            
        # Create paths
        os.makedirs(self.base_dir, exist_ok=True)
        
        model_path = os.path.join(self.base_dir, model_name)
        metadata_path = os.path.join(self.base_dir, f"{os.path.splitext(model_name)[0]}_metadata.json")
        
        try:
            # Save model using native LightGBM format
            model.save_model(model_path)
            
            # Add additional metadata
            metadata.update({
                'saved_at': timestamp,
                'model_type': model_type,
                'lightgbm_version': model.params.get('version', 'unknown')
            })
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"✨ Saved {model_type} model to {model_path}")
            return model_path
            
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            raise

    def load_model(self, model_path: str) -> tuple[Booster, Dict[str, Any]]:
        """
        Load a model and its metadata.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Tuple of (loaded model, metadata dictionary)
        """
        try:
            # Load model
            model = Booster(model_file=model_path)
            
            # Load metadata if it exists
            metadata_path = f"{os.path.splitext(model_path)[0]}_metadata.json"
            metadata = {}
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
            return model, metadata
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise

class JaneStreetLGBM:
    def __init__(self):
        self.model = LGBMRegressor(
            **config.LGBM_PARAMS,
            disable_default_eval_metric=True
        )
        self.model_manager = ModelManager()
    
    def train(self, X_train, y_train, w_train, X_val, y_val, w_val):
        """Train the model using weighted R² evaluation."""
        print("\n🚀 Starting LightGBM training...")
        start_time = time.time()


        
        # Keep track of training data length for reliable comparison
        train_length = len(y_train)

        def weighted_r2_eval(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
            # Use exact length match of training data to determine which set we're evaluating
            is_training = len(y_true) == train_length
            weights = w_train if is_training else w_val

            # In weighted_r2_eval:
            # if not is_training:  # If validation data
            #     print("\nTraining Validation - Last 20 predictions:")
            #     for i in range(20):
            #         idx = len(y_pred) - 20 + i  # Get last 20 indices
            #         print(f"Pred: {y_pred[idx]:.6f}, Weight: {weights[idx]:.6f}, True: {y_true[idx]:.6f}")

            #     print("\nValidation Summary:")
            #     print(f"Mean prediction: {np.mean(y_pred):.6f}")
            #     print(f"Mean weight: {np.mean(weights):.6f}")
            #     print(f"Mean true value: {np.mean(y_true):.6f}")

            score = r2_score_weighted(y_true, y_pred, weights)
            return 'weighted_r2', score, True
        
        # Initialize callback
        callback = CustomLGBMCallback(self)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric=weighted_r2_eval,
            callbacks=[
                callback,
                early_stopping(stopping_rounds=100, verbose=True)
            ]
        )
        
        training_time = time.time() - start_time
        print(f"\n✨ Training completed in {training_time:.2f} seconds")
        
        # Save final model with metadata
        metadata = {
            'best_iteration': self.model.best_iteration_,
            'best_score': self.model.best_score_,
            'training_time': training_time,
            'model_params': self.model.get_params(),
            'wandb_run_id': wandb.run.id if wandb.run else None,
            'feature_importance': dict(zip(
                X_train.columns,
                self.model.feature_importances_
            ))
        }
        
        self.model_manager.save_model(
            self.model.booster_,
            metadata,
            model_type='final',
            custom_name=f"model_iter_{self.model.best_iteration_}_valr2_{callback.best_score:.6f}"
        )

class CustomLGBMCallback:
    def __init__(self, model: JaneStreetLGBM, logging_interval: int = 10):
        self.logging_interval = logging_interval
        self.start_time = time.time()
        self.best_score = float('-inf')
        self.pbar = None
        self.model = model
        
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
                
                # Save checkpoint with metadata
                metadata = {
                    'iteration': env.iteration,
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'elapsed_time': time.time() - self.start_time,
                    'wandb_run_id': wandb.run.id if wandb.run else None
                }
                
                self.model.model_manager.save_model(
                    env.model,
                    metadata,
                    model_type='checkpoint',
                    custom_name=f"best"
                )
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