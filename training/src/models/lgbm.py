import os
import json
import time
from datetime import datetime
from tqdm import tqdm
import wandb
import numpy as np
from typing import Optional, Dict, Any

from lightgbm import LGBMRegressor, Booster

from src.utils.metrics import r2_score_weighted

from config import EXPERIMENT, VALIDATION, LGBM_PARAMS, CV_SETTINGS


# TODO: CHange best model back

class ModelManager:
    def __init__(self, base_dir: str = 'trained_models'):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
    def save_model(self, 
                model: Booster, 
                metadata: Dict[str, Any],
                model_type: str = 'checkpoint',
                custom_name: Optional[str] = None) -> str:

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
            
            def convert_to_native(obj):
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(item) for item in obj]
                return obj
            
            # Convert metadata recursively
            converted_metadata = convert_to_native(metadata)
            
            # Add additional metadata
            converted_metadata.update({
                'saved_at': timestamp,
                'model_type': model_type,
                'lightgbm_version': convert_to_native(model.params.get('version', 'unknown'))
            })
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(converted_metadata, f, indent=2)
                
            print(f"✨ Saved {model_type} model to {model_path}")
            return model_path
            
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            raise

class JaneStreetLGBM:
    def __init__(self):
        self.model = LGBMRegressor(
            **LGBM_PARAMS,
            disable_default_eval_metric=True
        )
        self.model_manager = ModelManager()
        self.data_loader = None
        self.fold_models = []
        self.fold_callbacks = []     
        self.callbacks = {}  
        self.current_iteration = 0
        self.all_fold_scores = {}  # {iteration: {fold_idx: score}}
        self.best_weighted_cv_score = float('-inf')


    def train(self, X_train, y_train, w_train, X_val, y_val, w_val, data_loader):
        """Main training method that handles both single split and CV"""
        if EXPERIMENT['use_cv']:
            return self.train_cv([(X_train, X_val, y_train, y_val, w_train, w_val)], data_loader)
        else:
            return self.train_single(X_train, y_train, w_train, X_val, y_val, w_val, data_loader)
    
    def train_single(self, X_train, y_train, w_train, X_val, y_val, w_val, data_loader):
        """Train the model using weighted R² evaluation."""
        self.data_loader = data_loader  # Store the data_loader instance

        print("\n🚀 Starting LightGBM training...")
        start_time = time.time()

        # Keep track of training data length for reliable comparison
        train_length = len(y_train)
        def weighted_r2_eval(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
            # Use exact length match of training data to determine which set we're evaluating
            is_training = len(y_true) == train_length
            weights = w_train if is_training else w_val
            score = r2_score_weighted(y_true, y_pred, weights)
            return 'weighted_r2', score, True
        
        # Initialize callback
        callback = CustomLGBMCallback(self)
        
        # Always train with validation set for consistency
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric=weighted_r2_eval,
            callbacks=[callback]
        )
        
        # If skip_validation is True, we don't care about the validation score
        if VALIDATION.get('skip_validation', False):
            best_score = None
            best_iteration = self.model.n_estimators_
        else:
            best_score = callback.best_score
            best_iteration = self.model.best_iteration_
        
        training_time = time.time() - start_time
        print(f"\n✨ Training completed in {training_time:.2f} seconds")
        
        metadata = {
            'best_iteration': best_iteration,
            'best_score': best_score,
            'training_time': training_time,
            'model_params': self.model.get_params(),
            'wandb_run_id': wandb.run.id if wandb.run else None,
            'feature_importance': {
                f'feature_{i}': importance 
                for i, importance in enumerate(self.model.feature_importances_)
            },
            'experiment_config': {
                'experiment': EXPERIMENT,
                'validation': VALIDATION,
                'model_params': LGBM_PARAMS,
                'validation_details': data_loader.val_metadata 
            }
        }
        
        self.model_manager.save_model(
            self.model.booster_,
            metadata,
            model_type='final',
            custom_name=f"model_iter_{best_iteration}_{'valr2_' + str(best_score) if best_score is not None else 'last'}"
        )

    def _store_fold_scores(self, fold_idx, iteration, score):
        if iteration not in self.all_fold_scores:
            self.all_fold_scores[iteration] = {}
        self.all_fold_scores[iteration][fold_idx] = score

    def train_cv_fold(self, fold_idx: int, X_train, y_train, w_train, X_val, y_val, w_val):
        """Train a single CV fold"""
        print(f"\n🚀 Training CV fold {fold_idx + 1}/{len(CV_SETTINGS['windows'])}")
        
        model = LGBMRegressor(**LGBM_PARAMS, disable_default_eval_metric=True)
        train_length = len(y_train)
        
        def weighted_r2_eval(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
            is_training = len(y_true) == train_length
            weights = w_train if is_training else w_val
            score = r2_score_weighted(y_true, y_pred, weights)
            return 'weighted_r2', score, True
        
        callback = CustomLGBMCallback(self)
        callback.set_fold_info(fold_idx, CV_SETTINGS['windows'][fold_idx]['weight'])
        
        # Store the callback for accessing scores
        self.callbacks[fold_idx] = callback
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric=weighted_r2_eval,
            callbacks=[callback]
        )
        
        window_weight = CV_SETTINGS['windows'][fold_idx]['weight']
        weighted_score = callback.best_score * window_weight

        print(f"Fold {fold_idx + 1} | R² Score: {callback.best_score:.6f} | (weight: {window_weight:.4f}) | Weighted Score: {weighted_score:.6f}")
        
        return model, callback.best_score

    def train_cv(self, data_loader):
        """Train using CV folds"""
        self.data_loader = data_loader
        
        n_folds = len(CV_SETTINGS['windows'])
        
        for fold_idx in range(n_folds):
            # Get data for just this fold
            print(f"\n🚀 Loading and training fold {fold_idx + 1}/{n_folds}")
            X_train, X_val, y_train, y_val, w_train, w_val = data_loader.get_fold_data(fold_idx)
            
            # Train fold
            model, score = self.train_cv_fold(
                fold_idx, X_train, y_train, w_train, X_val, y_val, w_val
            )
            
            # Clear fold data from memory
            del X_train, X_val, y_train, y_val, w_train, w_val


class CustomLGBMCallback:
    def __init__(self, model: JaneStreetLGBM, logging_interval: int = LGBM_PARAMS['logging_interval']):
        self.logging_interval = logging_interval
        self.start_time = time.time()
        self.best_score = float('-inf')
        self.pbar = None
        self.model = model
        self.current_fold = None
        self.fold_weight = None
        
    def set_fold_info(self, fold_idx: int, weight: float):
        """Set current fold information"""
        self.current_fold = fold_idx
        self.fold_weight = weight
        
    def __call__(self, env):
        if self.pbar is None:
            self.pbar = tqdm(
                total=env.end_iteration,
                desc="Training LightGBM",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            
        self.pbar.update(1)
        

        if env.iteration % self.logging_interval == 0:
            # Always get both training and validation results since we're always passing validation data
            train_r2 = env.evaluation_result_list[1][2]
            val_r2 = env.evaluation_result_list[3][2]
                    
            # Only track improvements and save checkpoints if we're not skipping validation
            if not VALIDATION.get('skip_validation', False):
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
            else:
                improved = "  "  # No improvements tracked when skipping validation
            
            elapsed = time.time() - self.start_time
            
            # Update progress bar based on available metrics
            postfix = {'train_r2': f"{train_r2:.9f}"}
            if val_r2 is not None:
                postfix.update({
                    'val_r2': f"{val_r2:.9f}",
                    'best': f"{self.best_score:.9f}"
                })
            self.pbar.set_postfix(postfix)

            # Separate logging for CV and non-CV
            if not EXPERIMENT['use_cv']:
                # If validation is skipped, only log training metrics
                if VALIDATION.get('skip_validation', False):
                    wandb.log({
                        "iteration": env.iteration,
                        "train_r2": train_r2,
                        "elapsed_time": elapsed
                    })
                else:
                    # Existing logging for validation runs
                    wandb.log({
                        "iteration": env.iteration,
                        "train_r2": train_r2,
                        "val_r2": val_r2,
                        "best_val_r2": self.best_score,
                        "elapsed_time": elapsed
                    })
            else:
                # CV-specific logging remains unchanged
                self.model._store_fold_scores(
                    self.current_fold, 
                    env.iteration, 
                    val_r2
                )
                
                if self.current_fold == len(CV_SETTINGS['windows']) - 1:
                    all_fold_scores = self.model.all_fold_scores.get(env.iteration, {})
                    if len(all_fold_scores) == len(CV_SETTINGS['windows']):
                        total_weighted_score = 0
                        for fold_idx, score in all_fold_scores.items():
                            fold_weight = CV_SETTINGS['windows'][fold_idx]['weight']
                            total_weighted_score += score * fold_weight
                        
                        if total_weighted_score > self.model.best_weighted_cv_score:
                            self.model.best_weighted_cv_score = total_weighted_score
                        
                        wandb.log({
                            "iteration": env.iteration,
                            "weighted_cv_score": total_weighted_score,
                            "best_weighted_cv_score": self.model.best_weighted_cv_score
                        })
            
            # Print status based on available metrics
            status = f"\nIteration {env.iteration:4d} {improved} | Train R²: {train_r2:.9f}"
            if val_r2 is not None:
                status += f" | Val R²: {val_r2:.9f} | Best: {self.best_score:.9f}"
            status += f" | Time: {elapsed:.1f}s"
            print(status)