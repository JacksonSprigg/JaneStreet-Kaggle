import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
import numpy as np
import time
from tqdm import tqdm
from datetime import datetime
import os
import json
from typing import Optional, Dict, Any

from src.utils.metrics import r2_score_weighted_torch
from config import EXPERIMENT, VALIDATION, MLP_PARAMS, CV_SETTINGS

class MLPModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, MLP_PARAMS['hidden_dims'][0]),
            nn.ReLU(),
            nn.Dropout(MLP_PARAMS['dropout']),
            *[nn.Sequential(
                nn.Linear(MLP_PARAMS['hidden_dims'][i], MLP_PARAMS['hidden_dims'][i+1]),
                nn.ReLU(),
                nn.Dropout(MLP_PARAMS['dropout'])
            ) for i in range(len(MLP_PARAMS['hidden_dims'])-1)],
            nn.Linear(MLP_PARAMS['hidden_dims'][-1], 1)
        )
        
    def forward(self, x):
        return self.layers(x).squeeze()

class ModelManager:
    def __init__(self, base_dir: str = 'trained_models'):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
    def save_model(self, 
                model: nn.Module, 
                metadata: Dict[str, Any],
                model_type: str = 'checkpoint',
                custom_name: Optional[str] = None) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{custom_name}.pt"
        
        os.makedirs(self.base_dir, exist_ok=True)
        model_path = os.path.join(self.base_dir, model_name)
        metadata_path = os.path.join(self.base_dir, f"{os.path.splitext(model_name)[0]}_metadata.json")
        
        try:
            torch.save(model.state_dict(), model_path)
            
            def convert_to_native(obj):
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, torch.Tensor):
                    return obj.cpu().detach().numpy().tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(x) for x in obj]
                return obj
            
            converted_metadata = convert_to_native(metadata)
            converted_metadata.update({
                'saved_at': timestamp,
                'model_type': model_type,
                'torch_version': torch.__version__
            })
            
            with open(metadata_path, 'w') as f:
                json.dump(converted_metadata, f, indent=2)
                
            print(f"✨ Saved {model_type} model to {model_path}")
            return model_path
            
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            raise

class JaneStreetMLP:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None  # Will be initialized when we know input dimension
        self.model_manager = ModelManager()
        self.data_loader = None
        self.fold_models = []
        self.fold_callbacks = []
        self.callbacks = {}
        self.current_epoch = 0
        self.all_fold_scores = {}
        self.best_weighted_cv_score = float('-inf')

    def _initialize_model(self, input_dim: int):
        """Initialize model and move to appropriate device"""
        model = MLPModel(input_dim)
        return model.to(self.device)

    def _prepare_data_loaders(self, X_train, y_train, w_train, X_val, y_val, w_val):
        # Ensure arrays are writable
        X_train = np.array(X_train, copy=True)
        y_train = np.array(y_train, copy=True)
        w_train = np.array(w_train, copy=True)
        X_val = np.array(X_val, copy=True)
        y_val = np.array(y_val, copy=True)
        w_val = np.array(w_val, copy=True)
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        w_train = torch.FloatTensor(w_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        w_val = torch.FloatTensor(w_val).to(self.device)
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train, w_train)
        val_dataset = TensorDataset(X_val, y_val, w_val)
        
        # Create data loaders - no shuffling for time series data
        train_loader = DataLoader(
            train_dataset, 
            batch_size=MLP_PARAMS['batch_size'],
            shuffle=False,  # Changed to False for time series
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=MLP_PARAMS['batch_size'],
            shuffle=False,
            pin_memory=True
        )
        
        return train_loader, val_loader

    def train(self, X_train, y_train, w_train, X_val, y_val, w_val, data_loader):
        if EXPERIMENT['use_cv']:
            return self.train_cv([(X_train, X_val, y_train, y_val, w_train, w_val)], data_loader)
        else:
            return self.train_single(X_train, y_train, w_train, X_val, y_val, w_val, data_loader)
    
    def _evaluate(self, model, data_loader):
        model.eval()
        # Pre-allocate tensors on the correct device
        total_samples = len(data_loader.dataset)
        all_preds = torch.empty(total_samples, device=self.device)
        all_targets = torch.empty(total_samples, device=self.device)
        all_weights = torch.empty(total_samples, device=self.device)
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc="Evaluating", leave=False)
            current_idx = 0
            
            for X, y, w in pbar:
                X, y, w = X.to(self.device), y.to(self.device), w.to(self.device)
                pred = model(X)
                batch_size = X.size(0)
                
                # Store everything on GPU
                all_preds[current_idx:current_idx + batch_size] = pred
                all_targets[current_idx:current_idx + batch_size] = y
                all_weights[current_idx:current_idx + batch_size] = w
                
                current_idx += batch_size
                
        return r2_score_weighted_torch(all_targets, all_preds, all_weights)
        
    def train_single(self, X_train, y_train, w_train, X_val, y_val, w_val, data_loader):
        self.data_loader = data_loader
        print("\n🚀 Starting MLP training...")
        start_time = time.time()
        
        # Initialize model if not already done
        if self.model is None:
            self.model = self._initialize_model(X_train.shape[1])
        
        train_loader, val_loader = self._prepare_data_loaders(
            X_train, y_train, w_train, X_val, y_val, w_val
        )
        
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=MLP_PARAMS['learning_rate'],
            weight_decay=MLP_PARAMS['weight_decay']
        )
        
        best_score = float('-inf')
        
        for epoch in range(MLP_PARAMS['epochs']):
            self.model.train()
            epoch_loss = 0
            
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{MLP_PARAMS['epochs']}", 
                position=0
            )
            
            for X, y, w in pbar:
                X, y, w = X.to(self.device), y.to(self.device), w.to(self.device)
                
                optimizer.zero_grad()
                pred = self.model(X)
                loss = -r2_score_weighted_torch(y, pred, w)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            # Evaluate once per epoch
            train_r2 = -epoch_loss / len(train_loader)
            val_r2 = self._evaluate(self.model, val_loader)
            
            print(f"\nEpoch {epoch+1}/{MLP_PARAMS['epochs']} - train_r2: {train_r2:.6f}, val_r2: {val_r2:.6f}")
            
            # Log metrics - regardless of validation
            metrics = {
                "epoch": epoch,
                "train_r2": train_r2,
            }
            
            if not VALIDATION.get('skip_validation', False):
                metrics["val_r2"] = val_r2
            
            wandb.log(metrics)
            
            # Save model based on validation strategy
            if VALIDATION.get('skip_validation', False):
                if epoch == MLP_PARAMS['epochs'] - 1:  # Last epoch
                    metadata = {
                        'epoch': epoch,
                        'train_r2': train_r2,
                        'model_params': MLP_PARAMS,
                        'wandb_run_id': wandb.run.id if wandb.run else None,
                    }
                    
                    self.model_manager.save_model(
                        self.model,
                        metadata,
                        model_type='final',
                        custom_name=f"model_final_epoch_{epoch}"
                    )
            else:
                # Original validation-based saving
                if val_r2 > best_score:
                    best_score = val_r2
                    metadata = {
                        'epoch': epoch,
                        'train_r2': train_r2,
                        'val_r2': val_r2,
                        'model_params': MLP_PARAMS,
                        'wandb_run_id': wandb.run.id if wandb.run else None,
                    }
                    
                    self.model_manager.save_model(
                        self.model,
                        metadata,
                        model_type='checkpoint',
                        custom_name="best"
                    )

        training_time = time.time() - start_time
        print(f"\n✨ Training completed in {training_time:.2f} seconds")

    def train_cv_fold(self, fold_idx: int, X_train, y_train, w_train, X_val, y_val, w_val):
        print(f"\n🚀 Training CV fold {fold_idx + 1}/{len(CV_SETTINGS['windows'])}")
        
        model = MLPModel(X_train.shape[1]).to(self.device)
        train_loader, val_loader = self._prepare_data_loaders(
            X_train, y_train, w_train, X_val, y_val, w_val
        )
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=MLP_PARAMS['learning_rate'],
            weight_decay=MLP_PARAMS['weight_decay']
        )
        
        best_score = float('-inf')
        pbar = tqdm(range(MLP_PARAMS['epochs']), desc=f"Training Fold {fold_idx + 1}")
        
        for epoch in pbar:
            model.train()
            for X, y, w in train_loader:
                X, y, w = X.to(self.device), y.to(self.device), w.to(self.device)
                optimizer.zero_grad()
                pred = model(X)
                loss = -r2_score_weighted_torch(y, pred, w)
                loss.backward()
                optimizer.step()
            
            val_r2 = self._evaluate(model, val_loader)
            if val_r2 > best_score:
                best_score = val_r2
            
            window_weight = CV_SETTINGS['windows'][fold_idx]['weight']
            weighted_score = val_r2 * window_weight
            
            # Store score for this fold and epoch
            self._store_fold_scores(fold_idx, epoch, val_r2)
            
            # If this is the last fold, calculate and log weighted scores
            if fold_idx == len(CV_SETTINGS['windows']) - 1:
                all_fold_scores = self.all_fold_scores.get(epoch, {})
                if len(all_fold_scores) == len(CV_SETTINGS['windows']):
                    total_weighted_score = 0
                    for fold_idx, score in all_fold_scores.items():
                        fold_weight = CV_SETTINGS['windows'][fold_idx]['weight']
                        total_weighted_score += score * fold_weight
                    
                    if total_weighted_score > self.best_weighted_cv_score:
                        self.best_weighted_cv_score = total_weighted_score
                    
                    wandb.log({
                        "epoch": epoch,
                        "weighted_cv_score": total_weighted_score,
                        "best_weighted_cv_score": self.best_weighted_cv_score
                    })
            
            pbar.set_postfix({
                'val_r2': f'{val_r2:.6f}', 
                'weighted': f'{weighted_score:.6f}'
            })
        
        print(f"Fold {fold_idx + 1} | R² Score: {best_score:.6f} | (weight: {window_weight:.4f}) | Weighted Score: {weighted_score:.6f}")
        
        return model, best_score

    def _store_fold_scores(self, fold_idx, epoch, score):
        if epoch not in self.all_fold_scores:
            self.all_fold_scores[epoch] = {}
        self.all_fold_scores[epoch][fold_idx] = score

    def train_cv(self, data_loader):
        self.data_loader = data_loader
        n_folds = len(CV_SETTINGS['windows'])
        
        for fold_idx in range(n_folds):
            print(f"\n🚀 Loading and training fold {fold_idx + 1}/{n_folds}")
            X_train, X_val, y_train, y_val, w_train, w_val = data_loader.get_fold_data(fold_idx)
            
            model, score = self.train_cv_fold(
                fold_idx, X_train, y_train, w_train, X_val, y_val, w_val
            )
            
            # Clear fold data from memory
            del X_train, X_val, y_train, y_val, w_train, w_val