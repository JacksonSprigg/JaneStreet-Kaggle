import os
import wandb

from src.data_loading import DataLoader
from src.models.lgbm import JaneStreetLGBM
from src.models.mlp import JaneStreetMLP
from config import EXPERIMENT, AVAILABLE_DATA, LGBM_PARAMS, MLP_PARAMS, BASE_DIR

def init_wandb():
    """Initialize Weights & Biases logging"""
    wandb.require("core")
    config = LGBM_PARAMS if EXPERIMENT['model_type'] == 'lgbm' else MLP_PARAMS
    return wandb.init(
        project=EXPERIMENT['wandb_project'],
        name=EXPERIMENT['wandb_run_name'],
        config=config,
        dir=os.path.join(BASE_DIR, "trained_models/wandb")
    )

def get_model():
    """Factory function to get the appropriate model based on config"""
    if EXPERIMENT['model_type'] == 'lgbm':
        return JaneStreetLGBM()
    elif EXPERIMENT['model_type'] == 'mlp':
        return JaneStreetMLP()
    else:
        raise ValueError(f"Unknown model type: {EXPERIMENT['model_type']}")

def main():
    # Initialize wandb
    init_wandb()
    
    # Load data using configured data source
    data_loader = DataLoader(AVAILABLE_DATA[EXPERIMENT['data_source']])
    
    # Get appropriate model
    model = get_model()
    
    if EXPERIMENT['use_cv']:
        data_loader.load_and_prepare_data()  # Just loads base DataFrame
        model.train_cv(data_loader)
    else:
        # Original single-split code
        X_train, X_val, y_train, y_val, w_train, w_val = data_loader.load_and_prepare_data()
        model.train(X_train, y_train, w_train, X_val, y_val, w_val, data_loader)
    
    wandb.finish()

if __name__ == "__main__":
    main()