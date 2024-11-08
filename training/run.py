import os
import wandb

from src.data_loading import DataLoader
from src.models.lgbm import JaneStreetLGBM
from config import EXPERIMENT, AVAILABLE_DATA, LGBM_PARAMS, BASE_DIR

def init_wandb():
    """Initialize Weights & Biases logging"""
    wandb.require("core")  # Fixes a bug with wandb maybe
    return wandb.init(
        project=EXPERIMENT['wandb_project'],
        name=EXPERIMENT['wandb_run_name'],
        tags=EXPERIMENT['wandb_tags'],
        config=LGBM_PARAMS,
        dir=os.path.join(BASE_DIR, "trained_models/wandb")
    )

def main():
    # Initialize wandb
    init_wandb()
    
    # Load data using configured data source
    data_loader = DataLoader(AVAILABLE_DATA[EXPERIMENT['data_source']])
    X_train, X_val, y_train, y_val, w_train, w_val = data_loader.load_and_prepare_data()
    
    # Initialize model with configured parameters
    model = JaneStreetLGBM()
    
    # Train model
    model.train(
        X_train, y_train, w_train,
        X_val, y_val, w_val,
        data_loader
    )
    

    wandb.finish()

if __name__ == "__main__":
    main()