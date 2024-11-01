from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class Config:
    # Wandb Settings
    PROJECT_NAME: str = "jane-street-market"
    RUN_NAME: str = "lgbm_last_1000_days_val_20"
    WANDB_SAVE_DIR: str = "trained_models/wandb"
    
    # Data Settings
    TARGET: str = "responder_6"
    OFFLINE_START_DATE: int = (1698 - 1000)
    SPLIT_DATE_ID: int = (1698 - 20)  # Note that there are 1698 days
    
    # Model Settings
    RANDOM_STATE: int = 42
    LGBM_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "regression_l2",
        "n_estimators": 1000,
        "max_depth": 20,
        "learning_rate": 0.03,
        "colsample_bytree": 0.6,
        "subsample": 0.80,
        "reg_lambda": 1,
        "reg_alpha": 0.001,
        "verbosity": -1,
        #"n_jobs": -1  # Added to use all CPU cores

        # GPU settings
        "device": "gpu",
        # Use multiple GPUs
        "num_gpu": 2
    })

# Create a config instance
config = Config()