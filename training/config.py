from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class Config:
    # Wandb Settings
    PROJECT_NAME: str = "jane-street-market"
    RUN_NAME: str = "lgbm_baseline_v1"
    WANDB_SAVE_DIR: str = "trained_models/wandb"
    
    # Data Settings
    TARGET: str = "responder_6"
    OFFLINE_START_DATE: int = 500
    SPLIT_DATE_ID: int = 1649  # New parameter for date-based split
    
    # Model Settings
    RANDOM_STATE: int = 42
    LGBM_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "regression_l2",
        "n_estimators": 1000,
        "max_depth": 10,
        "learning_rate": 0.03,
        "colsample_bytree": 0.55,
        "subsample": 0.80,
        "reg_lambda": 1.25,
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