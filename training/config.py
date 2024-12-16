import os

EXPERIMENT = {
    # Which altered dataset to use
    'data_source': 'null_flags_and_forward_fill',  # Options: 'raw' or 'null_flags_and_forward_fill'
    
    # Model Selection
    'model_type': 'mlp',  # Options: 'lgbm' or 'mlp'
    
    # Wandb settings
    'wandb_run_name': "mlp_test_1",
    'wandb_project': "jane-street-market",
    
    # Core data settings
    'target': "responder_6",
    'start_after_day': -1, # -1 to start at day zero
    
    # Validation Strategy
    'use_cv': False,  # If True, uses CV_SETTINGS below. If False, uses VALIDATION settings
    
    # Training settings, Only relevant for LGBM
    'use_gpu': False,
    'num_gpus': 2,
}

# Used when EXPERIMENT['use_cv'] = True
CV_SETTINGS = {
    'windows': [
        {"start": 399, "end": 406, "weight": 0.0357},
        {"start": 624, "end": 632, "weight": 0.0405},
        {"start": 1216, "end": 1235, "weight": 0.1022},
        {"start": 1302, "end": 1316, "weight": 0.0943},
        {"start": 1422, "end": 1440, "weight": 0.1199},
        {"start": 1518, "end": 1564, "weight": 0.3206},
        {"start": 1638, "end": 1680, "weight": 0.2869}
    ]
}

# Used when EXPERIMENT['use_cv'] = False
VALIDATION = {
    # Time-based validation
    'time': {
        'train_date_stop': 1648,   # Last date in training set
        'val_date_start': 1648,    # First date in validation set 
        'val_date_stop': 1698      # Last date in validation set
    },
    'split_type': 'time',  # Options: 'time'
    'skip_validation': False  # Option to skip validation
}

# LightGBM parameters
LGBM_PARAMS = {
    # Model structure
    "objective": "regression_l2",
    "n_estimators": 10,
    "max_depth": 20,
    "learning_rate": 0.001,
    
    # Feature sampling
    "colsample_bytree": 0.6,
    "subsample": 0.80,
    
    # Regularization
    "reg_lambda": 1,
    "reg_alpha": 0.001,
    
    # Performance settings
    "verbosity": -1,
    "device": "gpu" if EXPERIMENT['use_gpu'] else "cpu",
    "num_gpu": EXPERIMENT['num_gpus'] if EXPERIMENT['use_gpu'] else None,

    # Training control
    "logging_interval": 5,
    "base_model_dir": "trained_models"
}

# MLP parameters
MLP_PARAMS = {
    # Architecture
    'hidden_dims': [192, 128, 64],  # Hidden layer dimensions
    'dropout': 0.05,
    
    # Training
    'batch_size': 4096, #4096
    'epochs': 10,
    'learning_rate': 0.005,
    'weight_decay': 0, # 1e-5?
    
    # Performance settings
    'num_workers': 6,
    
    # Paths
    'base_model_dir': "trained_models"
}

# Base paths
BASE_DIR = "/home/jsprigg/ys68/JaneStreet-Kaggle"
AVAILABLE_DATA = {
    'raw': os.path.join(BASE_DIR, "jane-street-real-time-market-data-forecasting/train.parquet"),
    'null_flags_and_forward_fill': os.path.join(BASE_DIR, "feature_enhanced_data/null_flags_and_forward_fill.parquet")
}