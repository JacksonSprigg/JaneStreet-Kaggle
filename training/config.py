import os

#TODO: Remember to change the name every time. You forget things if they aren't labelled.

# Experiment settings
EXPERIMENT = {
    # Which dataset to use
    'data_source': 'null_flags_and_forward_fill',  # Options: 'raw' or 'null_flags_and_forward_fill'
    
    # Wandb settings
    'wandb_run_name': "lgbm_first_1500_last_50",
    'wandb_tags': ['lightgbm', 'gpu'],
    'wandb_project': "jane-street-market", # "jane-street-market" is the original
    
    # Core data settings
    'target': "responder_6",
    'start_after_day': -1,  # Skip first X days of data (-1 uses all data)
     
    # Validation strategy
    'split_type': 'time',  # Options: 'time', 'random', 'consecutive'
    
    # Training settings
    'use_gpu': True,
    'num_gpus': 2,
}

# Validation settings - only relevant settings for each type are used
# Note that there are a total of 1699 days, indexed from 0-1698
VALIDATION = {
    # Time-based validation
    'time': {
        'train_date_stop': 1500,   # Last date in training set
        'val_date_start': 1648,    # First date in validation set 
        'val_date_stop': 1698      # Last date in validation set
    },
    
    # Consecutive validation
    'consecutive': {
        'n_sequences': 5,         # Number of sequences to select
        'sequence_length': 28,    # Val Days per sequence
        'purge_before': 90,       # Days to purge before sequence
        'purge_after': 90,        # Days to purge after start (includes sequence)
    }
}

# Model parameters
LGBM_PARAMS = {
    # Model structure
    "objective": "regression_l2",
    "n_estimators": 600,
    "max_depth": 20,
    "learning_rate": 0.03,
    
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
    "logging_interval": 10,
    "base_model_dir": "trained_models"
}

# Base paths
BASE_DIR = "/home/jsprigg/ys68/JaneStreet-Kaggle"
AVAILABLE_DATA = {
    'raw': os.path.join(BASE_DIR, "jane-street-real-time-market-data-forecasting/train.parquet"),
    'null_flags_and_forward_fill': os.path.join(BASE_DIR, "feature_enhanced_data/null_flags_and_forward_fill.parquet")
}