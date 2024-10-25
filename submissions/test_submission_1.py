import os
import pandas as pd
import polars as pl
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

### Environment Setup ###
IS_KAGGLE = False  # Flag to switch between environments

# Setup paths based on environment
if IS_KAGGLE:
    data_path = '/kaggle/input/jane-street-real-time-market-data-forecasting'
else:
    # For local testing
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "jane-street-real-time-market-data-forecasting"))
    data_path = os.path.join(os.path.dirname(os.getcwd()), "jane-street-real-time-market-data-forecasting")

import kaggle_evaluation.jane_street_inference_server

# Global variable to store lags
lags_ : pl.DataFrame | None = None

def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    global lags_
    
    logger.info(f"Received test data with shape: {test.shape}")
    if lags is not None:
        lags_ = lags
        logger.info(f"Received lags data with shape: {lags_.shape}")
        
    # Simple prediction
    predictions = test.select(
        'row_id',
        pl.lit(0.0).alias('responder_6'),
    )    
    
    logger.info(f"Generated predictions with shape: {predictions.shape}")
    
    # Log score if we have actual values
    if 'responder_6' in test.columns and 'weight' in test.columns:
        weights = test['weight'].to_numpy()
        y_true = test['responder_6'].to_numpy()
        y_pred = predictions['responder_6'].to_numpy()
        
        # Calculate weighted R2
        numerator = np.sum(weights * (y_true - y_pred) ** 2)
        denominator = np.sum(weights * y_true ** 2)
        r2 = 1 - numerator / denominator
        logger.info(f"Batch weighted R2 score: {r2:.4f}")

    # The predict function must return a DataFrame
    assert isinstance(predictions, pl.DataFrame | pd.DataFrame)
    # with columns 'row_id', 'responer_6'
    assert predictions.columns == ['row_id', 'responder_6']
    # and as many rows as the test data.
    assert len(predictions) == len(test)

    return predictions

# Set up inference server
inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)

# Run based on environment
if os.getenv('KAGGLE_IS_COMPETITION_RERUN') or IS_KAGGLE:
    inference_server.serve()
else:
    logger.info("Starting local gateway...")
    inference_server.run_local_gateway(
        (
            os.path.join(data_path, "test.parquet"),
            os.path.join(data_path, "lags.parquet"),
        )
    )