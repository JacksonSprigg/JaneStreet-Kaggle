import os
import wandb

from src.data_loading import DataLoader
from src.models.lgbm import JaneStreetLGBM, CustomLGBMCallback
from src.utils.logging_utils import init_wandb, log_error

def main():
    # Initialize wandb
    init_wandb()
    try:
        
        # Setup data path
        data_path = os.path.join(os.path.dirname(os.getcwd()), 
                                "jane-street-real-time-market-data-forecasting")
        
        # Load data
        data_loader = DataLoader(data_path)
        X_train, X_val, y_train, y_val, w_train, w_val = data_loader.load_and_prepare_data()
        
        # Initialize and train model
        model = JaneStreetLGBM()
        callback = CustomLGBMCallback()
        
        model.train(
            X_train, y_train, w_train,
            X_val, y_val, w_val,
            callback
        )
        
    except Exception as e:
        log_error(e)
        raise e
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()