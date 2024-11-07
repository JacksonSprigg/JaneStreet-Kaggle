import wandb

from src.data_loading import DataLoader
from src.models.lgbm import JaneStreetLGBM
from src.utils.logging_utils import init_wandb

def main():

    # Initialize wandb
    init_wandb()
        
    # Setup data path
    data_path = "/home/jsprigg/ys68/JaneStreet-Kaggle/jane-street-real-time-market-data-forecasting"
    
    # Load data
    data_loader = DataLoader(data_path)
    X_train, X_val, y_train, y_val, w_train, w_val = data_loader.load_and_prepare_data()
    
    # Initialize and train model
    model = JaneStreetLGBM()
    
    model.train(
        X_train, y_train, w_train,
        X_val, y_val, w_val
    )
    
    wandb.finish()

if __name__ == "__main__":
    main()