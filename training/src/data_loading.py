import os
import wandb
import polars as pl
import pandas as pd
import numpy as np

from typing import Tuple, List
from config import Config
from tqdm import tqdm
from time import time

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def get_feature_columns(self, df: pl.LazyFrame) -> List[str]:
        print("\n📊 Analyzing column structure...")
        all_cols = df.collect_schema().names()
        feature_cols = [c for c in all_cols if c.startswith('feature_')]
        exclude_cols = ['id', 'date_id', 'time_id', 'partition_id'] + \
                      [c for c in all_cols if 'responder' in c and c != Config.TARGET]
        print(f"Found {len(feature_cols)} features and {len(exclude_cols)} columns to exclude")
        return feature_cols, exclude_cols
    
    def handle_nulls(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Strategy:
        1. Create null flags for ALL features (not just ones with nulls)
        2. Forward fill within each symbol_id group
        3. Backward fill remaining nulls (handles first rows)
        """
        print("\n🔍 Processing null values...")
        
        # Store original shape for validation
        original_shape = df.shape
        
        # Create null flags for ALL features
        null_features = []
        for col in tqdm(feature_cols, desc="Creating null flags"):
            # Create flag column regardless of null presence
            flag_col = f'{col}_is_null'
            df[flag_col] = df[col].isnull().astype(np.int8)
            
            # Still track statistics for logging
            null_count = df[col].isnull().sum()
            if null_count > 0:
                null_features.append(col)
                print(f"  {col}: {null_count:,} nulls ({(null_count/len(df))*100:.2f}%)")
        
        # First forward fill within each symbol_id group
        print("\n📈 Forward filling values within symbol groups...")
        ffill_start = time()
        df[feature_cols] = df.groupby('symbol_id')[feature_cols].ffill()
        ffill_time = time() - ffill_start
        print(f"Forward fill completed in {ffill_time:.2f} seconds")
        
        # Handle remaining nulls (first rows) with backward fill
        remaining_nulls = df[feature_cols].isnull().sum()
        if remaining_nulls.any():
            print("\n⚠️ Backward filling remaining nulls (first rows)...")
            bfill_start = time()
            df[feature_cols] = df.groupby('symbol_id')[feature_cols].bfill()
            bfill_time = time() - bfill_start
            print(f"Backward fill completed in {bfill_time:.2f} seconds")
            
            # Check if any nulls still remain (this would happen if entire column is null for a symbol)
            final_nulls = df[feature_cols].isnull().sum()
            if final_nulls.any():
                print("\n⚠️ Warning: Some columns still have nulls after forward and backward fill.")
                print("These are likely entire null columns for some symbols.")
                # Fill these with 0 or another appropriate value
                zero_fill_start = time()
                df[feature_cols] = df[feature_cols].fillna(0)
                zero_fill_time = time() - zero_fill_start
                print(f"Zero fill completed in {zero_fill_time:.2f} seconds")
        
        # Validate processing
        assert df[feature_cols].isnull().sum().sum() == 0, "Found remaining nulls after processing"
        assert df.shape[0] == original_shape[0], "Row count changed during processing"
        
        return df
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
        print("\n🚀 Starting data loading process...")
        start_time = time()
        
        wandb.log({"data_loading_start": True})
        
        print("Reading parquet file...")
        train = pl.scan_parquet(os.path.join(self.data_path, "train.parquet")).\
            select(
                pl.int_range(pl.len(), dtype=pl.UInt64).alias("id"),
                pl.all(),
            )
        
        feature_cols, exclude_cols = self.get_feature_columns(train)
        
        # Log feature information
        wandb.config.update({
            "num_features": len(feature_cols),
            "target": Config.TARGET,
            "offline_start_date": Config.OFFLINE_START_DATE
        })
        
        print(f"\nFiltering data after date {Config.OFFLINE_START_DATE}...")
        train_data = train.filter(pl.col("date_id") > Config.OFFLINE_START_DATE)
        
        print("Converting to pandas DataFrame...")
        with tqdm(total=1, desc="Loading data") as pbar:
            train_df = train_data.collect().to_pandas()
            pbar.update(1)
        
        print(f"\nLoaded DataFrame shape: {train_df.shape}")
        
        # Handle null values before splitting
        train_df = self.handle_nulls(train_df, feature_cols)
          
        print("\nSplitting data into train/validation sets...")
        result = self.split_data(train_df, exclude_cols)
        
        end_time = time()
        print(f"\nData loading completed in {end_time - start_time:.2f} seconds")
        print(f"Train set shape: {result[0].shape}")
        print(f"Validation set shape: {result[1].shape}\n")
        
        return result

    def split_data(self, df: pd.DataFrame, exclude_cols: List[str]) -> Tuple:
        # Get unique dates before splitting to understand our ranges
        all_dates = df['date_id'].unique()
        
        # Use the configured split date
        split_date = Config.SPLIT_DATE_ID
        
        # Get the data split
        train_data = df[df['date_id'] < split_date]
        val_data = df[df['date_id'] >= split_date]
        
        # Calculate train size for array splitting
        train_size = train_data.shape[0]
        
        print("\n📅 Date ranges for train/validation splits:")
        print(f"Train dates     : {min(all_dates)} to {split_date-1}")
        print(f"Validation dates: {split_date} to {max(all_dates)}")
        print(f"Total unique dates: {len(all_dates)}")
        
        # Calculate some basic statistics
        train_dates = train_data['date_id'].unique()
        val_dates = val_data['date_id'].unique()
        print(f"\nData ranges:")
        print(f"Training  : {len(train_dates)} days")
        print(f"           date_id range: {train_data['date_id'].min()} to {train_data['date_id'].max()}")
        print(f"Validation: {len(val_dates)} days")
        print(f"           date_id range: {val_data['date_id'].min()} to {val_data['date_id'].max()}")
        
        # Split the data
        X = df.drop(exclude_cols + [Config.TARGET], axis=1).to_numpy()
        y = df[Config.TARGET].to_numpy()
        weights = df['weight'].to_numpy()
        
        return (
            X[:train_size], X[train_size:],
            y[:train_size], y[train_size:],
            weights[:train_size], weights[train_size:]
        )