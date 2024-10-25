import os
import polars as pl
import pandas as pd
import wandb
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
        
        print("\nSplitting data into train/validation sets...")
        result = self._split_data(train_df, exclude_cols)
        
        end_time = time()
        print(f"\nData loading completed in {end_time - start_time:.2f} seconds")
        print(f"Train set shape: {result[0].shape}")
        print(f"Validation set shape: {result[1].shape}\n")
        
        return result
    
    def _split_data(self, df: pd.DataFrame, exclude_cols: List[str]) -> Tuple:
        X = df.drop(exclude_cols + [Config.TARGET], axis=1)
        y = df[Config.TARGET]
        weights = df['weight'].values
        
        train_size = int(len(X) * Config.TRAIN_VAL_SPLIT)
        
        return (
            X[:train_size], X[train_size:],
            y[:train_size], y[train_size:],
            weights[:train_size], weights[train_size:]
        )