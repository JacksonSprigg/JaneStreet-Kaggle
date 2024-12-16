import os
import polars as pl
import numpy as np

from typing import Tuple, List
from config import EXPERIMENT, VALIDATION, CV_SETTINGS
from time import time

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.val_metadata = None
        
    def get_feature_columns(self, df: pl.LazyFrame) -> Tuple[list, list]:
        """Extract feature columns and columns to exclude"""
        print("\n📊 Analyzing column structure...")
        all_cols = df.collect_schema().names()
        feature_cols = [c for c in all_cols if c.startswith('feature_')]
        exclude_cols = ['id', 'date_id', 'time_id', 'partition_id'] + \
                      [c for c in all_cols if 'responder' in c and c != EXPERIMENT['target']]
        print(f"Found {len(feature_cols)} features and {len(exclude_cols)} columns to exclude")
        return feature_cols, exclude_cols
    
    def _prepare_arrays(self, df: pl.DataFrame, mask: pl.Series, exclude_cols: List[str]) -> Tuple:
        """Convert filtered Polars DataFrame to numpy arrays for model"""
        print(f"Original DataFrame size: {df.estimated_size()}")
        filtered = df.filter(mask)
        print(f"Filtered DataFrame size: {filtered.estimated_size()}")
        
        # Get features (everything not in exclude_cols or target)
        X = filtered.select([
            col for col in filtered.columns 
            if col not in exclude_cols + [EXPERIMENT['target']]
        ]).to_numpy()
        
        # Get target and weights
        y = filtered.select(EXPERIMENT['target']).to_numpy().ravel()
        weights = filtered.select('weight').to_numpy().ravel()
        
        return X, y, weights

    def split_data_time(self, df: pl.DataFrame, exclude_cols: List[str]) -> Tuple:
        """Temporal split with optional gap between train and validation"""
        print("\n📈 Performing temporal split...")
        
        # Get unique dates for logging
        all_dates = df.get_column('date_id').unique().sort()
        time_config = VALIDATION['time']
        
        # Create masks for splitting
        train_mask = pl.col('date_id') <= time_config['train_date_stop']
        val_mask = (pl.col('date_id') >= time_config['val_date_start']) & \
                (pl.col('date_id') <= time_config['val_date_stop'])
        
        # Get data splits
        train_dates = df.filter(train_mask).get_column('date_id').unique().sort()
        val_dates = df.filter(val_mask).get_column('date_id').unique().sort()
        
        print("\n📅 Date ranges for train/validation splits:")
        print(f"Train dates     : {min(train_dates)} to {max(train_dates)}")
        if time_config['val_date_start'] - time_config['train_date_stop'] > 1:
            print(f"Gap            : {time_config['train_date_stop'] + 1} to {time_config['val_date_start'] - 1}")
        print(f"Validation dates: {min(val_dates)} to {max(val_dates)}")
        print(f"Total unique dates: {len(all_dates)}")
        
        print(f"\nData ranges:")
        print(f"Training  : {len(train_dates)} days")
        print(f"           date_id range: {train_dates.min()} to {train_dates.max()}")
        print(f"Validation: {len(val_dates)} days")
        print(f"           date_id range: {val_dates.min()} to {val_dates.max()}")
        
        if time_config['val_date_start'] - time_config['train_date_stop'] > 1:
            print(f"Gap size : {time_config['val_date_start'] - time_config['train_date_stop'] - 1} days")
        
        # Store metadata
        self.val_metadata = {
            'train_dates': train_dates.to_list(),
            'validation_dates': val_dates.to_list(),
            'train_date_stop': time_config['train_date_stop'],
            'val_date_start': time_config['val_date_start'],
            'val_date_stop': time_config['val_date_stop']
        }
        
        # Convert to numpy arrays
        X_train, y_train, w_train = self._prepare_arrays(df, train_mask, exclude_cols)
        X_val, y_val, w_val = self._prepare_arrays(df, val_mask, exclude_cols)
        
        return X_train, X_val, y_train, y_val, w_train, w_val
    
    def split_data_cv(self, df: pl.DataFrame, exclude_cols: List[str]) -> int:
        """
        Instead of returning all fold data, just returns number of folds
        and stores df and exclude_cols for later use
        """
        self.cv_df = df  # Store DataFrame for later use
        self.cv_exclude_cols = exclude_cols
        
        print("\nCV Fold Information:")
        for fold_idx, window in enumerate(CV_SETTINGS['windows']):
            print(f"\nFold {fold_idx + 1}")
            print(f"Validation window: Days {window['start']}-{window['end']} (weight: {window['weight']:.4f})")
            print(f"Training: Days 0-{window['start']-1} ({window['start']} days)")
            print(f"Validation: Days {window['start']}-{window['end']} ({window['end'] - window['start'] + 1} days)")
        
        return len(CV_SETTINGS['windows'])

    def get_fold_data(self, fold_idx: int) -> Tuple:
        """Get data for a specific fold"""
        window = CV_SETTINGS['windows'][fold_idx]
        
        train_mask = pl.col('date_id') < window['start']
        val_mask = (pl.col('date_id') >= window['start']) & (pl.col('date_id') <= window['end'])
        
        X_train, y_train, w_train = self._prepare_arrays(self.cv_df, train_mask, self.cv_exclude_cols)
        X_val, y_val, w_val = self._prepare_arrays(self.cv_df, val_mask, self.cv_exclude_cols)
        
        return X_train, X_val, y_train, y_val, w_train, w_val
    
    def load_and_prepare_data(self) -> Tuple:
        """Load, filter, and split the data"""
        print("\n🚀 Starting data loading process...")
        start_time = time()
                
        # Load data and get column info
        print("Reading parquet file(s)...")
        train = pl.scan_parquet(self.data_path).select(
            pl.int_range(pl.len(), dtype=pl.UInt64).alias("id") if 'id' not in pl.scan_parquet(self.data_path).columns else pl.col("id"),
            pl.all().exclude("id"),
        )        
        
        feature_cols, exclude_cols = self.get_feature_columns(train)
        
        print(f"\nFiltering data after day {EXPERIMENT['start_after_day']}...")
        train_data = train.filter(pl.col("date_id") > EXPERIMENT['start_after_day'])
        
        # Collect to DataFrame but stay in Polars
        print("Collecting DataFrame...")
        start_time_df_collect = time()
        train_df = train_data.collect(parallel=True)
        print(f"Collection completed in {time() - start_time_df_collect:.2f} seconds")
        
        print(f"\nLoaded DataFrame shape: {train_df.shape}")
        
        # Split data based on validation strategy
        print("\nSplitting data into train/validation sets...")
        if EXPERIMENT['use_cv']:
            n_folds = self.split_data_cv(train_df, exclude_cols)
            print(f"CV mode: Prepared for {n_folds} folds")
            print(f"\nData loading completed in {time() - start_time:.2f} seconds")
            return n_folds
        else:
            if VALIDATION['split_type'] == 'time':
                result = self.split_data_time(train_df, exclude_cols)
            else:
                raise ValueError(f"Unknown split type: {VALIDATION['split_type']}")
            
            print(f"Train set shape: {result[0].shape}")
            print(f"Validation set shape: {result[1].shape}\n")
            print(f"\nData loading completed in {time() - start_time:.2f} seconds")
            return result