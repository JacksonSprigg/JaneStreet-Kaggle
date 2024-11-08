import os
import polars as pl
import numpy as np

from typing import Tuple, List
from config import EXPERIMENT, VALIDATION
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
        filtered = df.filter(mask)
        
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
    
    def split_data_consecutive(self, df: pl.DataFrame, exclude_cols: List[str]) -> Tuple:
        """Random consecutive days with purge windows that can overlap each other"""
        print("\n🎲 Performing consecutive-day split with purge windows...")
        
        consecutive_config = VALIDATION['consecutive']
        all_dates = df.get_column('date_id').unique().sort().to_list()
        total_days = len(all_dates)
        
        # Don't restrict valid starts - any day could be a valid start
        valid_starts = all_dates[:-consecutive_config['sequence_length']]  # Only ensure enough days for sequence
        start_dates = []
        val_and_purge_days = set()  # Track days that can't be used for new validation periods
        validation_only_days = set() # Track just validation days
        
        while len(start_dates) < consecutive_config['n_sequences'] and len(valid_starts) > 0:
            candidate = np.random.choice(valid_starts)
            candidate_idx = all_dates.index(candidate)
            
            # Check if validation period would overlap with any protected days
            validation_range = all_dates[candidate_idx:candidate_idx + consecutive_config['sequence_length']]
            if not any(day in val_and_purge_days for day in validation_range):
                start_dates.append(candidate)
                
                # Add validation days to both sets
                validation_only_days.update(validation_range)
                val_and_purge_days.update(validation_range)
                
                # Calculate and add purge days only to val_and_purge_days
                purge_start_idx = max(0, candidate_idx - consecutive_config['purge_before'])
                purge_end_idx = min(total_days - 1, 
                                candidate_idx + consecutive_config['sequence_length'] + 
                                (consecutive_config['purge_after'] - consecutive_config['sequence_length']))
                
                # Add purge windows to protected days
                val_and_purge_days.update(all_dates[purge_start_idx:candidate_idx])
                val_and_purge_days.update(all_dates[
                    candidate_idx + consecutive_config['sequence_length']:purge_end_idx + 1
                ])
                
                # Only remove validation days from valid starts
                valid_starts = [d for d in valid_starts if d not in validation_range]
        
        if len(start_dates) < consecutive_config['n_sequences']:
            print(f"\n⚠️ Warning: Could only find {len(start_dates)} non-overlapping sequences")
        
        # Create validation and purge dates
        val_dates = list(validation_only_days)
        purge_dates = [d for d in val_and_purge_days if d not in validation_only_days]
        
        # Create masks for splitting
        val_mask = pl.col('date_id').is_in(val_dates)
        purge_mask = pl.col('date_id').is_in(purge_dates)
        train_mask = ~(val_mask | purge_mask)
        
        # Get stats for printing
        n_train = df.filter(train_mask).height
        n_val = df.filter(val_mask).height
        n_purge = df.filter(purge_mask).height
        total = df.height
        
        print("\n📊 Split Statistics:")
        print(f"Total samples: {total:,}")
        print(f"Training samples: {n_train:,} ({n_train/total*100:.1f}%)")
        print(f"Validation samples: {n_val:,} ({n_val/total*100:.1f}%)")
        print(f"Purged samples: {n_purge:,} ({n_purge/total*100:.1f}%)")
        
        print("\n📅 Date Coverage:")
        print(f"Training days: {len(df.filter(train_mask).get_column('date_id').unique())}")
        print(f"Validation days: {len(set(val_dates))} "
            f"({len(start_dates)} sets of {consecutive_config['sequence_length']} days)")
        print(f"Purge days: {len(set(purge_dates))}")
        
        # Store metadata
        self.val_metadata = {
            'validation_dates': sorted(val_dates),
            'purge_dates': sorted(purge_dates),
            'n_sequences_found': len(start_dates)
        }
        
        # Convert to numpy arrays
        X_train, y_train, w_train = self._prepare_arrays(df, train_mask, exclude_cols)
        X_val, y_val, w_val = self._prepare_arrays(df, val_mask, exclude_cols)
        
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
        
        # Split data based on configured split type
        print("\nSplitting data into train/validation sets...")
        split_type = EXPERIMENT['split_type']
        if split_type == 'time':
            result = self.split_data_time(train_df, exclude_cols)
        elif split_type == 'consecutive':
            result = self.split_data_consecutive(train_df, exclude_cols)
        else:
            raise ValueError(f"Unknown split type: {split_type}")
        
        end_time = time()
        print(f"\nData loading completed in {end_time - start_time:.2f} seconds")
        print(f"Train set shape: {result[0].shape}")
        print(f"Validation set shape: {result[1].shape}\n")
        
        return result