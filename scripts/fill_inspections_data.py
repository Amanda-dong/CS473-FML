import pandas as pd
import numpy as np

def fill_inspections():
    df = pd.read_parquet('data/processed/inspections.parquet')
    
    print(f"Shape: {df.shape}")
    print(f"Nulls before: \n{df.isnull().sum()}")

    # Fill cuisine_type
    # Group by restaurant_id and find mode
    cuisine_modes = df.groupby('restaurant_id')['cuisine_type'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    df['cuisine_type'] = df['cuisine_type'].fillna(df['restaurant_id'].map(cuisine_modes))
    df['cuisine_type'] = df['cuisine_type'].fillna('Unknown')

    # Fill zipcode
    zip_modes = df.groupby('restaurant_id')['zipcode'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    df['zipcode'] = df['zipcode'].fillna(df['restaurant_id'].map(zip_modes))
    df['zipcode'] = df['zipcode'].fillna('00000')

    print(f"Nulls after: \n{df.isnull().sum()}")
    print("Sample of filled rows:")
    # Show some rows that had nulls (example)
    print(df.head())
    
    df.to_parquet('data/processed/inspections.parquet')

if __name__ == "__main__":
    fill_inspections()
