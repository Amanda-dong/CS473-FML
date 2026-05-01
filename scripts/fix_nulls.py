import pandas as pd
import numpy as np

def audit_and_fix():
    # 1. census_nta_features.csv
    census_path = 'data/raw/census_nta_features.csv'
    df_census = pd.read_csv(census_path)
    before_census = df_census['median_household_income'].isnull().sum()
    df_census['median_household_income'] = df_census['median_household_income'].fillna(0)
    after_census = df_census['median_household_income'].isnull().sum()
    df_census.to_csv(census_path, index=False)
    print(f"census_nta_features.csv: Nulls {before_census} -> {after_census}")

    # 2. yelp_business_zones.csv
    yelp_path = 'data/raw/yelp_business_zones.csv'
    df_yelp = pd.read_csv(yelp_path)
    before_yelp_nta = df_yelp['nta'].isnull().sum()
    before_yelp_zone = df_yelp['zone_id'].isnull().sum()
    df_yelp['nta'] = df_yelp['nta'].fillna('out_of_nyc')
    df_yelp['zone_id'] = df_yelp['zone_id'].fillna('out_of_nyc')
    after_yelp_nta = df_yelp['nta'].isnull().sum()
    after_yelp_zone = df_yelp['zone_id'].isnull().sum()
    df_yelp.to_csv(yelp_path, index=False)
    print(f"yelp_business_zones.csv: NTA nulls {before_yelp_nta} -> {after_yelp_nta}, Zone nulls {before_yelp_zone} -> {after_yelp_zone}")

    # 3 & 4. gemini_labels_full.csv and gemini_labels_halal_filtered.csv
    label_files = ['data/raw/gemini_labels_full.csv', 'data/raw/gemini_labels_halal_filtered.csv']
    
    for f in label_files:
        df = pd.read_csv(f)
        before_time = df['time_key'].isnull().sum()
        before_zone = df['zone_id'].isnull().sum()
        
        # time_key fix
        df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
        df['time_key'] = df['time_key'].fillna(df['review_date'].dt.year).fillna(2023).astype(int)
        
        # zone_id fix
        df['zone_id'] = df['zone_id'].fillna('out_of_nyc')
        
        after_time = df['time_key'].isnull().sum()
        after_zone = df['zone_id'].isnull().sum()
        
        df.to_csv(f, index=False)
        print(f"{f}: Time nulls {before_time} -> {after_time}, Zone nulls {before_zone} -> {after_zone}")

if __name__ == "__main__":
    audit_and_fix()
