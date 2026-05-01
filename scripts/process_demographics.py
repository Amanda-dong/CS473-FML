import pandas as pd
from src.config.settings import settings
from src.utils.logger import logger

def process_demographics():
    excel_path = settings.RAW_DATA_DIR / "demographics.xlsx"
    output_path = settings.RAW_DATA_DIR / "nta_demographics_processed.csv"
    
    logger.info(f"Processing demographics from {excel_path}")
    
    # Mapping for 2010 NTA codes used in Yelp data to 2020 NTA codes or names
    # MN22 = Washington Square Park area in 2010
    # In 2020, this is roughly MN0202 (Greenwich Village) or MN0501 (Midtown South)
    
    try:
        df = pd.read_excel(excel_path, sheet_name=1)
        
        # Filter for NTA2020 level
        if 'GeoType' in df.columns:
            df = df[df['GeoType'] == 'NTA2020']

        nta_col = 'GeoID' if 'GeoID' in df.columns else next((c for c in df.columns if 'NTA' in str(c).upper()), None)
        pop_col = 'Pop1' if 'Pop1' in df.columns else next((c for c in df.columns if 'POP20' in str(c).upper()), None)
        name_col = 'Name' if 'Name' in df.columns else 'NTAName'

        if nta_col and pop_col:
            result = df[[nta_col, pop_col, name_col]].copy()
            result.columns = ['nta_id', 'population', 'nta_name']
            
            # Manual Mapping for common 2010 -> 2020 overlaps to ensure our Yelp NTAs match
            # This is a heuristic mapping for the "MN22" issue
            extra_mappings = [
                {'nta_id': 'MN22', 'population': df[df['GeoID'] == 'MN0202']['Pop1'].values[0] if 'MN0202' in df['GeoID'].values else 30000},
                {'nta_id': 'MN23', 'population': df[df['GeoID'] == 'MN0203']['Pop1'].values[0] if 'MN0203' in df['GeoID'].values else 25000},
                {'nta_id': 'MN25', 'population': df[df['GeoID'] == 'MN0101']['Pop1'].values[0] if 'MN0101' in df['GeoID'].values else 60000},
                {'nta_id': 'BK31', 'population': df[df['GeoID'] == 'BK0602']['Pop1'].values[0] if 'BK0602' in df['GeoID'].values else 40000},
            ]
            
            # Convert to list of dicts and concat
            mapping_df = pd.DataFrame(extra_mappings)
            
            final_result = pd.concat([result[['nta_id', 'population']], mapping_df], ignore_index=True)
            
            final_result['nta_id'] = final_result['nta_id'].astype(str)
            final_result['population'] = pd.to_numeric(final_result['population'], errors='coerce')
            final_result = final_result.dropna(subset=['population'])
            
            final_result.to_csv(output_path, index=False)
            logger.info(f"Successfully saved {len(final_result)} NTA demographic records (including manual mappings) to {output_path}")
        else:
            logger.error(f"Could not find NTA or Population columns.")
            
    except Exception as e:
        logger.error(f"Error processing Excel: {e}")

if __name__ == "__main__":
    process_demographics()
