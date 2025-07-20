import pandas as pd
import argparse

def replace_column(first_csv, second_csv, output_csv):
    # Load CSVs
    df_old = pd.read_csv(first_csv)
    df_new = pd.read_csv(second_csv)
    
    # Normalize IDs for matching
    df_old['ID_norm'] = df_old['ID'].str.replace('.npy', '', regex=False)
    df_new['ID_norm'] = df_new['ID'].str.replace('_A_', '_', regex=False)
    
    # Map and replace
    mapping = df_old.set_index('ID_norm')['1s_tri']
    df_new['1s_tri'] = df_new['ID_norm'].map(mapping).fillna(df_new['1s_tri'])
    
    df_new = df_new.drop(columns='ID_norm')
    df_new.to_csv(output_csv, index=False)
    print(f"Updated CSV saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace 1s_tri column in second CSV with first CSV values.")
    parser.add_argument("first_csv", help="Path to first CSV (ID,1s_tri).")
    parser.add_argument("second_csv", help="Path to second CSV (with 1s_tri column to replace).")
    parser.add_argument("output_csv", help="Path to output CSV.")
    
    args = parser.parse_args()
    replace_column(args.first_csv, args.second_csv, args.output_csv)